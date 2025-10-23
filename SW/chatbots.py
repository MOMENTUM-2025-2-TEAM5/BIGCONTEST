import os
import pandas as pd
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv

from langchain.tools import Tool
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.prompts import ChatPromptTemplate

# ====== 환경 변수 로드 ======
load_dotenv()
API_KEY = os.getenv("API_KEY")

# ====== LLM 및 Embeddings 설정 ======
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=API_KEY)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", google_api_key=API_KEY
)

# ====== 데이터 준비 및 Pandas Agent 생성 ======
df = pd.read_excel("public/최종 데이터셋.xlsx")
pandas_agent = create_pandas_dataframe_agent(
    llm=llm, df=df, verbose=True, allow_dangerous_code=True
)

data_analysis_tool = Tool(
    name="data_analyst",
    description="가맹점 매출·고객 데이터를 분석할 때 사용하는 도구",
    func=pandas_agent.invoke,
    handle_parsing_errors=True,
)


# === RAG 초기화 함수 ===
def initialize_rag(
    embeddings,
    pdf_folder: str = "public/RAG_Documents",
    persist_path: str = "public/vectorstores",
):
    os.makedirs(persist_path, exist_ok=True)

    if os.path.exists(os.path.join(persist_path, "index.faiss")):
        print("Loading existing FAISS vectorstore from disk...")
        vectorstore = FAISS.load_local(
            persist_path, embeddings, allow_dangerous_deserialization=True
        )
    else:
        print("Creating new FAISS vectorstore from PDF documents...")

        all_docs: List[Document] = []
        for file in os.listdir(pdf_folder):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(pdf_folder, file))
                docs = loader.load()
                all_docs.extend(docs)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        split_docs = text_splitter.split_documents(all_docs)

        vectorstore = FAISS.from_documents(split_docs, embeddings)
        vectorstore.save_local(persist_path)
        print("Complteted creating and saving FAISS vectorstore.")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return retriever


retriever = initialize_rag(embeddings)

marketing_prompt = ChatPromptTemplate.from_template(
    """
당신은 최고의 프랜차이즈 마케팅 전략가입니다.
구체적인 실행 가능한 전략을 제안하세요.

### 참고 자료
{context}

### 질문
{question}

### 전략 제안
"""
)


def rag_chain(query: str) -> str:
    """RAG 기반 마케팅 전략 생성"""
    docs = retriever.invoke(query)
    context = "\n".join([d.page_content for d in docs])
    prompt_text = marketing_prompt.format_prompt(
        context=context, question=query
    ).to_string()
    return llm.invoke(prompt_text).content


# ====== State 정의 ======
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    next_node: str


# ====== Node 정의 ======
def data_analyst_node(state: State) -> dict:
    """
    데이터 분석 노드
    - 마지막 사용자 메시지를 분석
    - 결과를 ToolMessage로 반환
    """
    last_msg = state["messages"][-1]
    analysis_result = data_analysis_tool.invoke(last_msg.content)
    return {
        "messages": [
            ToolMessage(content=str(analysis_result), tool_call_id="data_analysis")
        ]
    }


def marketing_strategist_node(state: State) -> dict:
    """
    마케팅 전략 노드
    - 데이터 분석 결과가 있으면 활용, 없으면 사용자 질문 그대로 사용
    - RAG 기반 전략 생성 후 ToolMessage로 반환
    """
    analysis_result = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage) and msg.tool_call_id == "data_analysis":
            analysis_result = msg.content
            break

    query = analysis_result or state["messages"][-1].content
    strategy = rag_chain(query)
    return {
        "messages": [ToolMessage(content=strategy, tool_call_id="marketing_strategy")]
    }


def supervisor_node(state: State) -> dict:
    """
    슈퍼바이저 노드
    - 사용자 요청 분석 후 필요한 노드를 순차 호출
    - 노드 호출이 필요 없으면 직접 답변
    - 모든 결과를 통합해 HumanMessage로 반환
    """
    last_msg = state["messages"][-1].content
    final_messages: List[BaseMessage] = []

    # === 1. 필요 노드 판단 (LLM 판단) ===
    system_prompt = """
당신은 가맹점 컨설팅 AI 총괄입니다.
사용자 질문을 보고 필요한 전문가를 결정하세요:
- 데이터 분석 필요 → data_analyst
- 마케팅 전략 필요 → marketing_strategist
- 둘 다 필요 → both
- 필요 없음 → FINISH
출력은 반드시 "data_analyst", "marketing_strategist", "both", "FINISH" 중 하나
"""
    dialogue = "\n".join(
        [f"{type(m).__name__}: {getattr(m,'content','')}" for m in state["messages"]]
    )
    supervisor_input = (
        f"{system_prompt}\n[대화 기록]\n{dialogue}\n[마지막 질문]\n{last_msg}"
    )
    decision = llm.invoke(supervisor_input).content.strip()
    if decision not in ["data_analyst", "marketing_strategist", "both", "FINISH"]:
        decision = "FINISH"

    print(f"DEBUG: Supervisor 판단 -> {decision}")

    # === 2. 필요한 노드 순차 호출 ===
    if decision in ["data_analyst", "both"]:
        final_messages.extend(data_analyst_node(state)["messages"])

    if decision in ["marketing_strategist", "both"]:
        final_messages.extend(marketing_strategist_node(state)["messages"])

    # === 3. FINISH일 때 간단 답변 생성 ===
    if decision == "FINISH" and not final_messages:
        # 사용자가 단순 질문(인사 등)인 경우
        reply_prompt = f"사용자의 질문에 간단히 답해주세요. 질문: {last_msg}"
        answer = llm.invoke(reply_prompt).content
        final_messages.append(HumanMessage(content=answer))

    # === 4. 최종 포맷 통합 ===
    final_output = f"사용자 질문:\n{last_msg}\n\n"
    for msg in final_messages:
        if isinstance(msg, ToolMessage):
            if msg.tool_call_id == "data_analysis":
                final_output += "[데이터 분석 결과]\n" + msg.content + "\n\n"
            elif msg.tool_call_id == "marketing_strategy":
                final_output += "[마케팅 전략 제안]\n" + msg.content + "\n\n"
        elif isinstance(msg, HumanMessage):
            final_output += msg.content + "\n\n"

    return {"messages": [HumanMessage(content=final_output)], "next_node": "FINISH"}


# ====== Graph 설정 ======
workflow = StateGraph(State)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("data_analyst", data_analyst_node)
workflow.add_node("marketing_strategist", marketing_strategist_node)

workflow.set_entry_point("supervisor")

# 슈퍼바이저에서 직접 노드 순차 호출 → next_node에 따라 흐름 제어
workflow.add_edge("data_analyst", "marketing_strategist")
workflow.add_edge("data_analyst", "supervisor")
workflow.add_edge("marketing_strategist", "supervisor")

app = workflow.compile()

# ====== Main Loop ======
if __name__ == "__main__":
    print("\n채팅 시작 (exit/q 입력 시 종료)")
    while True:
        user_input = input("\n> 사용자 질문: ")
        if user_input.lower() in ["exit", "q"]:
            break

        inputs = {"messages": [HumanMessage(content=user_input)]}
        final_messages = []

        for output in app.stream(inputs, stream_mode="values"):
            final_messages.extend(output.get("messages", []))

        # 출력 통합 (슈퍼바이저가 최종 포맷 담당)
        for msg in final_messages:
            if isinstance(msg, HumanMessage):
                print("\n> 에이전트 답변: " + msg.content)
        print("-" * 50)
