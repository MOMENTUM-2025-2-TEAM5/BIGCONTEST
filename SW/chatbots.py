import os
import re
import pandas as pd
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException

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
df_storetype_sales_competition = pd.read_csv(
    "public/상권_업종별_매출_경쟁강도_분석결과리스트.csv"
)
df_area_sales_competition = pd.read_csv(
    "public/상권별_매출_경쟁강도_분석결과리스트.csv"
)
df_storetype_features = pd.read_csv("public/상권_업종별_특징분석결과_리스트.csv")
df_area_features = pd.read_csv("public/상권별_특징분석결과_리스트.csv")

pandas_agent = create_pandas_dataframe_agent(
    llm=llm,
    df=[
        df,
        df_storetype_sales_competition,
        df_area_sales_competition,
        df_storetype_features,
        df_area_features,
    ],
    verbose=True,
    allow_dangerous_code=True,
    agent_executor_kwargs={"handle_parsing_errors": True},
)


# === RAG 초기화 함수 ===
def initialize_rag(
    embeddings,
    pdf_folder: str = "public/rag_documents",
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
당신은 최고의 마케팅 전략가입니다.
당신은 성동구의 식당, 카페 가맹점들에 대해 마케팅 전략을 수립하는 역할을 합니다.
사용자 질문에 대한 효과적인 마케팅 전략을 제안하세요. 
"데이터 분석"이 있다면 반드시 참고하여 답변하고 참고 자료를 최대한 활용하세요. 
제안한 마케팅 전략에는 근거를 반드시 포함하세요. 

[데이터 분석]
{analysis_result}

[참고 자료]
{context}

[질문]
{question}

[전략 제안]
"""
)


def rag_chain(query: str, analysis_result: str) -> str:
    """RAG 기반 마케팅 전략 생성"""
    docs = retriever.invoke(query)
    context = "\n".join([d.page_content for d in docs])
    prompt_text = marketing_prompt.format_prompt(
        context=context, question=query, analysis_result=analysis_result
    ).to_string()

    return llm.invoke(prompt_text)


# ====== State 정의 ======
class State(TypedDict):
    query: str
    analysis_result: str
    marketing_strategy: str
    chain_both: bool
    next_node: str
    final_answer: str


# ====== Node 정의 ======
def data_analyst_node(state: State) -> State:
    """
    데이터 분석 노드
    - 열자리의 가맹점구분번호를 기준으로 분석 수행할 것 (가맹점구분번호 예시: 01EDD97C21 등)
    - 필요한 데이터프레임을 선택하고 마지막 사용자 메시지의 의도를 파악하여 분석 수행
    - 결과를 ToolMessage로 반환
    - 필요시 여러 데이터프레임을 분석 후 결과를 정리
    """
    query = state["query"]
    prompt = f"query: {query}\n당신은 데이터 분석가 입니다. 사용자 쿼리에 맞게 주어진 데이터프레임들을 분석하여 마케팅 전략 수립가에게 분석 결과 및 인사이트를 넘겨주어야 합니다. 전략 제안은 당신의 역할이 아닙니다. 데이터 분석만 진행하세요. "
    try:
        analysis_result = pandas_agent.invoke(prompt)
        response_content = analysis_result["output"]
    except (OutputParserException, ValueError) as e:
        error_message = str(e)

        match = re.search(
            r"Could not parse LLM output: `(.*)`", error_message, re.DOTALL
        )

        if match:
            response_content = match.group(1).strip()
        else:
            response_content = (
                "에이전트 파싱 오류가 발생했으나 원본 텍스트를 추출하지 못했습니다."
            )
    state["analysis_result"] = response_content
    return state


def marketing_strategist_node(state: State) -> State:
    """
    마케팅 전략 노드
    - RAG 기반 전략 생성
    - 반드시 근거가 포함된 전략 제안
    - 어떤 문서의 어떤 부분을 참고했는지 포함
    """
    query = state["query"]
    prompt = f"query: {query}\n당신은 최고의 마케팅 전략가입니다. 사용자 질문에 대한 효과적인 마케팅 전략을 제안하세요. 데이터 분석값이 있다면 반드시 참고하여 답변하고 참고 자료를 최대한 활용하세요. 제안한 마케팅 전략에는 근거를 반드시 포함하세요."
    analysis_result = state.get("analysis_result", "")
    strategy = rag_chain(prompt, analysis_result).content
    state["marketing_strategy"] = strategy
    return state


def supervisor_node(state: State) -> State:
    """
    슈퍼바이저 노드
    - 사용자 질문 분석 후 다음에 실행할 노드를 결정
    - 그래프 흐름 제어
    """
    query = state["query"]
    llm_prompt = """
당신은 성동구 가맹점 마케팅 컨설팅 및 데이터 분석 AI들의 총괄입니다.
가맹점주들의 요구에 맞는 최적의 전문가를 배정하세요. 
사용자 질문을 보고 필요한 전문가를 결정하세요:
- 데이터 분석 필요 → data_analyst
    - 성동구 전체 가맹점 데이터(매출, 고객 정보 등), 상권 및 업종 별 경쟁 강도와 특징 분석 데이터 보유 중
- 마케팅 전략 필요 → marketing_strategist
- 둘 다 필요 → both
- 둘 다 필요 없음 → FINISH
출력은 반드시 다음 중 하나: data_analyst, marketing_strategist, both, FINISH
"""
    decision_input = f"{llm_prompt}\n[질문]\n{query}"
    decision = llm.invoke(decision_input).content.strip()

    if decision not in ["data_analyst", "marketing_strategist", "both", "FINISH"]:
        decision = "FINISH"

    print(f"DEBUG: Supervisor 판단 -> {decision}")

    if decision == "data_analyst":
        state["next_node"] = "data_analyst"
    elif decision == "marketing_strategist":
        state["next_node"] = "marketing_strategist"
    elif decision == "both":
        state["next_node"] = "data_analyst"
        state["chain_both"] = True
    elif decision == "FINISH":
        analysis = state.get("analysis_result", "")
        marketing = state.get("marketing_strategy", "")

    return state


def summerize_final_answer_node(state: State) -> dict:
    """
    최종 요약 답변 생성 노드
    - 데이터 분석 결과와 마케팅 전략을 종합하여 최종 답변 생성
    """
    query = state["query"]
    analysis_result = state.get("analysis_result", "")
    marketing_strategy = state.get("marketing_strategy", "")

    if analysis_result or marketing_strategy:
        summary_prompt = f"""
당신은 가맹점 컨설팅 AI돌 중 결과 정리 역할입니다.
다음은 지금까지 수행된 분석 및 전략 제안 결과입니다.
사용자 질의에 맞는 답변을 생성하세요.
이를 참고하여 핵심만 요약해 사용자에게 한눈에 보이게 정리하세요.
데이터 분석 내용과 마케팅 전략의 근거를 반드시 명시해야 합니다.

[사용자 질문]
{query}

[데이터 분석 결과]
{analysis_result}

[마케팅 전략 제안]
{marketing_strategy}

[요약 및 최종 제안]
"""
        final_answer = llm.invoke(summary_prompt).content
        state["final_answer"] = final_answer
    return state


# ====== Graph 설정 ======
workflow = StateGraph(State)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("data_analyst", data_analyst_node)
workflow.add_node("marketing_strategist", marketing_strategist_node)
workflow.add_node("final_summarizer", summerize_final_answer_node)

workflow.add_edge(START, "supervisor")

workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["next_node"],
    {
        "data_analyst": "data_analyst",
        "marketing_strategist": "marketing_strategist",
        "both": "data_analyst",
        "FINISH": "final_summarizer",
    },
)

workflow.add_conditional_edges(
    "data_analyst",
    lambda state: "marketing_strategist" if state.get("chain_both") else "FINISH",
    {
        "marketing_strategist": "marketing_strategist",
        "FINISH": "final_summarizer",
    },
)

workflow.add_edge("marketing_strategist", "final_summarizer")
workflow.add_edge("final_summarizer", END)

app = workflow.compile()

# ====== Main Loop ======
if __name__ == "__main__":
    print("\n채팅 시작 (exit/q 입력 시 종료)")
    while True:
        user_input = input("\n> 사용자 질문: ")
        if user_input.lower() in ["exit", "q"]:
            break

        state = {
            "query": user_input,
            "analysis_result": "",
            "marketing_strategy": "",
            "chain_both": False,
            "next_node": "",
            "final_answer": "",
        }

        result = app.invoke(state)

        # ====== 결과 출력 ======
        print("\n" + "=" * 60)
        print("📊 [데이터 분석 결과]")
        print(result.get("analysis_result", "(없음)"))

        print("\n💡 [마케팅 전략 제안]")
        print(result.get("marketing_strategy", "(없음)"))

        print("\n🧭 [최종 요약 답변]")
        print(result.get("final_answer", "(없음)"))
        print("=" * 60)
