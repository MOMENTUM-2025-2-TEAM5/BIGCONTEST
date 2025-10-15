import os
import pandas as pd
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv

from langchain.tools import Tool
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

load_dotenv()
API_KEY = os.getenv("API_KEY")

# ====== import LLM, Embedding Model ======
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=API_KEY)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", google_api_key=API_KEY
)

# ====== import DataFrame, Agent ======
df = pd.read_excel("public/최종 데이터셋.xlsx")

pandas_agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    verbose=True,
    allow_dangerous_code=True,
)

data_analysis_tool = Tool(
    name="data_analyst",
    description=(
        "가맹점의 매출, 고객, 성과 등 숫자 데이터를 분석하고 비교할 때 사용됩니다. "
        "사용자가 '왜', '비교해줘', '어떤 특징이 있어?' 등 데이터에 기반한 질문을 하면 이 도구를 호출하세요."
    ),
    func=pandas_agent.invoke,
    handle_parsing_errors=True,
)

# ====== RAG (임시) ======
marketing_knowledge_base = [
    {
        "content": "성공사례 1: B012 가맹점은 인스타그램 스토리 투표 기능으로 고객 선호를 파악, DM 쿠폰으로 재방문율을 20% 상승시켰습니다.",
        "source": "internal_case_study_01",
    },
    {
        "content": "성공사례 2: C077 가맹점은 지역 대학 커뮤니티와 제휴, 신입생 첫 방문 20% 할인 프로모션으로 신규 고객 비중을 15%→35%로 향상시켰습니다.",
        "source": "internal_case_study_02",
    },
]

docs = [
    Document(page_content=item["content"], metadata={"source": item["source"]})
    for item in marketing_knowledge_base
]

vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})  # 관련 문서 1개만 검색

# ====== RAG Chain ======
marketing_prompt = ChatPromptTemplate.from_template(
    """
당신은 대한민국 최고의 프랜차이즈 마케팅 전략가입니다.
구체적인 마케팅 액션 아이템을 제안해야 합니다.
전략을 세울 때는 반드시 아래 [참고 자료]에 나온 성공 방식을 창의적으로 응용하세요.
추상적인 제안이 아닌, 가맹점주가 바로 실행할 수 있도록 구체적으로 설명해주세요.

[참고 자료]
{context}

[구체적인 마케팅 전략 제안]
"""
)

rag_chain = (
    {
        "context": retriever,
    }
    | marketing_prompt
    | llm
    | StrOutputParser()
)


# ====== State ======
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    next_node: str


# ====== Node ======
def data_analyst_node(state: State):
    """데이터 분석가 노드"""
    last_message = state["messages"][-1]
    tool_response = data_analysis_tool.invoke(last_message.content)
    return {
        "messages": [
            ToolMessage(content=str(tool_response), tool_call_id="data_analysis")
        ]
    }


def marketing_strategist_node(state: State):
    """마케팅 전략가 노드 (RAG 기반)"""
    # 가장 최근 data_analysis 결과 찾기
    analysis_result = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage) and msg.tool_call_id == "data_analysis":
            analysis_result = msg.content
            break

    if not analysis_result:
        return {
            "messages": [
                HumanMessage(content="분석 결과가 없어 전략을 세울 수 없습니다.")
            ]
        }

    marketing_strategy = rag_chain.invoke(analysis_result)
    return {"messages": [HumanMessage(content=marketing_strategy)]}


def supervisor_node(state: State):
    """총괄 매니저(Supervisor) 노드: LLM이 다음 단계 완전 판단"""
    messages = state["messages"]
    last_msg = messages[-1].content

    system_prompt = """
당신은 가맹점 컨설팅 AI의 총괄 매니저입니다.
사용자의 질문과 지금까지의 대화를 기반으로 다음에 호출해야 할 전문가를 결정하세요.

전문가 목록:
- data_analyst → 데이터 기반 분석(매출, 고객, 경쟁 등)을 수행.
- marketing_strategist → 분석 결과를 바탕으로 혹은 바로 마케팅 전략을 제안.
- both → 분석 후 전략까지 모두 필요.
- FINISH → 더 이상 할 일이 없을 때.

출력 형식은 반드시 아래 중 하나로만 하세요:
"data_analyst"
"marketing_strategist"
"both"
"FINISH"
그 외 다른 설명이나 문장은 절대 포함하지 마세요.
"""

    dialogue = "\n".join(
        [f"{type(msg).__name__}: {getattr(msg, 'content', '')}" for msg in messages]
    )
    supervisor_input = f"{system_prompt}\n\n[대화 기록]\n{dialogue}\n\n[마지막 사용자 질문]\n{last_msg}"

    response = llm.invoke(supervisor_input)
    next_action = response.content.strip()

    # 예외 처리
    if next_action not in ["data_analyst", "marketing_strategist", "both", "FINISH"]:
        print(f"예외 응답 감지: {next_action} -> FINISH")
        next_action = "FINISH"

    print(f"DEBUG: Supervisor 판단 -> {next_action}")
    return {"next_node": next_action}


# ====== Graph ======
workflow = StateGraph(State)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("data_analyst", data_analyst_node)
workflow.add_node("marketing_strategist", marketing_strategist_node)

workflow.set_entry_point("supervisor")

workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["next_node"],
    {
        "data_analyst": "data_analyst",
        "marketing_strategist": "marketing_strategist",
        "FINISH": END,
    },
)

workflow.add_edge("data_analyst", "supervisor")
workflow.add_edge("marketing_strategist", "supervisor")

app = workflow.compile()


# ====== main ======
if __name__ == "__main__":
    print("\n채팅 시작")

    while True:
        user_input = input("\n> 사용자 질문: ")
        if user_input.lower() in ["exit", "q"]:
            print("에이전트를 종료합니다.")
            break

        inputs = {"messages": [HumanMessage(content=user_input)]}

        print("\n--- 에이전트 작동 시작 ---")
        for output in app.stream(inputs, stream_mode="values"):
            last_msg = output["messages"][-1]

            if isinstance(last_msg, ToolMessage):
                print(f"\n[데이터 분석 결과]\n{last_msg.content}")
            else:
                print(f"\n[마케팅 전략 제안]\n{last_msg.content}")

            print("-" * 30)
