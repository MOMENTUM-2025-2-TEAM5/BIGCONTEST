import os
import re
import pandas as pd
from typing import TypedDict, List
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException

# =========================
# Loading API Key
# =========================
load_dotenv()
API_KEY = os.getenv("API_KEY")

# =========================
# LLM / Embeddings
# =========================
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=API_KEY)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", google_api_key=API_KEY
)

# =========================
# Loading Data + Pandas Agent
# =========================
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


# =========================
# Initailizing RAG
# =========================
def initialize_rag(
    embeddings,
    pdf_folder: str = "public/rag_documents",
    persist_path: str = "public/vectorstores",
):
    os.makedirs(persist_path, exist_ok=True)

    faiss_index_path = os.path.join(persist_path, "index.faiss")
    if os.path.exists(faiss_index_path):
        print("Loading existing FAISS vectorstore from disk...")
        vectorstore = FAISS.load_local(
            persist_path, embeddings, allow_dangerous_deserialization=True
        )
    else:
        print("Creating new FAISS vectorstore from PDF documents...")

        all_docs: List[Document] = []
        if not os.path.exists(pdf_folder):
            raise FileNotFoundError(f"No RAG Folder: {pdf_folder}")

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
        print("Completed creating and saving FAISS vectorstore.")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return retriever


retriever = initialize_rag(embeddings)

marketing_prompt = ChatPromptTemplate.from_template(
    """
당신은 최고의 마케팅 전략가입니다.
당신은 성동구의 식당, 카페 가맹점들에 대해 마케팅 전략을 수립하는 역할을 합니다.

규칙:
- "데이터 분석"을 반드시 참고하세요.
- "참고 자료(RAG)"를 최대한 활용하세요.
- 전략에는 실행 방안(무엇/어디서/어떻게)을 포함하세요.
- 전략의 근거를 반드시 포함하세요(데이터 분석 기반 + 참고 자료 기반).
- 가능하면 어떤 문서/페이지를 참고했는지 언급하세요.

[데이터 분석]
{analysis_result}

[참고 자료]
{context}

[질문]
{question}

[전략 제안]
"""
)


def rag_chain(query: str, analysis_result: str) -> tuple[str, str, str, str]:
    docs = retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs])

    # 문서 목록
    debug_lines = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "unknown")
        preview = d.page_content[:200].replace("\n", " ")
        debug_lines.append(f"{i}) source={src} page={page} preview={preview}...")
    rag_debug = "\n".join(debug_lines)

    # RAG 디버깅 및 근거, 출처 표시
    ref_lines = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "unknown")
        excerpt = d.page_content.strip().replace("\n", " ")
        excerpt = excerpt[:250] + ("..." if len(excerpt) > 250 else "")
        ref_lines.append(f"- [{i}] {src} (p.{page})\n  └ 인용: {excerpt}")

    rag_refs = "\n".join(ref_lines)
    rag_query = query

    prompt_text = marketing_prompt.format_prompt(
        context=context, question=query, analysis_result=analysis_result
    ).to_string()

    strategy = llm.invoke(prompt_text).content
    return strategy, rag_debug, rag_query, rag_refs


# =========================
# State
# =========================
class State(TypedDict):
    query: str
    store_code: str
    user_goal: str
    analysis_result: str
    marketing_strategy: str
    rag_debug: str
    rag_query: str
    rag_refs: str
    final_answer: str


# =========================
# Nodes
# =========================
def query_analyzer_node(state: State) -> State:
    """
    사용자 질의 에이전트:
    - 가맹점코드 추출
    - user_goal 저장
    """
    q = state["query"]

    m = re.search(r"\b[A-Z0-9]{10}\b", q.upper())
    store_code = m.group(0) if m else ""

    goal = q.replace(store_code, "").strip() if store_code else q.strip()

    state["store_code"] = store_code
    state["user_goal"] = goal
    return state


def data_analyst_node(state: State) -> State:
    """
    데이터 분석 에이전트:
    - 분석/인사이트만 생성 (전략 제안 X)
    """
    query = state["query"]
    store_code = state.get("store_code", "")
    goal = state.get("user_goal", "")

    prompt = f"""
당신은 데이터 분석가입니다.
역할: 마케팅 전략을 직접 제안하지 말고, 주어진 데이터프레임들을 분석해 전략가에게 전달할 인사이트를 만드세요.

입력:
- 사용자 질문: {query}
- 가맹점구분번호(있다면): {store_code}
- 사용자의 목표/요청: {goal}

규칙:
- 가능한 한 가맹점구분번호 기준으로 해당 가게의 매출/고객/상권 특성 분석
- 필요 시 업종별/상권별 경쟁강도/특징 분석 데이터도 비교 인사이트로 활용
- 결과는 '핵심 요약 + 근거 지표/관찰 + 시사점' 구조
"""

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
            response_content = "원본 텍스트를 추출하지 못함"

    state["analysis_result"] = response_content
    return state


def marketing_strategist_node(state: State) -> State:
    """
    마케팅 전략 에이전트(RAG):
    - query(사용자 질문)로 검색
    - 분석 결과 + RAG 컨텍스트로 전략 생성
    - rag_query / rag_refs 저장
    """
    query = state["query"]
    analysis_result = state.get("analysis_result", "")

    strategy, rag_debug, rag_query, rag_refs = rag_chain(query, analysis_result)

    state["marketing_strategy"] = strategy
    state["rag_debug"] = rag_debug
    state["rag_query"] = rag_query
    state["rag_refs"] = rag_refs
    return state


def final_summarizer_node(state: State) -> State:
    """
    요약 에이전트
    """
    query = state["query"]
    analysis_result = state.get("analysis_result", "")
    marketing_strategy = state.get("marketing_strategy", "")
    rag_query = state.get("rag_query", "")
    rag_refs = state.get("rag_refs", "")

    summary_prompt = f"""
당신은 가맹점 컨설팅 결과 정리 담당입니다.
아래 내용을 바탕으로 "최종 정리"만 작성하세요.

요구:
- 짧고 핵심 위주
- 근거(데이터/문서 기반) 포인트를 3개 내외 bullet로 포함
- 사용자 질문에 직접 답하도록 구성

[사용자 질문]
{query}

[데이터 분석 결과]
{analysis_result}

[마케팅 전략 제안]
{marketing_strategy}

[최종 정리]
"""
    summary = llm.invoke(summary_prompt).content

    rag_footer = ""
    if rag_query or rag_refs:
        rag_footer = "\n\n[RAG 참고]\n" f"- 검색 질의: {rag_query}\n" f"{rag_refs}"

    state["final_answer"] = (
        f"[데이터 분석]\n{analysis_result}\n\n"
        f"[마케팅 전략 수립]\n{marketing_strategy}\n\n"
        f"[최종 정리]\n{summary}"
        f"{rag_footer}"
    )
    return state


# =========================
# Graph
# =========================
graph = StateGraph(State)

graph.add_node("query_analyzer", query_analyzer_node)
graph.add_node("data_analyst", data_analyst_node)
graph.add_node("marketing_strategist", marketing_strategist_node)
graph.add_node("final_summarizer", final_summarizer_node)

graph.add_edge(START, "query_analyzer")
graph.add_edge("query_analyzer", "data_analyst")
graph.add_edge("data_analyst", "marketing_strategist")
graph.add_edge("marketing_strategist", "final_summarizer")
graph.add_edge("final_summarizer", END)

agents = graph.compile()

# =========================
# CLI
# =========================
if __name__ == "__main__":
    print("\n채팅 시작 (exit/q 입력 시 종료)")
    while True:
        user_input = input("\n> 사용자 질문: ")
        if user_input.lower() in ["exit", "q"]:
            break

        state: State = {
            "query": user_input,
            "store_code": "",
            "user_goal": "",
            "analysis_result": "",
            "marketing_strategy": "",
            "rag_debug": "",
            "rag_query": "",
            "rag_refs": "",
            "final_answer": "",
        }

        result = agents.invoke(state)

        print("\n" + "=" * 60)
        print("🧾 [RAG DEBUG]")
        print(result.get("rag_debug", "(없음)"))

        print("\n🧭 [FINAL ANSWER]")
        print(result.get("final_answer", "(없음)"))
        print("=" * 60)
