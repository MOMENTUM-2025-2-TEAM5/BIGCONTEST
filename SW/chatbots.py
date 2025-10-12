import os
import pandas as pd
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import Tool
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

load_dotenv()
API_KEY = os.getenv("API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=API_KEY)

df = pd.read_excel("data/outer join.xlsx")

pandas_agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    verbose=True,
    allow_conversational_chat=True,
    allow_dangerous_code=True, 
)

column_descriptions = {
    "ENCODED_MCT": "가맹점 고유 식별번호",
    "MCT_BSE_AR": "가맹점 주소",
    "MCT_NM": "가맹점 이름",
    "MCT_BRD_NUM": "브랜드 구분 코드",
    "MCT_SIGUNGU_NM": "가맹점이 위치한 시군구",
    "HPSN_MCT_ZCD_NM": "가맹점의 업종",
    "HPSN_MCT_BZN_CD_NM": "가맹점의 상권 구분",
    "ARE_D": "가맹점 개설일",
    "MCT_ME_D": "가맹점 폐업일",

    "TA_YM": "데이터 기준 연월",
    "MCT_OPE_MS_CN": "가맹점 운영 개월 수 구간",
    "RC_M1_SAA": "월 매출 금액 구간",
    "RC_M1_TO_UE_CT": "월 매출 건수 구간",
    "RC_M1_UE_CUS_CN": "월 유니크 고객 수 구간",
    "RC_M1_AV_NP_AT": "월 평균 객단가 구간",
    "APV_CE_RAT": "취소율 구간",
    "DLV_SAA_RAT": "배달 매출 금액 비율",
    "M1_SME_RY_SAA_RAT": "동일 업종 평균 대비 매출 금액 비율",
    "M1_SME_RY_CNT_RAT": "동일 업종 평균 대비 매출 건수 비율",
    "M12_SME_RY_SAA_PCE_RT": "동일 업종 내 매출 순위 비율",
    "M12_SME_BZN_SAA_PCE_RT": "동일 상권 내 매출 순위 비율",
    "M12_SME_RY_ME_MCT_RAT": "동일 업종 내 해지 가맹점 비중",
    "M12_SME_BZN_ME_MCT_RAT": "동일 상권 내 해지 가맹점 비중",

    "M12_MAL_1020_RAT": "남성 20대 이하 고객 비중",
    "M12_MAL_30_RAT": "남성 30대 고객 비중",
    "M12_MAL_40_RAT": "남성 40대 고객 비중",
    "M12_MAL_50_RAT": "남성 50대 고객 비중",
    "M12_MAL_60_RAT": "남성 60대 이상 고객 비중",
    "M12_FME_1020_RAT": "여성 20대 이하 고객 비중",
    "M12_FME_30_RAT": "여성 30대 고객 비중",
    "M12_FME_40_RAT": "여성 40대 고객 비중",
    "M12_FME_50_RAT": "여성 50대 고객 비중",
    "M12_FME_60_RAT": "여성 60대 이상 고객 비중",
    "MCT_UE_CLN_REU_RAT": "재방문 고객 비중",
    "MCT_UE_CLN_NEW_RAT": "신규 고객 비중",
    "RC_M1_SHC_RSD_UE_CLN_RAT": "거주지 기반 고객 비율",
    "RC_M1_SHC_WP_UE_CLN_RAT": "직장 기반 고객 비율",
    "RC_M1_SHC_FLP_UE_CLN_RAT": "유동인구 기반 고객 비율"
}

PROMPT_PREFIX = "이 데이터프레임의 컬럼은 다음과 같습니다:\n"
for col, description in column_descriptions.items():
    PROMPT_PREFIX += f"- {col}: {description}\n"
PROMPT_PREFIX += "질문 답변에 반드시 참고하세요. "


data_analysis_tool = Tool(
    name="data_analyst",
    description="가맹점의 매출, 고객, 성과 등 숫자 데이터를 분석하고 비교할 때 사용, "
                "사용자가 '왜', '비교해줘', '얼마나' 등 데이터에 기반한 질문을 하면 반드시 이 도구를 호출해야함",
    func=pandas_agent.invoke,
    handle_parsing_errors=True,
    prefix=PROMPT_PREFIX,
)


class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    next_node: str

def data_analyst_node(state: State):
    """데이터 분석 도구를 실행하고 결과를 반환하는 노드"""
    last_message = state["messages"][-1]
    
    tool_response = data_analysis_tool.invoke(last_message.content)
    
    return {"messages": [ToolMessage(content=str(tool_response), tool_call_id="data_analysis")]}


def supervisor_node(state: State):
    """사용자의 질문을 분석하여 다음 단계를 결정하는 노드"""
    
    messages = state["messages"]
    
    system_prompt = (
        "당신은 가맹점 컨설팅 AI 에이전트의 총괄 매니저입니다. "
        "사용자의 질문과 이전 대화 기록을 바탕으로 다음 단계를 결정해야 합니다.\n"
        "당신이 선택할 수 있는 전문가는 다음과 같습니다:\n"
        " - data_analyst: 데이터 분석이 필요할 때 호출합니다.\n\n"
        "만약 전문가의 분석이 끝나서 사용자에게 최종 답변을 할 수 있다고 판단되면 'FINISH'라고만 답하세요.\n"
        "그렇지 않다면, 호출할 전문가의 이름만 답하세요."
    )
    
    prompt = f"{system_prompt}\n\n[대화 기록]\n"
    for msg in messages:
        prompt += f"{msg.pretty_repr()}\n"
    
    response = llm.invoke(prompt)
    next_action = response.content.strip()

    if next_action == "FINISH":
        return {"next_node": "FINISH"}
    else:
        return {"next_node": next_action}


workflow = StateGraph(State)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("data_analyst", data_analyst_node)

workflow.set_entry_point("supervisor")

workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["next_node"],
    {
        "data_analyst": "data_analyst",
        "FINISH": END,
    }
)

workflow.add_edge("data_analyst", "supervisor")

app = workflow.compile()


if __name__ == "__main__":
    print("\n질문을 입력하세요. ")

    while True:
        user_input = input("\n> 질문: ")
        if user_input.lower() in ["exit", "q"]:
            break

        inputs = {"messages": [HumanMessage(content=user_input)]}
        
        for output in app.stream(inputs, stream_mode="values"):
            if "messages" in output and output["messages"]:
                last_message = output["messages"][-1]
                if isinstance(last_message, ToolMessage):
                    if isinstance(last_message.content, dict):
                        print(last_message.content.get('output', ''))
                    else:
                        print(last_message.content)
                else:
                    print(last_message.content)
            print("-" * 30)