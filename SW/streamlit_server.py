import streamlit as st
from langchain_core.messages import HumanMessage
from chatbots import app  # chatbots.py에서 app을 가져옵니다.
import time  # 타이핑 효과를 위한 time 모듈 import

st.set_page_config(page_title="가맹점 컨설팅 챗봇", layout="centered")
st.title("💬 프랜차이즈 컨설팅 챗봇")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 채팅 기록 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("질문을 입력하세요.")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    state = {
        "query": user_input,
        "analysis_result": "",
        "marketing_strategy": "",
        "chain_both": False,
        "next_node": "",
        "final_answer": "",
    }

    with st.chat_message("assistant"):

        with st.spinner("컨설팅 AI가 분석 및 전략을 수립 중입니다..."):
            final_state = app.invoke(state)

        final_answer_text = final_state.get(
            "final_answer", "최종 답변을 생성하는 데 실패했습니다."
        )

        def stream_text_generator(text_string: str):
            """문자열을 받아, 단어별로 time.sleep을 주며 yield하는 생성기"""

            for line in text_string.split("\n"):
                for word in line.split(" "):
                    yield word + 
                    time.sleep(0.03)
                yield "\n"

        st.write_stream(stream_text_generator(final_answer_text))

        st.session_state.messages.append(
            {"role": "assistant", "content": final_answer_text}
        )
