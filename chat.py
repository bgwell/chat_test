# pip install streamlit

import streamlit as st

from dotenv import load_dotenv
from chat_llm import get_ai_response

# 환경변수를 불러옴
load_dotenv()

st.set_page_config(page_title="비지웰 챗봇", page_icon="🐱‍👤")

st.title("🐱‍👤 비지웰 챗봇")
st.caption("비지웰 취업규칙 문의")

if 'message_list' not in st.session_state:
  st.session_state.message_list = []

for message in st.session_state.message_list:
  with st.chat_message(message["role"]):
    st.write(message["content"])

if user_question := st.chat_input(placeholder="비지웰 취업규칙에 관련된 궁금한 내용들을 말씀해 주세요") :
  with st.chat_message("user") :
    st.write(user_question)
  st.session_state.message_list.append({"role": "user", "content": user_question})

  with st.spinner("답변을 생성하는 중입니다"):
    ai_response = get_ai_response(user_question)
    with st.chat_message("ai") :
      ai_message = st.write_stream(ai_response)
      st.session_state.message_list.append({"role": "ai", "content": ai_message})