import streamlit as st

st.set_page_config(
    page_title="FullstackGPT Home",
    page_icon="🧙‍♂️",
)

st.markdown(
    """
# 파이썬 🐍 스터디 - 풀스택 GPT:

- <a href="/DocumentGPT" target="_self">📃 DocumentGPT</a>
- <a href="/QuizGPT" target="_self">🧠 QuizGPT</a>
- <a href="/SiteGPT" target="_self">🌐 SiteGPT</a>
- <a href="/OpenAI_Assistants" target="_self">🤖🌟 OpenAI Assistants</a>
""",
    unsafe_allow_html=True
)