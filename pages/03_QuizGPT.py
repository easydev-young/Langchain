import json
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st


st.set_page_config(
    page_title="QuizGPT",
    page_icon="🧠",
)

st.title("QuizGPT")

function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}


questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
        "system",
        """
        You are a useful assistant in the role of a teacher.
    
        The difficulty level of the questions ranges from 1 to 5, and this question is written at {level} difficulty.

        The questions are related to the specified {topic} and are generated at the difficulty level you choose.
    
        Each question must have four answers, three of which must be incorrect and one must be correct.         
        
        The output is in Korean.
         """,
        )
    ]
)

@st.cache_resource(show_spinner="Making quiz...")
def run_quiz_chain(topic, level):
    chain = questions_prompt | llm
    return chain.invoke({
        "level":level,
        "topic":topic        
        })


if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = ""

with st.sidebar:
    level = None
    topic = None

    openai_api_key = st.session_state["openai_api_key"]
    
    if openai_api_key:
        st.markdown(openai_api_key)
        
    else:
        openai_api_key = st.sidebar.text_input("OpenAI API KEY")
        st.session_state["openai_api_key"] = openai_api_key

    topic = st.text_input("도전하고 싶은 문제 주제는?")
    level = st.selectbox(
        "문제의 난이도를 선택하세요.",
        (
            "",
            "1",
            "2",
            "2",
            "3",
            "4",
            "5"
        ),
    )
                    



if not level:
    st.markdown(
        """
    QuizGPT에 오신 것을 환영합니다.
    """
    )

if openai_api_key and topic and level:
    llm = ChatOpenAI(
        temperature=0.1,
        model="gpt-3.5-turbo-1106",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        openai_api_key=openai_api_key,
    ).bind(
        function_call="auto",
        functions=[
            function,
        ],
    )
        
    response = run_quiz_chain(topic, level)
    response = response.additional_kwargs["function_call"]["arguments"]
    response = json.loads(response)

    num_correct = 0
    num_incorrect = 0
    quiz_count = len(response["questions"])

    with st.form("questions_form"):
        for question in response["questions"]:
            st.write(question["question"])
            value = st.radio(
                "Select an option.",
                [answer["answer"] for answer in question["answers"]],
                index=None,
            )
            if {"answer": value, "correct": True} in question["answers"]:
                st.success("✅")
                num_correct += 1
            elif value is not None:
                st.error("❌")
                num_incorrect += 1
        button = st.form_submit_button(
            disabled = num_correct == quiz_count
        )

    st.write(f"정답: {num_correct}, 오답: {num_incorrect}")
    if num_correct == quiz_count:
        st.balloons()


st.sidebar.markdown(
    """
    GitHub : https://github.com/easydev-young/Langchain/blob/Assignment16/pages/03_QuizGPT.py

    """
)