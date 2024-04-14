from langchain_community.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st
import re

url = "https://developers.cloudflare.com/assets/sitemap.xml"

def get_openai_api_key():
    if "openai_api_key" not in st.session_state:
        st.session_state["openai_api_key"] = ""
        return ""
        
    openai_api_key = st.session_state["openai_api_key"]
    return openai_api_key

def set_openai_api_key(openai_api_key):
    st.session_state["openai_api_key"] = openai_api_key

def remove_openai_api_key():
    st.session_state["openai_api_key"] = ""

def validate_openai_api_key(openai_api_key):
  regex = r"^sk-[a-zA-Z0-9]{40,60}$"
  return bool(re.match(regex, openai_api_key))

def is_openai_api_key_error(e):
  e_str = str(e).lower()
  return bool(re.search(r"(api)(_|-|\s)(key)", e_str))

openai_api_key = get_openai_api_key()

llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    openai_api_key=openai_api_key if openai_api_key else "_",
)

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    Question: {question}
"""
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            If the score is 1 or higher, be sure to cite the source and return the source as it is in your answer, do not change it.

            If your score is 0, never cite the source.
            
            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("CloseSearch Submit Blog", "")
    )


@st.cache_resource(show_spinner="Loading Cloudflare...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        url,
        filter_urls=[
            r"https:\/\/developers.cloudflare.com\/(ai-gateway|vectorize|workers-ai).*"
        ],
        parsing_function=parse_page,
    )
    loader.requests_per_second = 2
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings(
        openai_api_key=openai_api_key
    ))
    return vector_store.as_retriever()

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🌐",
)

with st.sidebar:    
    if openai_api_key:
        if st.button("Remove Key"):
            remove_openai_api_key()
            st.rerun()
    else:
        api_key = st.sidebar.text_input("OpenAI API KEY")
        
        if api_key:
            if validate_openai_api_key(api_key):
                set_openai_api_key(api_key)
                with st.empty():
                    st.rerun()
            else:
                with st.empty():
                    st.error("키가 유효하지 않습니다.")

if openai_api_key:
    try:
        retriever = load_website(url)
        query = st.text_input("Ask a question to the website.")
        if query:
            chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(get_answers)
                | RunnableLambda(choose_answer)
            )
            result = chain.invoke(query)
            st.markdown(result.content.replace("$", "\$"))
    except Exception as e:
        if is_openai_api_key_error(e):
            st.error("API_KEY 를 확인해 주세요.")
        else:
            st.error(f"알 수 없는 오류가 발생했습니다. (오류 메시지: {e})")
else:
    st.markdown("""
    <style>
    @keyframes blinker {
        50% { opacity: 0; }
    }
    .blinking-arrow {
        animation: blinker 1s linear infinite;
        font-size: 24px;
        color: orange;
        transform: rotate(180deg);
    }
    </style>
    <div>
        <span class="blinking-arrow">←</span>
        <span style="margin-left: 8px;">Please provide an <span style="color: blue;">OpenAI API Key</span> on the sidebar.</span>
    </div>
    """, unsafe_allow_html=True)
        

st.sidebar.markdown(
     """
    GitHub : https://github.com/easydev-young/Langchain/blob/Assignment17/pages/04_SiteGPT.py

    """
)