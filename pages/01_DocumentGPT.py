from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory
import streamlit as st
import os

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    os.makedirs(f"./.cache/files/", exist_ok=True)
    os.makedirs(f"./.cache/embeddings", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

def load_memory(_):
    return memory.load_memory_variables({})["history"]

st.title("DocumentGPT")

st.markdown(
    """
이 챗봇을 사용하여 AI에게 파일에 대해 질문하세요! 

사이드바에 파일을 업로드하세요.
"""
)

if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = ""

with st.sidebar:
    openai_api_key = st.session_state["openai_api_key"]
    
    if openai_api_key:
        st.markdown(openai_api_key)
        
        
    else:
        openai_api_key = st.sidebar.text_input("OpenAI API KEY")
        st.session_state["openai_api_key"] = openai_api_key



if openai_api_key:
    file = st.sidebar.file_uploader(
            "Upload a .txt .pdf or .docx file",
            type=["pdf", "txt", "docx"],
        )

    if file:
        llm = ChatOpenAI(
            temperature=0.1,
            streaming=True,
            callbacks=[
                ChatCallbackHandler(),
            ],
            openai_api_key=openai_api_key,
        )

        memory = ConversationBufferMemory(memory_key='history', return_messages=True)


        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
                    
                    Context: {context}
                    """,
                ),        
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )

        retriever = embed_file(file)
        send_message("I'm ready! Ask away!", "ai", save=False)
        paint_history()
        message = st.chat_input("Ask anything about your file...")
        if message:
            send_message(message, "human")
            chain = (
                {
                    "context": retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough(),
                    "history": RunnableLambda(load_memory)
                }
                | prompt
                | llm
            )
            with st.chat_message("ai"):
                response = chain.invoke(message)

    else:
        st.session_state["messages"] = []
else:
    st.session_state["openai_api_key"] =""

st.sidebar.markdown(
    """
    GitHub : https://github.com/easydev-young/Langchain/blob/Assignment15/app.py

    """
)