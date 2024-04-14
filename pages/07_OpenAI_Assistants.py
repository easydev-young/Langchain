from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.utilities import WikipediaAPIWrapper
from openai import OpenAI
import streamlit as st
import re
import json
import time

def DuckDuckGoSearch(inputs):
    ddg = DuckDuckGoSearchAPIWrapper()
    query = inputs["query"]
    return ddg.run(query)


def WikipediaSearch(inputs):
    wikipedia = WikipediaAPIWrapper()
    query = inputs["query"]
    return wikipedia.run(query)

functions_map = {
    "DuckDuckGoSearch": DuckDuckGoSearch,
    "WikipediaSearch": WikipediaSearch,
}

functions = [
    {
        "type": "function",
        "function": {
            "name": "DuckDuckGoSearch",
            "description": "Use this tool to search on DuckDuckGo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query you will search for",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "WikipediaSearch",
            "description": "Use this tool to search on Wikipedia.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query you will search for",
                    }
                },
                "required": ["query"],
            },
        },
    },
]

@st.cache_resource(show_spinner="create assistant...")
def create_assistant():
    return client.beta.assistants.create(
        name="Research Assistant",
        instructions="The agent attempts to search DuckDuckGo and Wikipedia, extracts the content, and then provides the research to you.",
        model="gpt-3.5-turbo",
        tools=functions,
    )
    
if "message" not in st.session_state:
    st.session_state["messages"] = []
        
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

def save_message_display(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

def send_message_display(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message_display(message, role)

def paint_history():
    for message in st.session_state["messages"]:
        send_message_display(
            message["message"],
            message["role"],
            save=False,
        )

openai_api_key = get_openai_api_key()
client = OpenAI(api_key=openai_api_key if openai_api_key else "_",)

@st.cache_resource(show_spinner="create thread...")
def create_thread(query):
    thread = client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": query,
            }
        ]
    )
    return thread

@st.cache_resource(show_spinner="create run...")
def create_run(thread_id, assistant_id):
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
    )
    return run

def get_run(run_id, thread_id):
    return client.beta.threads.runs.retrieve(
        run_id=run_id,
        thread_id=thread_id,
    )

def get_run_status(run_id, thread_id):
    return get_run(run_id, thread_id).status

def send_message(thread_id, content):
    return client.beta.threads.messages.create(
        thread_id=thread_id, role="user", content=content
    )

def get_messages(thread_id):
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    messages = list(messages)
    messages.reverse()
    threads_messages = ""
    for message in messages:
        threads_messages += f"\n\n{message.content[0].text.value}"

    return threads_messages

def get_tool_outputs(run_id, thread_id):
    run = get_run(run_id, thread_id)
    outputs = []
    for action in run.required_action.submit_tool_outputs.tool_calls:
        action_id = action.id
        function = action.function
        #print(f"Calling function: {function.name} with arg {function.arguments}")
        outputs.append(
            {
                "output": functions_map[function.name](json.loads(function.arguments)),
                "tool_call_id": action_id,
            }
        )
    return outputs

def submit_tool_outputs(run_id, thread_id):
    outpus = get_tool_outputs(run_id, thread_id)
    return client.beta.threads.runs.submit_tool_outputs(
        run_id=run_id,
        thread_id=thread_id,
        tool_outputs=outpus,
    )

st.set_page_config(
    page_title="OpenAI Assistants",
    page_icon="🤖🌟",
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
        send_message_display("I'm ready! Ask away!", "ai", save=False)
        paint_history()
        message = st.chat_input("What do you want research about?")
        if message:
            send_message_display(message, "human")
            assistant = create_assistant()
            send_message_display("assistant creation completed", "ai", save=False)
            thread = create_thread(message)
            send_message_display("Thread creation completed", "ai", save=False)
            run = create_run(thread.id, assistant.id)
            send_message_display("Run creation completed", "ai", save=False)

            threads_messages = ""
            with st.chat_message("ai"):
                 while True:
                    run_status = get_run_status(run.id, thread.id)

                    if run_status == "requires_action":
                        submit_tool_outputs(run.id, thread.id)

                    if run_status in ("expired", "cancelled", "failed"):
                        st.error(f"오류가 발생했습니다. (오류 메시지: {run_status})")
                        break

                    if run_status == "completed":
                        threads_messages = get_messages(thread.id)                        
                        break

                    time.sleep(2)

            if threads_messages:
                send_message_display(threads_messages, "ai", save=False)

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
    GitHub : https://github.com/easydev-young/Langchain/blob/Assignment19/pages/07_OpenAI_Assistants.py

    """
)