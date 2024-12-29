import streamlit as st
import os
from openai import AssistantEventHandler
import openai as client
import yfinance
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from typing_extensions import override
import json


# TA's 힌트
# 지난번 과제에서 만들었던 리서치 AI 에이전트 를 이번에는 OpenAI Assistant를 이용하여 구현해 보는 과제입니다.

# 지난 과제에서 만들었던 도구들을 OpenAI Assistant에서 사용할 수 있도록 바꿔줍니다. 
# (공식 문서의 Function calling 부분을 참고하세요.) 
# https://platform.openai.com/docs/assistants/tools/function-calling

# 사용자의 OpenAI API 키를 입력 받고, Assistant, Thread 를 만들고 Thread에 사용자의 Message를 입력한 다음 Run을 만들어 실행하세요. 
# (공식 문서의 QuickStart 를 보면 전체적인 흐름을 파악할 수 있습니다.)
# https://platform.openai.com/docs/assistants/quickstart

# Assistant, Thread, Run 이 화면 리렌더링 이후에도 유지될 수 있도록 session_state 을 사용합니다. 
# (Session State 공식 문서 참고)
# https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state

# TIP : 주피터 노트북에서 할 때는 간단하지만 Streamlit과 함께 인터페이스에 적용하는 것은 생각보다 복잡합니다. 
# 강의의 내용을 충분히 숙지한 후, 
# OpenAI Assistant 공식문서를 천천히 읽어보면서 진행하세요.
# https://platform.openai.com/docs/assistants/overview

# TIP 2 : OpenAI Dashboard의 Threads 탭을 보면 생성된 Thread의 목록과 Thread 내에 있는 Message를 볼 수 있어서 개발 중에 유용하게 사용할 수 있습니다.
# 기본적으로 Threads 탭이 비활성화 되어 있어서 설정에서 활성화해주어야 합니다.
# 먼저, OpenAI Organization Settings 로 접속합니다.
# Features and capabilities > Threads 에서 옵션을 Visible to organization owners 혹은 Visible to everyone 로 설정하고 밑에 Save 까지 눌러주세요.
# 이제 OpenAI Dashboard 왼쪽 탭에 활성화된 Threads 탭을 선택하면 스레드 목록이 보입니다.

st.title("ASSIGNMENT 10")

st.markdown(
    """
    Welcome to AssistantAPI.
            
    Start by writing the OPENAI API KEY and ask a question about a company and our Assistant will do the research for you.
"""
)

with st.sidebar:
    openai_api_key = st.text_input(
        label="OPENAI API KEY",
    )
    
ASSISTANT_NAME = "Investor Assistant"

class EventHandler(AssistantEventHandler):

    message = ""

    @override
    def on_text_created(self, text) -> None:
        self.message_box = st.empty()

    def on_text_delta(self, delta, snapshot):
        self.message += delta.value
        self.message_box.markdown(self.message.replace("$", "\$"))

    def on_event(self, event):

        if event.event == "thread.run.requires_action":
            submit_tool_outputs(event.data.id, event.data.thread_id)

# Tools
def get_ticker(inputs):
    ddg = DuckDuckGoSearchAPIWrapper()
    company_name = inputs["company_name"]
    return ddg.run(f"Ticker symbol of {company_name}")


def get_income_statement(inputs):
    ticker = inputs["ticker"]
    stock = yfinance.Ticker(ticker)
    return json.dumps(stock.income_stmt.to_json())


def get_balance_sheet(inputs):
    ticker = inputs["ticker"]
    stock = yfinance.Ticker(ticker)
    return json.dumps(stock.balance_sheet.to_json())


def get_daily_stock_performance(inputs):
    ticker = inputs["ticker"]
    stock = yfinance.Ticker(ticker)
    return json.dumps(stock.history(period="3mo").to_json())


functions_map = {
    "get_ticker": get_ticker,
    "get_income_statement": get_income_statement,
    "get_balance_sheet": get_balance_sheet,
    "get_daily_stock_performance": get_daily_stock_performance,
}

functions = [
    {
        "type": "function",
        "function": {
            "name": "get_ticker",
            "description": "Given the name of a company returns its ticker symbol",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "The name of the company",
                    }
                },
                "required": ["company_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_income_statement",
            "description": "Given a ticker symbol (i.e AAPL) returns the company's income statement.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol of the company",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_balance_sheet",
            "description": "Given a ticker symbol (i.e AAPL) returns the company's balance sheet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol of the company",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_daily_stock_performance",
            "description": "Given a ticker symbol (i.e AAPL) returns the performance of the stock for the last 100 days.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol of the company",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
]

#### Utilities
def get_run(run_id, thread_id):
    return client.beta.threads.runs.retrieve(
        run_id=run_id,
        thread_id=thread_id,
    )


def send_message(thread_id, content):
    return client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=content,
    )


def get_messages(thread_id):
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    messages = list(messages)
    messages.reverse()
    return messages


def get_tool_outputs(run_id, thread_id):
    run = get_run(run_id, thread_id)
    outputs = []
    for action in run.required_action.submit_tool_outputs.tool_calls:
        action_id = action.id
        function = action.function
        print(f"Calling function: {function.name} with arg {function.arguments}")
        outputs.append(
            {
                "output": functions_map[function.name](json.loads(function.arguments)),
                "tool_call_id": action_id,
            }
        )
    return outputs


def submit_tool_outputs(run_id, thread_id):
    outputs = get_tool_outputs(run_id, thread_id)
    with client.beta.threads.runs.submit_tool_outputs_stream(
        run_id=run_id,
        thread_id=thread_id,
        tool_outputs=outputs,
        event_handler=EventHandler(),
    ) as stream:
        stream.until_done()


def insert_message(message, role):
    with st.chat_message(role):
        st.markdown(message)


def paint_history(thread_id):
    messages = get_messages(thread_id)
    for message in messages:
        insert_message(
            message.content[0].text.value,
            message.role,
        )
        
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
    
    if "assistant" not in st.session_state:
        assistants = client.beta.assistants.list(limit=10)
        for a in assistants:
            if a.name == ASSISTANT_NAME:
                assistant = client.beta.assistants.retrieve(a.id)
                break
        else:
            assistant = client.beta.assistants.create(
                name=ASSISTANT_NAME,
                instructions="You help users do research on the given query using search engines. You give users the summarization of the information you got.",
                model="gpt-4o-mini",
                tools=functions,
            )
        thread = client.beta.threads.create()
        st.session_state["assistant"] = assistant
        st.session_state["thread"] = thread
    else:
        assistant = st.session_state["assistant"]
        thread = st.session_state["thread"]

    paint_history(thread.id)
    content = st.chat_input("What do you want to search?")
    if content:
        send_message(thread.id, content)
        insert_message(content, "user")

        with st.chat_message("assistant"):
            with client.beta.threads.runs.stream(
                thread_id=thread.id,
                assistant_id=assistant.id,
                event_handler=EventHandler(),
            ) as stream:
                stream.until_done()
    