import json
import uuid

import boto3
import requests
import streamlit as st
from markdownify import markdownify as md
from playwright.sync_api import sync_playwright

### 定数定義

agent_id = "I3JWB4PYW5"
agent_alias_id = "TSTALIASID"

tavily_api_key = "tvly-*****"

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
session_id = st.session_state.session_id

### ツール定義


def web_search(search_query: str) -> dict:

    response = requests.post(
        "https://api.tavily.com/search",
        json={"query": search_query, "api_key": tavily_api_key},
    )

    return response.json()


def url_crawl(url: str) -> str:

    with sync_playwright() as p:
        browser = p.chromium.launch(
            args=[
                "--single-process",
                "--no-zygote",
                "--no-sandbox",
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--headless=new",
            ]
        )

        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        context = browser.new_context(user_agent=user_agent)
        page = context.new_page()

        response = page.goto(url=url)

        content = page.content()
        title = page.title()

        browser.close()

    return {
        "status": response.status,
        "content": md(content),
        "title": title,
    }


tools = {
    "action-group-1": {"web_search": web_search},
    "action-group-2": {"url_crawl": url_crawl},
}


### UI

st.title("Bedrock Agent")


if "messages" not in st.session_state:
    st.session_state.messages = []
messages = st.session_state.messages

for message in messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def event_loop(response):

    completion = ""
    sessionState = {}

    for event in response.get("completion"):
        if "trace" in event:
            trace = event["trace"]

            with st.chat_message("assistant"):
                with st.expander("Trace", expanded=False):
                    st.write(trace)

        if "returnControl" in event:
            returnControl: dict = event["returnControl"]

            invocation_id: str = returnControl["invocationId"]
            invocation_inputs: list = returnControl["invocationInputs"]

            with st.chat_message("assistant"):
                with st.expander("Return Control", expanded=False):
                    st.write(returnControl)

            sessionState["invocationId"] = invocation_id

            for input in invocation_inputs:

                function_invocation_input = input["functionInvocationInput"]
                actionGroup = function_invocation_input["actionGroup"]
                actionInvocationType = function_invocation_input["actionInvocationType"]
                function = function_invocation_input["function"]

                parameters = function_invocation_input["parameters"]

                parameter = {}
                for p in parameters:
                    name = p["name"]
                    type = p["type"]
                    value = p["value"]

                    parameter[name] = value

                result = tools[actionGroup][function](**parameter)

                returnControlInvocationResult = {
                    "functionResult": {
                        "actionGroup": actionGroup,
                        "function": function,
                        "responseBody": {"string": {"body": json.dumps(result)}},
                    }
                }

                with st.chat_message("user"):
                    with st.expander("Return Control Result", expanded=False):
                        st.write(returnControlInvocationResult)

                sessionState["returnControlInvocationResults"] = [
                    returnControlInvocationResult
                ]

        if "chunk" in event:
            chunk = event["chunk"]
            completion = completion + chunk["bytes"].decode()

    return completion, sessionState


if prompt := st.chat_input():

    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.container(border=True):

        client = boto3.client("bedrock-agent-runtime")

        response = client.invoke_agent(
            agentId=agent_id,
            agentAliasId=agent_alias_id,
            sessionId=session_id,
            enableTrace=True,
            inputText=prompt,
        )

        completion, sessionState = event_loop(response)

        while len(sessionState) > 0:

            response = client.invoke_agent(
                agentId=agent_id,
                agentAliasId=agent_alias_id,
                sessionId=session_id,
                enableTrace=True,
                sessionState=sessionState,
            )

            completion, sessionState = event_loop(response)

    messages.append({"role": "assistant", "content": completion})
    with st.chat_message("assistant"):
        st.write(completion)
