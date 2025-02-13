import chainlit as cl
import json
import datetime
from dotenv import dotenv_values
from azure.identity import DefaultAzureCredential
from typing import Any, Callable, Set, Dict, List, Optional
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    FunctionTool,
    ToolSet,
    CodeInterpreterTool,
    AzureAISearchTool,
    BingGroundingTool,
    MessageTextContent
)

# from openai import util
env_vars = dotenv_values(".env")
PROJECT_CONNECTION_STRING = env_vars["PROJECT_CONNECTION_STRING"]

project_client = AIProjectClient.from_connection_string(
    credential=DefaultAzureCredential(), conn_str=PROJECT_CONNECTION_STRING
)


def fetch_current_datetime(format: Optional[str] = None) -> str:
    """
    Get the current time as a JSON string, optionally formatted.

    :param format (Optional[str]): The format in which to return the current time. Defaults to None, which uses a standard format.
    :return: The current time in JSON format.
    :rtype: str
    """
    current_time = datetime.datetime.now()

    # Use the provided format if available, else use a default format
    if format:
        time_format = format
    else:
        time_format = "%Y-%m-%d %H:%M:%S"

    time_json = json.dumps({"current_time": current_time.strftime(time_format)})
    return time_json


def fetch_weather(location: str) -> str:
    """
    Fetches the weather information for the specified location.

    :param location (str): The location to fetch weather for.
    :return: Weather information as a JSON string.
    :rtype: str
    """
    # In a real-world scenario, you'd integrate with a weather API.
    # Here, we'll mock the response.
    mock_weather_data = {
        "New York": "Sunny, 25°C",
        "London": "Cloudy, 18°C",
        "Tokyo": "Rainy, 22°C",
    }
    weather = mock_weather_data.get(
        location, "Weather data not available for this location."
    )
    weather_json = json.dumps({"weather": weather})
    return weather_json


def get_user_info(user_id: int) -> str:
    """Retrieves user information based on user ID.

    :param user_id (int): ID of the user.
    :rtype: int

    :return: User information as a JSON string.
    :rtype: str
    """
    mock_users = {
        1: {"name": "Alice", "email": "alice@example.com"},
        2: {"name": "Bob", "email": "bob@example.com"},
        3: {"name": "Charlie", "email": "charlie@example.com"},
    }
    user_info = mock_users.get(user_id, {"error": "User not found."})
    return json.dumps({"user_info": user_info})


# Statically defined user functions for fast reference
user_functions: Set[Callable[..., Any]] = {
    fetch_current_datetime,
    fetch_weather,
    get_user_info,
}

# 関数は関数でわけて定義しておく
functions = FunctionTool(user_functions)
# コードインタープリター
code_interpreter = CodeInterpreterTool()

toolset = ToolSet()
toolset.add(functions)
toolset.add(code_interpreter)

agent = project_client.agents.create_agent(
    model="gpt-4o-mini",
    name="Simple Chat Agent",
    instructions="""
    あなたは、丁寧なアシスタントです。あなたは以下の業務を遂行します。
    - 現在の時刻を回答します
    - 天気の情報を提供します
    - ユーザー情報について検索して回答します

    # 回答時のルール
    - 関数を呼び出した場合、回答の最後に改行をいれたうえで Called Function : "関数名" と表記してください

    # 制約事項
    - ユーザーからのメッセージは日本語で入力されます
    - ユーザーからのメッセージから忠実に情報を抽出し、それに基づいて応答を生成します。
    - ユーザーからのメッセージに勝手に情報を追加したり、不要な改行文字 \n を追加してはいけません

    """,
    toolset=toolset,
    headers={"x-ms-enable-preview": "true"},
)

thread = project_client.agents.create_thread()
messages = project_client.agents.list_messages(thread_id=thread.id)


@cl.on_chat_start
def start_chat():
    print(f"Created agent, agent ID: {agent.id}, thread ID: {thread.id}\n")

@cl.on_message
async def handle_message(message):
    
    message = project_client.agents.create_message(
        thread_id=thread.id,
        role="user",
        content=message.content,
    )
    run = project_client.agents.create_and_process_run(
        thread_id=thread.id, assistant_id=agent.id
    )
    print(f"Run finished with status: {run.status}")

    if run.status == "failed":
        print(f"Run failed: {run.last_error}")

    messages = project_client.agents.list_messages(thread_id=thread.id)
    for data_point in reversed(messages.data):
        last_message_content = data_point.content[-1]
        if isinstance(last_message_content, MessageTextContent):
            print(f"{data_point.role}: {last_message_content.text.value}")
            response = last_message_content.text.value

    # Chainlit の応答として返す
    await cl.Message(response).send()


if __name__ == "__main__":
    cl.run()
