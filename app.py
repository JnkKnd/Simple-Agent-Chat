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

# 環境変数の読み込み
env_vars = dotenv_values(".env")
PROJECT_CONNECTION_STRING = env_vars["PROJECT_CONNECTION_STRING"]
if not PROJECT_CONNECTION_STRING:
    raise ValueError("'.env' に PROJECT_CONNECTION_STRING が設定されていません。")


project_client = AIProjectClient.from_connection_string(
    credential=DefaultAzureCredential(), conn_str=PROJECT_CONNECTION_STRING
)


def fetch_current_datetime(format: Optional[str] = None) -> str:
    """
    Get the current time as a JSON string, optionally formatted.

    :param format (Optional[str]): 時刻のフォーマット。指定がない場合は "%Y-%m-%d %H:%M:%S" を使用
    :return: Json フォーマットの現在時刻
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
    指定された場所の天気情報を JSON 文字列で返します

    :param location (str): 天気情報を取得する場所
    :return: JSON 文字列の天気の情報
    :rtype: str
    """
    # In a real-world scenario, you'd integrate with a weather API.
    # Here, we'll mock the response.
    mock_weather_data = {
        "New York": "Sunny, 25°C",
        "London": "Cloudy, 18°C",
        "東京": "くもり, 9°C",
    }
    weather = mock_weather_data.get(
        location, "Weather data not available for this location."
    )
    weather_json = json.dumps({"weather": weather})
    return weather_json


def get_user_info(user_id: int) -> str:
    """
    ユーザーIDに基づいてユーザー情報を JSON 文字列で返す

    :param user_id (int): ユーザーのID
    :rtype: int

    :return: JSON 文字列のユーザー情報.
    :rtype: str
    """
    mock_users = {
        1: {"name": "Alice", "email": "alice@example.com"},
        2: {"name": "Bob", "email": "bob@example.com"},
        3: {"name": "Charlie", "email": "charlie@example.com"},
    }
    user_info = mock_users.get(user_id, {"error": "User not found."})
    return json.dumps({"user_info": user_info})


# ユーザ関数のセットアップ
user_functions: Set[Callable[..., Any]] = {
    fetch_current_datetime,
    fetch_weather,
    get_user_info,
}


functions = FunctionTool(user_functions)
# コードインタープリター
code_interpreter = CodeInterpreterTool()

toolset = ToolSet()
toolset.add(functions)
toolset.add(code_interpreter)


@cl.on_chat_start
def start_chat():
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

    # セッションにエージェントIDとスレッドIDを保存
    cl.user_session.set("agent_id", agent.id)
    cl.user_session.set("thread_id", thread.id)

    print(f"Created agent, agent ID: {agent.id}, thread ID: {thread.id}\n")

@cl.on_message
async def handle_message(user_message):
    # セッションからエージェントIDとスレッドIDを取得
    agent_id = cl.user_session.get("agent_id")
    thread_id = cl.user_session.get("thread_id")
    if not agent_id or not thread_id:
        await cl.Message(
            "内部エラー: エージェントまたはスレッドが見つかりません。"
        ).send()
        return

    # ユーザーメッセージをスレッドに登録
    message = project_client.agents.create_message(
        thread_id=thread_id,
        role="user",
        content=user_message.content,
    )
    run = project_client.agents.create_and_process_run(
        thread_id=thread_id, assistant_id = agent_id
    )
    print(f"Run finished with status: {run.status}")

    if run.status == "failed":
        print(f"Run failed: {run.last_error}")

    # 最新のテキストレスポンスを取得
    messages = project_client.agents.list_messages(thread_id=thread_id)
    response = None
    for data_point in reversed(messages.data):
        last_message_content = data_point.content[-1]
        if isinstance(last_message_content, MessageTextContent):
            print(f"{data_point.role}: {last_message_content.text.value}")
            response = last_message_content.text.value
    if response is None:
        response = "エージェントからの応答が得られませんでした。"
    await cl.Message(response).send()

@cl.on_chat_end
def end_chat():
    # セッションからエージェントIDとスレッドIDを取得してクリーンアップ
    agent_id = cl.user_session.get("agent_id")
    thread_id = cl.user_session.get("thread_id")
    if thread_id:
        project_client.agents.delete_thread(thread_id=thread_id)
    if agent_id:
        project_client.agents.delete_agent(assistant_id=agent_id)
    print(f"Deleted agent, agent ID: {agent_id}, thread ID: {thread_id}")


if __name__ == "__main__":
    cl.run()
