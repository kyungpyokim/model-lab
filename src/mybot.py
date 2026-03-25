import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
conversations = client.conversations.create()


class ChatRequest(BaseModel):
    message: str


class Message(BaseModel):
    role: str
    content: str


class ChatHistoryRequest(BaseModel):
    history: list[Message]


system_prompt = (
    '너는 엄격한 조선시대 선비야. 사극 톤(~하였느냐, ~이로다)을 사용하고 예법을 강조해.'
)


def mybot(user_message):
    response = client.responses.create(
        model='gpt-5-nano',
        input=[
            {
                'role': 'system',
                'content': system_prompt,
            },
            {'role': 'user', 'content': user_message},
        ],
        conversation=conversations.id,
    )
    # response = client.models.generate_content(
    #     model='gemini-2.5-flash-lite',
    #     contents=user_message,
    #     config=types.GenerateContentConfig(system_instruction=system_prompt),
    # )

    return response.output_text


def chatbot2(chat_history):
    input_list = [{'role': 'system', 'content': '당신은 친절한 챗봇입니다.'}]
    for chat in chat_history:
        role = 'assistant' if chat.role == 'ai' else 'user'

        input_list.append({'role': role, 'content': chat.content})

    response = client.responses.create(model='gpt-5-nano', input=input_list)

    return response.output_text
