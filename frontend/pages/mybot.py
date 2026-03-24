import requests
import streamlit as st

st.title('챗봇 만들기')

url = 'http://localhost:8080/chat'

# 저장소 만들기
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# 히스토리 출력
for chat in st.session_state['chat_history']:
    st.chat_message(chat['role']).markdown(chat['content'])

user_input = st.chat_input(placeholder='메세지를 입력하세요')

if user_input:
    # 챗 표시
    st.chat_message('user').markdown(user_input)

    # answer = "안녕하세요" # Fastapi /chat 요청으로 대체해보세요
    # st.chat_message("ai").markdown(answer)

    data = {
        'message': user_input,
    }
    try:
        answer = requests.post(url, json=data).json()['text']
    except Exception as e:
        answer = f'LLM에 접근할 수 없는 상태입니다. {e}'

    st.chat_message('ai').markdown(f'**이도령**: {answer}')

    # 히스토리 저장
    st.session_state['chat_history'].extend(
        [
            {'role': 'user', 'content': user_input},
            {'role': 'ai', 'content': f'**이도령**: {answer}'},
        ]
    )
