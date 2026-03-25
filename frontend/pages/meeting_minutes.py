import os
import tempfile
from datetime import timedelta

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 1. 초기 설정
st.set_page_config(page_title='Audio to Markdown with Timestamps', layout='centered')
st.title('🎙️ 회의록 정리')

# API 키 입력
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


# 시간 포맷팅 함수 (초 -> MM:SS)
def format_timestamp(seconds):
    td = timedelta(seconds=seconds)
    minutes, secs = divmod(td.seconds, 60)
    return f'{minutes:02d}:{secs:02d}'


# 2. 파일 업로드 UI
uploaded_file = st.file_uploader(
    '오디오 파일 업로드', type=['mp3', 'wav', 'm4a', 'webm']
)

if uploaded_file is not None:
    file_extension = os.path.splitext(uploaded_file.name)[1]

    if st.button('변환 시작'):
        with st.spinner('분석 중...'):
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=file_extension
            ) as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_file_path = tmp_file.name

            try:
                # 3. Whisper API 호출 (verbose_json 형식 사용)
                with open(tmp_file_path, 'rb') as audio_data:
                    response = client.audio.transcriptions.create(
                        model='whisper-1',
                        file=audio_data,
                        response_format='verbose_json',  # 시간 정보를 포함하기 위해 설정
                    )

                # 4. 데이터 가공 및 마크다운 생성
                markdown_result = '### 📝 회의 내용\n\n'

                # response.segments에 시간대별 텍스트가 담겨 있음
                for segment in response.segments:
                    start_time = format_timestamp(segment.start)
                    text = segment.text.strip()
                    markdown_result += f'- **[{start_time}]** {text}\n'

                # 5. 결과 출력
                st.success('변환 완료!')
                st.markdown('---')
                st.markdown(markdown_result)
                st.markdown('---')

                st.download_button(
                    label='마크다운 파일 다운로드',
                    data=markdown_result,
                    file_name=f'{os.path.splitext(uploaded_file.name)[0]}_timestamped.md',
                    mime='text/markdown',
                )

            except Exception as e:
                st.error(f'오류가 발생했습니다: {e}')
            finally:
                if os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)
