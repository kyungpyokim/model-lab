import io

import requests
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title='SAM3 Segmentation',
    page_icon='🧩',
    layout='wide',
    initial_sidebar_state='collapsed',
)

# ---------- 스타일 ----------
st.markdown(
    """
    <style>
    /* 전체 배경 */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #111827 45%, #1e293b 100%);
        color: #e5e7eb;
        font-family: 'Inter', 'Apple SD Gothic Neo', sans-serif;
    }

    /* 메인 컨테이너 여백 */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* 상단 헤더 박스 */
    .hero-card {
        background: rgba(17, 24, 39, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 24px;
        padding: 32px;
        backdrop-filter: blur(12px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
        margin-bottom: 24px;
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0.4rem;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #cbd5e1;
        line-height: 1.7;
    }

    .badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 999px;
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        color: white;
        font-size: 0.85rem;
        font-weight: 700;
        margin-bottom: 14px;
    }

    /* 카드 */
    .section-card {
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 22px;
        padding: 22px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
        height: 100%;
    }

    .card-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.35rem;
    }

    .card-desc {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 1rem;
    }

    /* 파일 업로드 */
    section[data-testid="stFileUploader"] {
        background: rgba(30, 41, 59, 0.55);
        border: 1px dashed rgba(96, 165, 250, 0.45);
        border-radius: 18px;
        padding: 10px;
    }

    /* 버튼 */
    .stButton > button {
        width: 100%;
        height: 3.2rem;
        border: none;
        border-radius: 14px;
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        color: white;
        font-size: 1rem;
        font-weight: 700;
        transition: 0.2s ease-in-out;
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.25);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 24px rgba(124, 58, 237, 0.32);
    }

    /* 이미지 박스 느낌 */
    .image-frame {
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 20px;
        padding: 16px;
        margin-top: 12px;
    }

    /* 메트릭 카드 */
    .info-box {
        background: rgba(30, 41, 59, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 18px;
        padding: 18px;
        text-align: center;
    }

    .info-label {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-bottom: 6px;
    }

    .info-value {
        color: #f8fafc;
        font-size: 1.2rem;
        font-weight: 800;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- 헤더 ----------
st.markdown(
    """
    <div class="hero-card">
        <div class="badge">AI Vision Dashboard</div>
        <div class="hero-title">SAM3 Segmentation</div>
        <div class="hero-subtitle">
            이미지를 업로드하고 세그멘테이션 결과를 확인하세요.<br>
            세그멘테이션이란? 물체감지를 픽셀 단위로 검출하는 기술<br>
            입력 이미지와 추출 결과를 한 화면에서 직관적으로 비교할 수 있도록 구성했습니다.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- 상단 정보 ----------
info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:
    st.markdown(
        """
        <div class="info-box">
            <div class="info-label">Model</div>
            <div class="info-value">SAM3</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with info_col2:
    st.markdown(
        """
        <div class="info-box">
            <div class="info-label">Task</div>
            <div class="info-value">Segmentation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with info_col3:
    st.markdown(
        """
        <div class="info-box">
            <div class="info-label">Status</div>
            <div class="info-value">Ready</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

# 1. 세션 상태 초기화 (이미지 저장용 변수 추가)
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False
if 'result_image' not in st.session_state:
    st.session_state.result_image = None
if 'last_text' not in st.session_state:
    st.session_state.last_text = ''

# ---------- 본문 ----------
left_col, right_col = st.columns([1, 1.2], gap='large')

with left_col:
    st.markdown(
        '<div class="section-card"><div class="card-title">설정 및 업로드</div></div>',
        unsafe_allow_html=True,
    )

    target_text = st.text_input(
        '찾을 객체 입력', value='bus', disabled=st.session_state.is_processing
    )

    uploaded_file = st.file_uploader(
        '이미지 파일 선택',
        type=['png', 'jpg', 'jpeg'],
        disabled=st.session_state.is_processing,
    )

    # 버튼 클릭 시 처리 시작 함수
    def start_processing():
        if uploaded_file is not None:
            st.session_state.is_processing = True
            st.session_state.result_image = None  # 새 추론 시작 시 이전 결과 삭제

    run_button = st.button(
        '추출하기', disabled=st.session_state.is_processing, on_click=start_processing
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.markdown("<div class='image-frame'>", unsafe_allow_html=True)
        st.image(image, caption='업로드 이미지', use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown(
        '<div class="section-card"><div class="card-title">Segmentation 결과</div></div>',
        unsafe_allow_html=True,
    )

    # 상황 1: 현재 추론 중일 때
    if st.session_state.is_processing:
        with st.spinner(f"'{target_text}' 추출 중..."):
            try:
                uploaded_file.seek(0)
                file_bytes = uploaded_file.read()
                files = {'file': (uploaded_file.name, file_bytes, uploaded_file.type)}
                data = {'text_prompt': target_text}

                response = requests.post(
                    'http://localhost:8000/segment_image', files=files, data=data
                )

                if response.status_code == 200:
                    # 결과를 세션 상태에 저장 (이미지 객체로 변환해서 저장)
                    st.session_state.result_image = Image.open(
                        io.BytesIO(response.content)
                    )
                    st.session_state.last_text = target_text
                else:
                    st.error(f'서버 에러: {response.status_code}')

            except Exception as e:
                st.error(f'오류 발생: {e}')

            finally:
                st.session_state.is_processing = False
                st.rerun()  # 상태를 반영하기 위해 다시 실행

    # 상황 2: 추론된 이미지가 세션에 저장되어 있을 때 표시
    if st.session_state.result_image is not None:
        st.success(f"'{st.session_state.last_text}' 추출 완료!")
        st.markdown("<div class='image-frame'>", unsafe_allow_html=True)
        st.image(
            st.session_state.result_image,
            caption=f'Result: {st.session_state.last_text}',
            use_container_width=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # 상황 3: 아무것도 없을 때
    elif not st.session_state.is_processing:
        st.info('오른쪽 영역에 결과 이미지가 표시됩니다.')
