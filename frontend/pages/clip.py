import requests
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title='CLIP Image-Text Matching',
    page_icon='🖼️',
    layout='wide',
    initial_sidebar_state='collapsed',
)

# ---------- 스타일 ----------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #111827 45%, #1e293b 100%);
        color: #e5e7eb;
        font-family: 'Inter', 'Apple SD Gothic Neo', sans-serif;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

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

    section[data-testid="stFileUploader"] {
        background: rgba(30, 41, 59, 0.55);
        border: 1px dashed rgba(96, 165, 250, 0.45);
        border-radius: 18px;
        padding: 10px;
    }

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

    .image-frame {
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 20px;
        padding: 16px;
        margin-top: 12px;
    }

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

    .result-card {
        background: rgba(30, 41, 59, 0.65);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 16px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }

    .result-label {
        color: #f8fafc;
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 6px;
    }

    .result-score {
        color: #93c5fd;
        font-size: 0.95rem;
        font-weight: 600;
    }

    .top1-box {
        background: linear-gradient(90deg, rgba(37, 99, 235, 0.18), rgba(124, 58, 237, 0.18));
        border: 1px solid rgba(96, 165, 250, 0.28);
        border-radius: 18px;
        padding: 18px;
        margin-bottom: 16px;
    }

    .top1-title {
        color: #94a3b8;
        font-size: 0.92rem;
        margin-bottom: 6px;
    }

    .top1-value {
        color: #f8fafc;
        font-size: 1.3rem;
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
        <div class="hero-title">CLIP Image-Text Matching</div>
        <div class="hero-subtitle">
            이미지를 업로드하고 여러 텍스트 후보와의 유사도를 비교하세요.<br>
            CLIP은 이미지와 텍스트를 같은 의미 공간에 매핑하여,
            어떤 문장이 이미지와 가장 잘 맞는지 판단하는 멀티모달 모델입니다.
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
            <div class="info-value">CLIP</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with info_col2:
    st.markdown(
        """
        <div class="info-box">
            <div class="info-label">Task</div>
            <div class="info-value">Image-Text Matching</div>
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

# ---------- 세션 상태 ----------
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False
if 'clip_results' not in st.session_state:
    st.session_state.clip_results = None
if 'top_label' not in st.session_state:
    st.session_state.top_label = ''
if 'top_score' not in st.session_state:
    st.session_state.top_score = 0.0

# ---------- 본문 ----------
left_col, right_col = st.columns([1, 1.2], gap='large')

with left_col:
    st.markdown(
        """
        <div class="section-card">
            <div class="card-title">설정 및 업로드</div>
            <div class="card-desc">
                비교할 텍스트 후보를 한 줄에 하나씩 입력하세요.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    candidate_text = st.text_area(
        '텍스트 후보 입력',
        value='a dog\na cat\na person',
        height=180,
        disabled=st.session_state.is_processing,
        placeholder='예:\na dog\na cat\na person riding a bicycle',
    )

    uploaded_file = st.file_uploader(
        '이미지 파일 선택',
        type=['png', 'jpg', 'jpeg'],
        disabled=st.session_state.is_processing,
    )

    def start_processing():
        if uploaded_file is None:
            st.warning('이미지를 먼저 업로드하세요.')
            return
        if not candidate_text.strip():
            st.warning('텍스트 후보를 한 줄 이상 입력하세요.')
            return

        st.session_state.is_processing = True
        st.session_state.clip_results = None

    st.button(
        '분석하기',
        disabled=st.session_state.is_processing,
        on_click=start_processing,
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.markdown("<div class='image-frame'>", unsafe_allow_html=True)
        st.image(image, caption='업로드 이미지', use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown(
        """
        <div class="section-card">
            <div class="card-title">CLIP 분석 결과</div>
            <div class="card-desc">
                입력한 텍스트 후보와 이미지 간 유사도 비교 결과입니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.is_processing:
        with st.spinner('CLIP 분석 중...'):
            try:
                uploaded_file.seek(0)
                file_bytes = uploaded_file.read()

                candidates = [
                    line.strip() for line in candidate_text.splitlines() if line.strip()
                ]

                files = {'file': (uploaded_file.name, file_bytes, uploaded_file.type)}
                data = {'classes': '\n'.join(candidates)}

                response = requests.post(
                    'http://localhost:8000/clip_image',
                    files=files,
                    data=data,
                )

                if response.status_code == 200:
                    result = response.json()

                    index = result.get('index', 0)
                    classes = result.get('classes', [])
                    probs = result.get('probs', [])

                    clip_results = [
                        {'label': label, 'score': float(score)}
                        for label, score in zip(classes, probs, strict=False)
                    ]
                    clip_results.sort(key=lambda x: x['score'], reverse=True)

                    st.session_state.top_label = (
                        classes[index] if 0 <= index < len(classes) else '-'
                    )
                    st.session_state.top_score = (
                        float(probs[index]) if 0 <= index < len(probs) else 0.0
                    )
                    st.session_state.clip_results = clip_results
                else:
                    st.error(f'서버 에러: {response.status_code}')

            except Exception as e:
                st.error(f'오류 발생: {e}')

            finally:
                st.session_state.is_processing = False
                st.rerun()

    if st.session_state.clip_results is not None:
        st.success('CLIP 분석 완료!')

        st.markdown(
            f"""
            <div class="top1-box">
                <div class="top1-title">Top 1 Prediction</div>
                <div class="top1-value">{st.session_state.top_label}</div>
                <div class="result-score">Similarity Score: {st.session_state.top_score:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for item in st.session_state.clip_results:
            label = item.get('label', '-')
            score = float(item.get('score', 0.0))

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">{label}</div>
                    <div class="result-score">Similarity Score: {score:.4f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    elif not st.session_state.is_processing:
        st.info('오른쪽 영역에 CLIP 분석 결과가 표시됩니다.')
