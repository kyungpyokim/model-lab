import time

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

from lib.utils.path import model_path

# --- 1. 페이지 설정 ---
st.set_page_config(
    page_title='VisionAI | Object Detection Dashboard',
    page_icon='🎯',
    layout='wide',
    initial_sidebar_state='expanded',
)

# --- 2. 커스텀 CSS ---
st.markdown(
    """
    <style>
    :root {
        --bg: #0b1020;
        --panel: rgba(17, 24, 39, 0.72);
        --panel-strong: rgba(15, 23, 42, 0.92);
        --border: rgba(255, 255, 255, 0.08);
        --text: #e5eefb;
        --muted: #94a3b8;
        --primary: #7c3aed;
        --primary-2: #3b82f6;
        --success: #22c55e;
        --danger: #ef4444;
        --shadow: 0 12px 40px rgba(0, 0, 0, 0.28);
        --radius: 22px;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(59, 130, 246, 0.16), transparent 28%),
            radial-gradient(circle at top right, rgba(124, 58, 237, 0.16), transparent 25%),
            linear-gradient(180deg, #0b1020 0%, #0f172a 100%);
        color: var(--text);
        font-family: "Inter", "Pretendard", "Apple SD Gothic Neo", sans-serif;
    }

    /* 기본 여백 */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1380px;
    }

    /* 사이드바 */
    section[data-testid="stSidebar"] {
        background: rgba(10, 15, 28, 0.95);
        border-right: 1px solid var(--border);
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.2rem;
    }

    /* 상단 히어로 */
    .hero-card {
        background: linear-gradient(
            135deg,
            rgba(59, 130, 246, 0.18) 0%,
            rgba(124, 58, 237, 0.18) 100%
        );
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 26px;
        padding: 30px 32px;
        box-shadow: var(--shadow);
        backdrop-filter: blur(14px);
        margin-bottom: 20px;
    }

    .hero-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.08);
        color: #cbd5e1;
        font-size: 0.84rem;
        font-weight: 600;
        margin-bottom: 14px;
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 0;
        letter-spacing: -1.2px;
        color: white;
    }

    .hero-title .accent {
        background: linear-gradient(90deg, #8b5cf6 0%, #60a5fa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-desc {
        margin-top: 12px;
        color: #cbd5e1;
        font-size: 1rem;
        line-height: 1.7;
    }

    /* 카드 */
    .glass-card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 22px;
        box-shadow: var(--shadow);
        backdrop-filter: blur(16px);
    }

    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 14px;
        letter-spacing: -0.3px;
    }

    .section-subtle {
        color: var(--muted);
        font-size: 0.92rem;
        margin-bottom: 10px;
    }

    /* 버튼 */
    .stButton > button {
        width: 100%;
        height: 3.3rem;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.08);
        background: linear-gradient(90deg, #4f46e5 0%, #2563eb 100%);
        color: white;
        font-weight: 700;
        font-size: 0.98rem;
        box-shadow: 0 10px 24px rgba(37, 99, 235, 0.28);
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 14px 28px rgba(37, 99, 235, 0.34);
        border-color: rgba(255,255,255,0.14);
    }

    .stButton > button:active {
        transform: translateY(0px);
    }

    /* 업로더 */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.03);
        border: 1px dashed rgba(255,255,255,0.14);
        border-radius: 16px;
        padding: 8px;
    }

    /* 슬라이더 */
    .stSlider {
        padding-top: 0.2rem;
        padding-bottom: 0.6rem;
    }

    /* 메트릭 */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 18px;
        padding: 16px 18px;
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted) !important;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 800;
        letter-spacing: -0.8px;
    }

    /* 데이터프레임 */
    .stDataFrame {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.08);
    }

    /* 알림 박스 */
    div[data-testid="stInfo"] {
        background: rgba(59, 130, 246, 0.10);
        border: 1px solid rgba(96, 165, 250, 0.22);
        color: #dbeafe;
        border-radius: 16px;
    }

    div[data-testid="stSuccess"] {
        background: rgba(34, 197, 94, 0.10);
        border: 1px solid rgba(34, 197, 94, 0.24);
        color: #dcfce7;
        border-radius: 16px;
    }

    div[data-testid="stError"] {
        border-radius: 16px;
    }

    /* expander */
    .streamlit-expanderHeader {
        font-weight: 700;
        color: #e2e8f0;
    }

    /* 구분선 */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(
            90deg,
            rgba(255,255,255,0) 0%,
            rgba(255,255,255,0.12) 50%,
            rgba(255,255,255,0) 100%
        );
        margin: 1rem 0 1.4rem 0;
    }

    /* 작은 설명 카드 */
    .mini-note {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 18px;
        padding: 18px;
        color: #cbd5e1;
        line-height: 1.7;
    }

    .mini-note strong {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- 3. 세션 상태 ---
if 'results' not in st.session_state:
    st.session_state.results = None
if 'result_image' not in st.session_state:
    st.session_state.result_image = None
if 'inference_time' not in st.session_state:
    st.session_state.inference_time = None


# --- 4. 모델 로드 ---
@st.cache_resource(show_spinner='딥러닝 모델을 불러오는 중...')
def load_model():
    return YOLO(model_path('yolo26n.pt'))


try:
    model = load_model()
except Exception as e:
    st.error(f'❌ 모델 파일을 찾을 수 없습니다. 경로를 확인해주세요.\\n\\n{e}')
    st.stop()

# --- 5. 사이드바 ---
with st.sidebar:
    st.markdown(
        """
        <div style="margin-bottom: 18px;">
            <div style="font-size: 1.2rem; font-weight: 800; color: white;">VisionAI</div>
            <div style="color: #94a3b8; font-size: 0.92rem; margin-top: 4px;">
                Intelligent Object Detection
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('---')

    st.subheader('📁 데이터 입력')
    uploaded_file = st.file_uploader(
        '이미지를 업로드하세요',
        type=['jpg', 'jpeg', 'png'],
        label_visibility='visible',
    )

    st.subheader('⚙️ 모델 설정')
    conf_value = st.slider(
        '신뢰도 임계값',
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help='모델이 객체라고 판단하는 최소 신뢰도입니다.',
    )

    iou_value = st.slider(
        '중복 제거 IoU',
        min_value=0.0,
        max_value=1.0,
        value=0.45,
        step=0.05,
        help='겹치는 박스를 얼마나 제거할지 결정합니다.',
    )

    st.markdown('---')
    st.caption('Developed by AI Lab Team')
    st.caption('Powered by Ultralytics YOLO')

# --- 6. 상단 헤더 ---
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-badge">AI Vision Dashboard</div>
        <h1 class="hero-title">
            <span class="accent">Object Detection</span><br/>
            for Fast Visual Analysis
        </h1>
        <div class="hero-desc">
            이미지를 업로드하고 객체 탐지 결과를 시각적으로 확인해보세요.
            탐지된 객체 수, 클래스 분포, 신뢰도 점수를 한 번에 확인할 수 있습니다.
        </div>
        <div class="hero-desc">
            <p>Object Detection은 이미지를 입력받아 사물을 추론하는 모델입니다.</p>
            <p>Object Detection으로 어떤 사물인지 판별 할수 있습니다.</p>
            <p>Object Detection을 학습할 때에는 노이즈가 섞이지 않게 라벨링을 주의해야합니다.</p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- 7. 메인 화면 ---
if uploaded_file:
    input_image = Image.open(uploaded_file).convert('RGB')

    col1, col2 = st.columns([1.65, 1], gap='large')

    with col1:
        st.markdown(
            '<div class="section-title">🖼️ Analysis View</div>', unsafe_allow_html=True
        )
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        image_placeholder = st.empty()

        if st.session_state.result_image is not None:
            image_placeholder.image(
                st.session_state.result_image,
                caption=f'최근 분석 결과 · {st.session_state.inference_time:.3f}초',
                use_container_width=True,
            )
        else:
            image_placeholder.image(
                input_image,
                caption='분석 준비 완료',
                use_container_width=True,
            )

        st.write('')
        detect_button = st.button('AI 분석 실행 🚀', key='detect_button')

        if detect_button:
            with st.spinner('AI 모델이 이미지를 분석 중입니다...'):
                start_time = time.time()
                results = model.predict(input_image, conf=conf_value, iou=iou_value)
                end_time = time.time()

                plotted = results[0].plot()
                result_image = Image.fromarray(plotted[:, :, ::-1])

                st.session_state.results = results
                st.session_state.result_image = result_image
                st.session_state.inference_time = end_time - start_time

                image_placeholder.image(
                    result_image,
                    caption=f'분석 완료 · {st.session_state.inference_time:.3f}초',
                    use_container_width=True,
                )

            st.success('✅ 이미지 분석이 완료되었습니다!')

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(
            '<div class="section-title">📊 Detection Summary</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        if st.session_state.results is not None:
            results = st.session_state.results
            boxes = results[0].boxes
            num_objects = len(boxes)

            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric('Detected Objects', f'{num_objects}개')
            with metric_col2:
                st.metric(
                    'Inference Time',
                    f'{st.session_state.inference_time:.3f}s'
                    if st.session_state.inference_time is not None
                    else '-',
                )

            st.write('')

            if num_objects > 0:
                class_names = [model.names[int(box.cls[0])] for box in boxes]
                counts = pd.Series(class_names).value_counts().reset_index()
                counts.columns = ['Object', 'Count']

                st.markdown(
                    '<div class="section-subtle">탐지된 객체 클래스 분포</div>',
                    unsafe_allow_html=True,
                )
                st.dataframe(counts, hide_index=True, use_container_width=True)

                with st.expander('신뢰도(Confidence) 점수 보기'):
                    conf_rows = []
                    for box in boxes:
                        label = model.names[int(box.cls[0])]
                        conf = float(box.conf[0])
                        conf_rows.append(
                            {
                                'Object': label,
                                'Confidence': f'{conf:.2%}',
                            }
                        )
                    st.dataframe(
                        pd.DataFrame(conf_rows),
                        hide_index=True,
                        use_container_width=True,
                    )

            else:
                st.info(
                    '감지된 객체가 없습니다. 다른 이미지나 설정값으로 다시 시도해보세요.'
                )

        else:
            st.info(
                "왼쪽 이미지 영역에서 **'AI 분석 실행 🚀'** 버튼을 누르면 결과가 표시됩니다."
            )
            st.markdown(
                """
                <div class="mini-note">
                    <strong>추천 사용 흐름</strong><br/>
                    1. 이미지를 업로드합니다.<br/>
                    2. Confidence / IoU 값을 조절합니다.<br/>
                    3. 분석을 실행하고 결과 이미지와 통계를 확인합니다.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('</div>', unsafe_allow_html=True)

else:
    col1, col2 = st.columns([1.1, 1], gap='large')

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">시작하기</div>', unsafe_allow_html=True)
        st.info(
            '👈 왼쪽 사이드바에서 이미지를 업로드하면 바로 분석을 시작할 수 있습니다.'
        )

        st.markdown(
            """
            <div class="mini-note">
                <strong>사용 방법</strong><br/>
                1. 사이드바에서 이미지를 업로드합니다.<br/>
                2. 신뢰도 임계값과 IoU를 조절합니다.<br/>
                3. <strong>AI 분석 실행 🚀</strong> 버튼을 누릅니다.<br/>
                4. 결과 이미지와 객체 통계를 확인합니다.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Preview</div>', unsafe_allow_html=True)
        st.image(
            'https://images.unsplash.com/photo-1620288627223-53302f4e8c74?q=80&w=1200&auto=format&fit=crop',
            caption='Vision-powered analysis interface',
            use_container_width=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)
