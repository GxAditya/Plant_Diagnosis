import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)


if "page" not in st.session_state:
    st.session_state.page = "Home"


def change_page(page):
    st.session_state.page = page
    st.rerun()


st.set_page_config(
    page_title="Plant Disease Detection System", layout="wide", page_icon="🌿"
)

st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,700&family=Playfair+Display:wght@600;700&display=swap');
        
        :root {
            --primary: #1a472a;
            --primary-light: #2d5a3d;
            --accent: #4ade80;
            --bg-dark: #0f1a12;
            --bg-card: #142318;
            --text-main: #e8f5e9;
            --text-muted: #81c784;
        }
        
        .stApp {
            background: linear-gradient(145deg, var(--bg-dark) 0%, #0a120c 100%);
            font-family: 'DM Sans', sans-serif;
        }
        
        .main-title {
            font-family: 'Playfair Display', serif;
            font-size: 3rem;
            font-weight: 700;
            color: var(--text-main);
            text-align: center;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, var(--text-main) 0%, var(--accent) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .subtitle {
            text-align: center;
            color: var(--text-muted);
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }
        
        .feature-card {
            background: var(--bg-card);
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid rgba(74, 222, 128, 0.15);
            transition: all 0.3s ease;
            min-height: 180px;
        }
        
        .feature-card:hover {
            border-color: var(--accent);
            transform: translateY(-4px);
            box-shadow: 0 12px 40px rgba(74, 222, 128, 0.1);
        }
        
        .feature-icon {
            width: 48px;
            height: 48px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 12px;
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%);
            color: var(--accent);
            font-size: 1.25rem;
            margin-bottom: 1rem;
        }
        
        .feature-title {
            font-weight: 600;
            color: var(--text-main);
            margin-bottom: 0.5rem;
        }
        
        .feature-desc {
            color: var(--text-muted);
            font-size: 0.9rem;
            line-height: 1.6;
        }
        
        .section-title {
            font-family: 'Playfair Display', serif;
            font-size: 1.75rem;
            font-weight: 600;
            color: var(--text-main);
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        
        .section-title i {
            color: var(--accent);
        }
        
        .tech-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            background: var(--bg-card);
            border-radius: 8px;
            border: 1px solid rgba(74, 222, 128, 0.2);
            color: var(--text-muted);
            font-size: 0.9rem;
        }
        
        .tech-badge i {
            color: var(--accent);
        }
        
        .prediction-result {
            background: linear-gradient(135deg, var(--bg-card) 0%, var(--primary) 100%);
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
            border: 1px solid var(--accent);
        }
        
        .prediction-label {
            color: var(--text-muted);
            font-size: 1rem;
            margin-bottom: 0.5rem;
        }
        
        .prediction-value {
            font-family: 'Playfair Display', serif;
            font-size: 2rem;
            font-weight: 700;
            color: var(--accent);
        }
        
        .instructions-list {
            list-style: none;
            padding: 0;
        }
        
        .instructions-list li {
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 1rem;
            background: var(--bg-card);
            border-radius: 12px;
            margin-bottom: 0.75rem;
            border: 1px solid rgba(74, 222, 128, 0.1);
        }
        
        .step-number {
            width: 32px;
            height: 32px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: var(--primary);
            border-radius: 8px;
            color: var(--accent);
            font-weight: 600;
        }
        
        .step-text {
            color: var(--text-muted);
        }
        
        .stSidebar {
            background: var(--bg-card) !important;
        }
        
        .stRadio > div {
            background: transparent !important;
        }
        
        div[data-testid="stRadio"] label {
            color: var(--text-muted) !important;
            padding: 0.75rem 1rem !important;
            border-radius: 10px !important;
            margin-bottom: 0.5rem !important;
            transition: all 0.3s ease !important;
        }
        
        div[data-testid="stRadio"] label:hover {
            background: rgba(74, 222, 128, 0.1) !important;
        }
        
        div[data-testid="stRadio"] label:has(input:checked) {
            background: var(--primary) !important;
            color: var(--accent) !important;
        }
        
        .stButton > button {
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%) !important;
            color: var(--accent) !important;
            border: 1px solid rgba(74, 222, 128, 0.3) !important;
            border-radius: 12px !important;
            padding: 0.75rem 1.5rem !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 30px rgba(74, 222, 128, 0.3) !important;
            border-color: var(--accent) !important;
        }
        
        .stFileUploader {
            background: var(--bg-card);
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid rgba(74, 222, 128, 0.15);
        }
        
        div[data-testid="stFileUploader"] label {
            color: var(--text-muted) !important;
        }
        
        div[data-testid="stSuccess"] {
            background: linear-gradient(135deg, var(--bg-card) 0%, var(--primary) 100%) !important;
            border: 1px solid var(--accent) !important;
            border-radius: 12px !important;
        }
        
        div[data-testid="stSuccess"] > div {
            color: var(--accent) !important;
        }
        
        div[data-testid="stError"] {
            background: #2d1a1a !important;
            border: 1px solid #ef4444 !important;
            border-radius: 12px !important;
        }
        
        .centered-img {
            display: block;
            margin: 0 auto;
            width: 180px;
            height: 180px;
            object-fit: contain;
            border-radius: 20px;
            box-shadow: 0 8px 40px rgba(74, 222, 128, 0.15);
            background: var(--bg-card);
            padding: 1rem;
            border: 1px solid rgba(74, 222, 128, 0.2);
        }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div style="display: flex; justify-content: space-between; align-items: center; padding: 1rem 2rem; border-bottom: 1px solid rgba(74, 222, 128, 0.15); margin-bottom: 2rem;">
    <div style="display: flex; align-items: center; gap: 1rem;">
        <span style="font-family: 'Playfair Display', serif; font-size: 1.5rem; font-weight: 700; color: #e8f5e9;">PlantDoc</span>
    </div>
    <div style="display: flex; gap: 0.75rem;">
        <a href="https://github.com/GxAditya" target="_blank" style="display: inline-flex; align-items: center; justify-content: center; width: 40px; height: 40px; border-radius: 10px; background: #142318; color: #81c784; text-decoration: none; border: 1px solid rgba(74, 222, 128, 0.2); transition: all 0.3s ease;" title="GitHub">
            <i class="fab fa-github"></i>
        </a>
        <a href="https://linkedin.com/in/aditya-kumar-3721012aa" target="_blank" style="display: inline-flex; align-items: center; justify-content: center; width: 40px; height: 40px; border-radius: 10px; background: #142318; color: #81c784; text-decoration: none; border: 1px solid rgba(74, 222, 128, 0.2); transition: all 0.3s ease;" title="LinkedIn">
            <i class="fab fa-linkedin-in"></i>
        </a>
        <a href="https://x.com/kaditya264?s=09" target="_blank" style="display: inline-flex; align-items: center; justify-content: center; width: 40px; height: 40px; border-radius: 10px; background: #142318; color: #81c784; text-decoration: none; border: 1px solid rgba(74, 222, 128, 0.2); transition: all 0.3s ease;" title="X (Twitter)">
            <i class="fab fa-x"></i>
        </a>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.sidebar.markdown("")
with st.sidebar.expander("☰ Navigation", expanded=False):
    selected_page = st.radio(
        "Go to",
        ["Home", "Disease Recognition"],
        index=0 if st.session_state.page == "Home" else 1,
        label_visibility="collapsed",
    )
    if selected_page != st.session_state.page:
        change_page(selected_page)

try:
    img = Image.open("Plant.png")
except FileNotFoundError:
    img = None

if st.session_state.page == "Home":
    st.markdown(
        '<h1 class="main-title">Plant Disease Detection</h1>', unsafe_allow_html=True
    )
    st.markdown(
        '<p class="subtitle">AI-Powered Disease Recognition for Sustainable Agriculture</p>',
        unsafe_allow_html=True,
    )

    if img is not None:
        import base64
        from io import BytesIO

        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        st.markdown(
            f'<img src="data:image/png;base64,{img_base64}" class="centered-img" alt="Plant">',
            unsafe_allow_html=True,
        )

    st.markdown("<br><br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(
            "Start Diagnosis",
            icon="🌿",
            use_container_width=True,
        ):
            change_page("Disease Recognition")

    st.markdown("")
    st.markdown("")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.markdown(
            """
        <div class="feature-card">
            <div class="feature-icon"><i class="fas fa-brain"></i></div>
            <div class="feature-title">Deep Learning</div>
            <div class="feature-desc">Convolutional Neural Networks trained on thousands of plant images for accurate disease detection.</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="feature-card">
            <div class="feature-icon"><i class="fas fa-bolt"></i></div>
            <div class="feature-title">Instant Results</div>
            <div class="feature-desc">Get disease predictions in seconds. Upload an image and receive immediate analysis.</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
        <div class="feature-card">
            <div class="feature-icon"><i class="fas fa-leaf"></i></div>
            <div class="feature-title">Sustainable</div>
            <div class="feature-desc">Help farmers detect diseases early and reduce crop loss through timely intervention.</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown("")

    st.markdown(
        """
    <h2 class="section-title"><i class="fas fa-cogs"></i> Technology Stack</h2>
    """,
        unsafe_allow_html=True,
    )

    tech_cols = st.columns(4)
    with tech_cols[0]:
        st.markdown(
            '<div class="tech-badge"><i class="fab fa-python"></i> TensorFlow</div>',
            unsafe_allow_html=True,
        )
    with tech_cols[1]:
        st.markdown(
            '<div class="tech-badge"><i class="fas fa-network-wired"></i> CNN Architecture</div>',
            unsafe_allow_html=True,
        )
    with tech_cols[2]:
        st.markdown(
            '<div class="tech-badge"><i class="fas fa-globe"></i> Streamlit</div>',
            unsafe_allow_html=True,
        )
    with tech_cols[3]:
        st.markdown(
            '<div class="tech-badge"><i class="fas fa-brain"></i> Keras</div>',
            unsafe_allow_html=True,
        )

elif st.session_state.page == "Disease Recognition":
    st.markdown(
        """
    <h2 class="section-title"><i class="fas fa-microscope"></i> Disease Recognition</h2>
    """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload a plant leaf image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(
                image, caption="Uploaded Image", use_container_width=True, clamp=True
            )

        st.markdown("")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button(
                "Predict Disease",
                icon="🔍",
                use_container_width=True,
            ):
                try:
                    prediction = model_prediction(uploaded_file)
                    class_names = ["Early Blight", "Late Blight", "Healthy"]

                    st.markdown(
                        f"""
                    <div class="prediction-result">
                        <div class="prediction-label">Prediction Result</div>
                        <div class="prediction-value">{class_names[prediction]}</div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

    st.markdown("")
    st.markdown("")

    st.markdown(
        """
    <h3 class="section-title" style="font-size: 1.25rem;"><i class="fas fa-info-circle"></i> Instructions</h3>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <ul class="instructions-list">
        <li><span class="step-number">1</span><span class="step-text">Upload a clear image of the plant leaf</span></li>
        <li><span class="step-number">2</span><span class="step-text">Click the 'Predict Disease' button</span></li>
        <li><span class="step-number">3</span><span class="step-text">View the prediction result instantly</span></li>
    </ul>
    """,
        unsafe_allow_html=True,
    )
