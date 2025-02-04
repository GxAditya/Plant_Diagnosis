import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# Initialize session state for navigation
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# Function to change page and rerun
def change_page(page):
    st.session_state.page = page
    st.rerun()  # Ensures the UI updates immediately

# Configure page settings
st.set_page_config(
    page_title="Plant Disease Detection System",
    layout="wide"
)

# Navigation and social links in header
header_cols = st.columns([3, 1])
with header_cols[1]:
    st.markdown("""
        <div style="text-align: right; padding: 1em;">
            <a href="https://github.com/GxAditya" target="_blank" style="margin: 0 10px;">GitHub</a>
            <a href="https://linkedin.com/in/aditya-kumar-3721012aa" target="_blank" style="margin: 0 10px;">LinkedIn</a>
            <a href="https://x.com/kaditya264?s=09" target="_blank" style="margin: 0 10px;">Twitter</a>
        </div>
    """, unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Plant Disease Detection System for Sustainable Agriculture")

# Properly update navigation state
selected_page = st.sidebar.radio("Select Page", ['Home', 'Disease Recognition'], 
                                 index=0 if st.session_state.page == 'Home' else 1)

if selected_page != st.session_state.page:
    change_page(selected_page)  # Update session state if needed

# Load plant image for home page
try:
    img = Image.open('Plant.png')
except FileNotFoundError:
    img = None

# Home Page
if st.session_state.page == 'Home':
    st.title("Welcome to Plant Disease Detection System")
    
    if img is not None:
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.image(img, use_container_width=True)
    
    st.markdown("""
    ## About the Project
    
    This innovative system uses Convolutional Neural Networks (CNN) to detect plant diseases, 
    helping farmers and agricultural professionals maintain healthy crops through early detection 
    and intervention.
    
    ### How It Works
    
    - The system utilizes a deep learning model trained on thousands of plant images
    - Our CNN architecture has been specifically optimized for plant disease recognition
    - The model can identify various plant diseases with high accuracy
    
    ### Technology Stack
    
    - **Deep Learning**: Implemented using TensorFlow and Keras
    - **Model Architecture**: Convolutional Neural Network (CNN)
    - **Frontend**: Streamlit for interactive web interface
    """)

    # Centered button for navigation
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        if st.button('Plant Diagnosis', use_container_width=True):
            change_page('Disease Recognition')

# Disease Recognition Page
elif st.session_state.page == 'Disease Recognition':
    st.title("Disease Recognition")

    # Create file uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.image(image, caption='Uploaded Image', use_container_width=True)
        
        # Add a button to trigger prediction
        if st.button('Predict Disease'):
            try:
                # Get prediction
                prediction = model_prediction(uploaded_file)
                
                # Define class names
                class_names = [
                    "Early Blight",
                    "Late Blight",
                    "Healthy"
                ]
                
                # Display prediction
                st.success(f"Prediction: {class_names[prediction]}")
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
                
    st.markdown("""
    ### Instructions:
    1. Upload a clear image of the plant leaf
    2. Click the 'Predict Disease' button
    3. Wait for the prediction result
    """)
