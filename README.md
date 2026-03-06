<div align="center">
  <img src="./Plant.png" width="120" height="120" alt="Plant Disease Detection">
</div>

<h1 align="center">Plant Disease Detection System</h1>

<p align="center">
  A deep learning-based system for detecting potato plant diseases using Convolutional Neural Networks (CNN). This project helps farmers and agricultural professionals identify plant diseases early, enabling timely intervention and crop protection.
</p>

---

## Overview

This system uses a trained CNN model to classify potato plant diseases from leaf images. The model can identify three conditions:

- Early Blight
- Late Blight
- Healthy

The application provides an intuitive web interface built with Streamlit for easy image upload and disease prediction.

## Features

- Real-time plant disease detection from uploaded images
- User-friendly web interface
- High accuracy CNN model trained on potato plant disease dataset
- Support for common image formats (JPG, JPEG, PNG)
- Immediate prediction results with disease classification

## Technology Stack

<p align="center">
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Matplotlib-FFB000?style=for-the-badge&logo=matplotlib&logoColor=white" alt="Matplotlib">
</p>

## Project Structure

```
Plant_Diagnosis/
├── web.py                              # Streamlit web application
├── Train_model.ipynb                   # Model training notebook
├── trained_plant_disease_model.keras   # Pre-trained model file
├── requirements.txt                    # Python dependencies
├── Plant.png                          # Application image asset
└── dataset/                           # Training and testing data
    ├── Train/                         # Training dataset
    ├── Test/                          # Testing dataset
    └── Valid/                         # Validation dataset
```

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/GxAditya/Plant_Diagnosis.git
cd Plant_Diagnosis
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Web Application

Start the Streamlit web application:
```bash
streamlit run web.py
```

The application will open in your default web browser. If it doesn't open automatically, navigate to the URL shown in the terminal (typically http://localhost:8501).

### Using the Application

1. Navigate to the "Disease Recognition" page from the sidebar
2. Upload a clear image of a potato plant leaf
3. Click the "Predict Disease" button
4. View the prediction result

### Training the Model

To train your own model or retrain with new data:

1. Ensure your dataset is organized in the `dataset/` directory with Train, Test, and Valid subdirectories
2. Open and run the `Train_model.ipynb` notebook in Jupyter
3. The trained model will be saved as `trained_plant_disease_model.keras`

## Model Information

The CNN model is trained to classify potato plant diseases with the following specifications:
- Input image size: 128x128 pixels
- Three output classes: Early Blight, Late Blight, and Healthy
- Training dataset: 900 images across three classes
- Architecture: Convolutional Neural Network optimized for plant disease recognition

## Dataset

The dataset is organized into three categories:
- **Potato Early Blight**: Images of potato leaves affected by early blight disease
- **Potato Late Blight**: Images of potato leaves affected by late blight disease
- **Potato Healthy**: Images of healthy potato leaves

Data is split into:
- Training set
- Validation set
- Testing set

## Dependencies

- tensorflow
- streamlit
- numpy
- matplotlib

For specific version requirements, refer to `requirements.txt`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available for educational and research purposes.

## Author

Aditya Kumar

- GitHub: [GxAditya](https://github.com/GxAditya)
- LinkedIn: [Aditya Kumar](https://linkedin.com/in/aditya-kumar-3721012aa)
- X (Twitter): [@kaditya264](https://x.com/kaditya264?s=09)

## Acknowledgments

This project demonstrates the application of deep learning in agriculture for sustainable farming practices and early disease detection in crops.
