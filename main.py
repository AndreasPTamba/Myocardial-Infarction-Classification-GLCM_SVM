import streamlit as st
import pandas as pd
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
import joblib
import pickle
from matplotlib import pyplot as plt
import seaborn as sns

# Existing functions remain the same
def resize_image(image, width=None, height=None):
    cv2.rectangle(image, (78, 294), (78 + 2089, 294 + 1217), (0, 255, 0), 2)
    roi = image[294:294+1217, 78:78+2089]
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    
    h, w = roi_rgb.shape[:2]
    aspect_ratio = w / h

    if width is None:
        new_height = int(height / aspect_ratio)
        resized_image = cv2.resize(roi_rgb, (height, new_height), interpolation=cv2.INTER_AREA)
    else:
        new_width = int(width * aspect_ratio)
        resized_image = cv2.resize(roi_rgb, (new_width, width), interpolation=cv2.INTER_AREA)

    return resized_image

def preprocess_image(image):
    resized_image = resize_image(image, 256)
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    return gray_image

def calc_glcm_all_agls(img, props, dists=[1], agls=[0, np.pi/4, np.pi/2, 3*np.pi/4], lvl=256, sym=True, norm=True):
    glcm = graycomatrix(img, 
                        distances=dists, 
                        angles=agls, 
                        levels=lvl,
                        symmetric=sym, 
                        normed=norm)
    feature = []
    glcm_props = [properti for name in props for properti in graycoprops(glcm, name)[0]]
    for item in glcm_props:
        feature.append(item)
    
    return feature

def extract_glcm_feature(image):
    properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']
    glcm_features = calc_glcm_all_agls(image, properties)
    
    columns = []
    angles = ['0', '45', '90', '135']
    for prop in properties:
        for ang in angles:
            columns.append(prop + '_' + ang)
    
    df = pd.DataFrame([glcm_features], columns=columns)
    test = df.iloc[:, :].values
    return test

def normalize_feature(scaled):
    scalerfile = './scaler.sav'
    scaler = pickle.load(open(scalerfile, 'rb'))
    normalized_feature = scaler.transform(scaled)
    return normalized_feature

def predict_image(image):
    gray_image = preprocess_image(image)
    glcm_feature = extract_glcm_feature(gray_image)
    
    svm = joblib.load('jbmodel.joblib')
    prediction = svm.predict(glcm_feature)
    return prediction

def classification_report():
    report = pd.read_csv('classification_report.csv')
    
    accuracy = report[report['Unnamed: 0'] == 'accuracy']
    macro_avg = report[report['Unnamed: 0'] == 'macro avg']
    weighted_avg = report[report['Unnamed: 0'] == 'weighted avg']
    
    accuracy_value = accuracy['precision'].values[0]
    macro_avg_precision = macro_avg['precision'].values[0]
    weighted_avg_f1 = weighted_avg['f1-score'].values[0]
    
    return accuracy_value, macro_avg_precision, weighted_avg_f1

def confusion_matrix():
    cm = pd.read_csv('confusion_matrix.csv')
    return cm

# Main application
def main():
    st.set_page_config(
        page_title="CardioSense AI - ECG Analysis",
        page_icon="❤️",
        layout="wide"
    )

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "ECG Analysis", "About MI", "Model Performance"])

    if page == "Home":
        show_home_page()
    elif page == "ECG Analysis":
        show_analysis_page()
    elif page == "About MI":
        show_about_mi_page()
    else:
        show_model_performance_page()

def show_home_page():
    st.title("🫀 CardioSense AI")
    st.subheader("Advanced ECG Analysis for Myocardial Infarction Detection")
    
    st.markdown("""
    ### Welcome to CardioSense AI
    
    This advanced platform uses artificial intelligence to analyze ECG images and detect potential signs of Myocardial Infarction (MI). Our system employs sophisticated image processing techniques and machine learning to provide rapid, accurate assessments of ECG readings.
    
    #### Key Features:
    - 📊 Advanced GLCM feature extraction
    - 🔍 Real-time ECG image analysis
    - 📈 Comprehensive performance metrics
    - 🎯 High accuracy detection
    
    #### How to Use:
    1. Navigate to the ECG Analysis page
    2. Upload your ECG image
    3. Get instant analysis results
    """)

def show_analysis_page():
    st.title("ECG Analysis")
    st.write("Upload your ECG image for analysis")
    
    image = st.file_uploader("Choose an ECG image...", type=['jpg', 'png', 'jpeg'])
    
    if image is not None:
        # Convert the file to an opencv image
        file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
        
        gray_image = preprocess_image(image)
        
        with col2:
            st.subheader("Processed Image")
            st.image(gray_image, use_column_width=True)
        
        with st.expander("View Technical Details"):
            glcm_feature = extract_glcm_feature(gray_image)
            normalized_feature = normalize_feature(glcm_feature)
            
            st.write("GLCM Features:")
            st.write(glcm_feature)
            st.write("Normalized Features:")
            st.write(normalized_feature)
        
        if st.button("Analyze ECG"):
            with st.spinner("Analyzing ECG..."):
                prediction = predict_image(image)
                
                st.subheader("Analysis Results")
                if prediction[0] == 1:
                    st.error("⚠️ Potential Myocardial Infarction Detected")
                    st.markdown("""
                        ### Recommended Actions:
                        1. Seek immediate medical attention
                        2. Contact your healthcare provider
                        3. Keep calm and rest
                        """)
                else:
                    st.success("✅ No signs of Myocardial Infarction detected")
                    st.markdown("""
                        ### Recommendations:
                        - Continue regular health monitoring
                        - Maintain heart-healthy lifestyle
                        - Regular check-ups with your doctor
                        """)

def show_about_mi_page():
    st.title("Understanding Myocardial Infarction")
    
    st.markdown("""
    ### What is Myocardial Infarction?
    
    Myocardial Infarction (MI), commonly known as a heart attack, occurs when blood flow to a part of the heart muscle is blocked, causing damage to the heart tissue.
    
    #### Key Indicators in ECG:
    1. ST-segment elevation or depression
    2. T-wave inversion
    3. Q-wave abnormalities
    4. Changes in R-wave progression
    
    #### Risk Factors:
    - High blood pressure
    - High cholesterol
    - Smoking
    - Diabetes
    - Family history
    - Age and gender
    - Obesity
    - Sedentary lifestyle
    
    #### Prevention:
    - Regular exercise
    - Healthy diet
    - Stress management
    - Regular medical check-ups
    - Blood pressure control
    - Cholesterol management
    """)
    
    st.image("https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80", 
             caption="ECG Reading Example")

def show_model_performance_page():
    st.title("Model Performance Metrics")
    
    accuracy, macro_avg, weighted_avg = classification_report()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Accuracy", f"{accuracy:.2%}")
    with col2:
        st.metric("Macro Avg Precision", f"{macro_avg:.2%}")
    with col3:
        st.metric("Weighted Avg F1", f"{weighted_avg:.2%}")
    
    st.subheader("Confusion Matrix")
    cm = confusion_matrix()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(fig)
    
    st.markdown("""
    ### Model Details
    
    Our model uses Support Vector Machine (SVM) classification with GLCM features extracted from ECG images. The model has been trained on a diverse dataset of ECG images and validated against expert diagnoses.
    
    #### Feature Extraction:
    - Gray Level Co-occurrence Matrix (GLCM)
    - Multiple angles (0°, 45°, 90°, 135°)
    - Key properties: dissimilarity, correlation, homogeneity, contrast, ASM, energy
    
    #### Model Training:
    - Cross-validation
    - Hyperparameter optimization
    - Regular retraining with new data
    """)

if __name__ == "__main__":
    main()
