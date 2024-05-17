import streamlit as st
import pandas as pd
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
import joblib

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

def predict_image(image):
    gray_image = preprocess_image(image)
    glcm_feature = extract_glcm_feature(gray_image)
    
    # Debugging output for feature
    st.write("GLCM Feature Shape:", glcm_feature.shape)
    st.write("GLCM Feature Data:", glcm_feature)
    
    svm = joblib.load('jbmodel.joblib')
    
    prediction = svm.predict(glcm_feature)
    
    # Debugging prediction output
    st.write("Model Prediction:", prediction)
    
    return prediction[0]

if __name__ == "__main__":
    st.title("D4 PPDM - EKG Simple Classification")
    st.write("This is a simple web app to predict Myocardial Infarction using GLCM feature")
    
    st.divider()
    st.write("Upload Image")
    image = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])
    
    if image is not None:
        file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.write("Image uploaded successfully")
        
        st.divider()
        st.write("Image Preview")
        st.image(image, use_column_width=True)
        
        gray_image = preprocess_image(image)
        
        st.write("Grayscale Image")
        st.image(gray_image, use_column_width=True)
        
        glcm_feature = extract_glcm_feature(gray_image)
        
        st.write("GLCM Feature")
        st.write(glcm_feature)
        
        if st.button("Predict"):
            prediction = predict_image(image)
            st.write("Prediction Result:", prediction)
