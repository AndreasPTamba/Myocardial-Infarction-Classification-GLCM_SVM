import streamlit as st
import modules.function as fn
import cv2
import numpy as np


if __name__ == "__main__":
  st.title("D4 PPDM - Myocardial Infarction Detection")
  st.write("This is a simple web app to predict Myocardial Infarction using GLCM feature")
  
  st.divider()
  st.write("Upload Image")
  image = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])
  
  if image is not None:
    # Convert the file to an opencv image
    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.write("Image uploaded successfully")
    
    st.divider()
    st.write("Image Preview")
    st.image(image, use_column_width=True)
    
    gray_image = fn.preprocess_image(image)
    
    st.write("Grayscale Image")
    st.image(gray_image, use_column_width=True)
    
    glcm_feature = fn.extract_glcm_feature(gray_image)
    # normalized_feature = fn.normalize_feature(glcm_feature)
    
    st.write("GLCM Feature")
    st.write(glcm_feature)
    # st.write(normalized_feature)
    # print(normalized_feature)
    # print(normalized_feature.shape)
  
  if st.button("Predict"):
    prediction = fn.predict_image(image)
    st.write(prediction)
    
