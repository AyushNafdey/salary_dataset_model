import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

# Load the trained model, label encoders, and scaler
try:
    model = joblib.load('best_model.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    scaler_X = joblib.load('scaler_X.pkl')
except FileNotFoundError:
    st.error("Model or preprocessor files not found. Please ensure 'best_model.pkl', 'label_encoders.pkl', and 'scaler_X.pkl' are in the same directory.")
    st.stop()

st.title('Salary Prediction App')
st.write('Enter the employee details to predict their salary.')

# Input fields for user
age = st.number_input('Age', min_value=18, max_value=100, value=30)
years_of_experience = st.number_input('Years of Experience', min_value=0, max_value=60, value=5)

# Define options for categorical features based on the loaded label encoders
gender_options = label_encoders['Gender'].classes_.tolist()
education_options = label_encoders['Education Level'].classes_.tolist()
job_title_options = label_encoders['Job Title'].classes_.tolist()

gender = st.selectbox('Gender', gender_options)
education_level = st.selectbox('Education Level', education_options)
job_title = st.selectbox('Job Title', job_title_options)

if st.button('Predict Salary'):
    # Preprocess input data
    try:
        gender_encoded = label_encoders['Gender'].transform([gender])[0]
        education_encoded = label_encoders['Education Level'].transform([education_level])[0]
        job_title_encoded = label_encoders['Job Title'].transform([job_title])[0]
    except ValueError as e:
        st.error(f"Error encoding categorical features: {e}. Please ensure selected options match the training data categories.")
        st.stop()

    # Create a DataFrame for the input
    input_data = pd.DataFrame([[age, gender_encoded, education_encoded, job_title_encoded, years_of_experience]],
                              columns=['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience'])

    # Scale numerical features (Age, Years of Experience are the only numerical ones in X that were scaled)
    # The scaler_X was fit on the original X, which included encoded categorical features.
    # So, we need to apply scaler_X on the whole input_data DataFrame.
    input_data_scaled = scaler_X.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)

    st.success(f'Predicted Salary: ${prediction[0]:,.2f}')