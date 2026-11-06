import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

#load the model
model = tf.keras.models.load_model('salary_regression_model.h5')

# load the label encoder, one hot encoder and scaler
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

with open('onehot_encoder.pkl', 'rb') as f:
    onehot_encoder = pickle.load(f)

with open('scaling.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("Bank Customer Salary Prediction")

geography = st.selectbox("Select Geography", onehot_encoder.categories_[0])
gender = st.selectbox("Select Gender", le.classes_)
age = st.slider("Select Age", 18, 100)
balance = st.number_input("Enter Balance", min_value=0.0, step=100.0)
credit_score = st.slider("Select Credit Score", 600, 900)
tenure = st.slider("Select Tenure", 0, 10)
num_of_products = st.slider("Select Number of Products", 1, 4)
has_credit_card = st.selectbox("Has Credit Card?", [0, 1])
is_active_member = st.selectbox("Is Active Member?", [0, 1])
exited = st.selectbox("Exited?", [0, 1])

#prepoare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [le.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited]
})  

geo_data = onehot_encoder.transform([[geography]]).toarray()
geo_df = pd.DataFrame(geo_data, columns=onehot_encoder.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True), geo_df], axis=1)

#scale the input data
input_data_scaled = scaler.transform(input_data)

#predict the salary
prediction = model.predict(input_data_scaled)
prediction_salary = prediction[0][0]

st.success(f"The predicted salary is ${prediction_salary:.2f}")