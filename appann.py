import os
import pickle
import tensorflow as tf
import streamlit as st
import pandas as pd
from pathlib import Path

# Get base directory safely
try:
    BASE_DIR = Path(_file_).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()


@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(BASE_DIR / "model.h5")

    with open(BASE_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open(BASE_DIR / "label_encoder_gender.pkl", "rb") as f:
        label_encoder_gender = pickle.load(f)

    with open(BASE_DIR / "onehot_encode_geo.pkl", "rb") as f:
        onehot_encode_geo = pickle.load(f)

    return model, scaler, label_encoder_gender, onehot_encode_geo


model, scaler, label_encoder_gender, onehot_encode_geo = load_artifacts()

# Streamlit UI
st.title("Customer Churning Prediction")

geography = st.selectbox("Geography", onehot_encode_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 92)
credit_score = st.number_input("Credit Score", min_value=300, max_value=900)
balance = st.number_input("Balance")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

# Prepare input
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [label_encoder_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
})

geo_encoded = onehot_encode_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encode_geo.get_feature_names_out(["Geography"])
)

input_data = pd.concat([input_data, geo_encoded_df], axis=1)

input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f"Churn Probability: *{prediction_proba:.2f}*")

if prediction_proba > 0.5:
    st.error("The customer is likely to churn.")
else:
    st.success("The customer is not likely to churn.")