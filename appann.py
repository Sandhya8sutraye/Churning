import os
import pickle
import tensorflow as tf
import streamlit as st
from pathlib import Path

# Get base directory safely (works locally + Streamlit Cloud)
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()


###loading the model
BASE_DIR = os.path.dirname(os.path.abspath(_file_))

# Load ANN model
# Load model
model = tf.keras.models.load_model(BASE_DIR / "model.h5")

# Load scaler
with open(BASE_DIR / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load label encoder
with open(BASE_DIR / "label_encoder_gender.pkl", "rb") as f:
    label_encoder_gender = pickle.load(f)

# Load one-hot encoder
with open(BASE_DIR / "onehot_encode_geo.pkl", "rb") as f:
    onehot_encode_geo = pickle.load(f)

#Streamlit app
st.title("Customer Churning Prediction")

#User input
geography=st.selectbox('Geography',onehot_encode_geo.categories_[0])
gender=st.selectbox('Gender', label_encoder_gender.classes_)
age=st.slider("Age",18,92)
balance=st.number_input("Balance")
credit_score=st.number_input("Credit Score")
estimated_salary =st.number_input("Estimated Salary")
tenure=st.slider("Tenure",0,10)
num_of_products=st.slider("'Number of Products'",1,4)
has_cr_card=st.selectbox("Has Credit Card",[0,1])
is_active_member=st.selectbox("Is Active Member",[0,1])


#Prepare input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encode_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encode_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

#Predict churn
predict=model.predict(input_data_scaled)
print("Predicted churn: ",predict)
prediction_proba = predict[0][0]  #churn,notchurn

st.write(f'Churn Probability: {prediction_proba:.2f}')  #2f 2d array
 
if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')

