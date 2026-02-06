import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models  import load_model
import pickle

#load model,onehot,encoder,scaler
model=load_model(r"C:\Users\rajen\OneDrive\Desktop\python\Qwen\myenv\DL\model.h5")

with open("label_encoder_gender.pkl",'rb') as file:
    label_encoder_gender=pickle.load(file)

with open("onehot_encode_geo.pkl","rb") as file:
    onehot_encode_geo=pickle.load(file)

with open("DL/scaler.pkl","rb") as file:
    scaler=pickle.load(file)

#Example input_data

input_data = {
    'CreditScore': 600,
    'Geography': 'France',
    'Gender': 'Male',
    'Age': 40,
    'Tenure': 3,
    'Balance': 60000,
    'NumOfProducts': 2,
    'HasCrCard': 1,
    'IsActiveMember': 1,
    'EstimatedSalary': 50000
}

#onehot_encoded_geography
geo_encoded=onehot_encode_geo.transform([[input_data["Geography"]]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=onehot_encode_geo.get_feature_names_out(['Geography']))
print(geo_encoded_df)

df=pd.DataFrame([input_data])


#encode gender
df["Gender"]=label_encoder_gender.transform(df["Gender"])


#concatinating onehotencoder
df=pd.concat([df.drop("Geography",axis=1),geo_encoded_df],axis=1)


#Scaling the input
input_scaled=scaler.transform(df)
print(input_scaled)

#Predict the data
pred=model.predict(input_scaled)
print(pred)


