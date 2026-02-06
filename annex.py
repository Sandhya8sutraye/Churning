import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle

#Load dataset

data=pd.read_csv(r"C:\Users\rajen\OneDrive\Desktop\python\Qwen\myenv\DL\Churn_Modelling.csv")


#Remove unwanted column
data=data.drop(["RowNumber","CustomerId","Surname"],axis=1)


#Encode Gender
label_encoder_gender=LabelEncoder()
data["Gender"]=label_encoder_gender.fit_transform(data["Gender"])


#Onehotencode geo
onehot_encode_geo=OneHotEncoder()
geo_encoder=onehot_encode_geo.fit_transform(data[["Geography"]]).toarray()


print(onehot_encode_geo.get_feature_names_out(["Geography"]))

geo_encoder_df=pd.DataFrame(geo_encoder,columns=onehot_encode_geo.get_feature_names_out(['Geography']))
print(geo_encoder_df)

#Combine OneHotEncoder Geogaphy
data=pd.concat([data.drop("Geography",axis=1),geo_encoder_df],axis=1)


#Save the ecncoder
with open("label_encoder_gender.pkl",'wb') as file:
    pickle.dump(label_encoder_gender,file)

with open("onehot_encode_geo.pkl","wb") as file:
    pickle.dump(onehot_encode_geo,file)
print(data.head())

#Divide the dataset into independenet and dependent
x=data.drop("Exited",axis=1)
y=data["Exited"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)

#Scale the features
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

with open("scaler.pkl","wb") as file:
    pickle.dump(scaler,file)


##ANN Implementation

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
import datetime

print(x_train.shape[1])

model=Sequential([
    Dense(64,activation="relu",input_shape=(x_train.shape[1],)),  #hidden layer 1 only 1st hl we need to give input
    Dense(32,activation="relu"),                                   #hidden layer2
    Dense(1,activation="sigmoid")                                 #outputlayer
]   
)
print(model.summary())

#Optimizer,loss

opt=tf.keras.optimizers.Adam(learning_rate=0.01)
loss=tf.keras.losses.BinaryCrossentropy()

#Compile (how to learn loss=mistake ,opt=reduce mistake,metrics measure performance)

model.compile(optimizer=opt,loss=loss,metrics=["accuracy"])

#Setup tensorboard

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorflow_callback=TensorBoard(log_dir=log_dir,histogram_freq=1)

#Set up early stopping
early_stopping_callback=EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)

#Train the model
history=model.fit(
    x_train,y_train,validation_data=(x_test,y_test),epochs=100,
    callbacks=[tensorflow_callback,early_stopping_callback]
)
model.save('model.h5')


## Load Tensorboard Extension
#ensorboard --logdir logs/fit













