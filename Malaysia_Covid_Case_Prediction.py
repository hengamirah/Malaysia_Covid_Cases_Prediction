# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 09:28:49 2022

@author: Amirah Heng
"""

import os
import pickle
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import TensorBoard
from Malaysia_Covid_Case_Module import ModelEvaluation,EDA
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error,mean_squared_error
from Malaysia_Covid_Case_Module import ModelCreation, ModelAnalysis
#%%Statics

log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_PATH = os.path.join(os.getcwd(),'logs', log_dir)
TRAIN_PATH= os.path.join(os.getcwd(),'dataset','cases_malaysia_train.csv')
TEST_PATH = os.path.join(os.getcwd(),'dataset','cases_malaysia_test.csv')
MMS_PATH= os.path.join(os.getcwd(),'model', 'mms.pkl')
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model','model.h5')
#%% STEP 1 - Data Loading

df = pd.read_csv(TRAIN_PATH)

#%% STEP 2 - Data Inspection

df.info() #There are 30 columns in this data  
df.tail(10) #679 dataset available in this

#Describe data mean,median, IQR
temp= df.describe().T 

#Change all cases to float data
for i in df.columns:
    df[i]=pd.to_numeric(df[i],errors='coerce')

#Plot graphs
EDA().plot_graph(df)

#Check for missing datas/NaN values
df.isna().sum() #There are 342 missing datas from cluster_import ,cluster_religious,
# cluster_community , cluster_education ,cluster_detentionCentre , cluster_workplace   
      
df.duplicated().sum() #no duplicated data in this dataset

#Check for outliers
df.boxplot(figsize=(40,15)) #Cases_active

#%% STEP 3 - Data Cleaning

#Interpolate data to fill in NaN values
df.isna().sum() 

#cases_new data shows a degree 3 wave graph hence polynomial degree 3 is used to interpolate
df['cases_new'].interpolate(method='polynomial', order=2, inplace=True)

df['cases_new'].isna().sum()
#Plot graph for target column
# EDA().plot_covid_data(column_names, dataset=df['cases_new'])
    
plt.figure()
plt.plot(df['cases_new'])
plt.xlabel('cases_new')
plt.show()


#%% STEP 4 - Features Selection
#Cases new will be the selected data for this model
#%% STEP 5 - Preprocessing

mms = MinMaxScaler()
df = mms.fit_transform(np.expand_dims(df['cases_new'],axis=-1))

#Create a container to develop model
X_train = []
y_train = []

win_size = 30

for i in range(win_size, np.shape(df)[0]):
    X_train.append(df[i-win_size:i, 0])
    y_train.append(df[i, 0])

X_train = np.array(X_train) 
y_train = np.array(y_train) 


#%% STEP 6 - Model Development
# USE LSTM layers, dropout, dense, input

nb_features=np.shape(X_train)[1] #30

MC= ModelCreation()
model = MC.simple_lstm_layer(nb_features,num_node=64,drop_rate=0.2,
                             output_node=1)

#%% STEP 7) MODEL ANALYSIS

#Show model architecture
plot_model(model,show_layer_names=(True), show_shapes=(True))

#compile model 
model.compile(optimizer='adam', loss='mse', metrics ='mape')

X_train = np.expand_dims(X_train, axis=-1) #(30,1)

#callbacks
tensorboard_callback= TensorBoard(log_dir=LOG_PATH)

#Train and test model created
hist= model.fit(X_train, y_train, batch_size=20, epochs=200,
                callbacks=tensorboard_callback)

# Evaluate model loss and mape in graph plot
MA= ModelAnalysis()
MA.PlotHistory(hist)

#%% STEP 8 - Model Evaluation

test_df = pd.read_csv(TEST_PATH)

#Change cases_new to float data
test_df.iloc[:,1]=pd.to_numeric(test_df.iloc[:,1],errors='coerce')

#There is 1 NaN in test dataset
test_df.isna().sum()

#Interpolate data to remove NaN
test_df['cases_new'].interpolate(method='polynomial', order=2,inplace=True)

test_df = mms.transform(np.expand_dims(test_df.iloc[:,1],axis=-1))

# Concatenate test_df + df 
con_test = np.concatenate((df,test_df),axis=0)
con_test = con_test[-(win_size+len(test_df)):] 

plt.figure()
plt.plot(test_df)
plt.show()

X_test =[]
for i in range(win_size, len(con_test)): #(30,130)
    X_test.append(con_test[i-win_size:i , 0])
    
#1st iteration ---> i=30 get first 30days of data
#2nd iteration ---> i=31 get next 30days of data
#first row of data can predict next row of data

X_test = np.array(X_test)
predicted= model.predict(np.expand_dims(X_test,axis=-1))

test_df_inverse=mms.inverse_transform(test_df)

predicted_inverse = mms.inverse_transform(predicted)

#%%Plotting of graphs

ME= ModelEvaluation()
ME.plot_predicted_graph(test_df,predicted)

#%% MAPE value

print("The mean_absolute_percentage_error(MAPE) value is:",mean_absolute_percentage_error(test_df_inverse, predicted_inverse))
print("The mean_squared_error: value is", mean_squared_error(test_df_inverse, predicted_inverse))
print("The mean_absolute_error: value is", mean_absolute_error(test_df_inverse, predicted_inverse))

# print((mean_absolute_error(test_df, predicted)/sum(abs(test_df))) *100)
#%% Step 9) Model Saving

#Save Model
model.save(MODEL_SAVE_PATH)
#Save scaler model
with open(MMS_PATH, 'wb') as file:
    pickle.dump(mms, file)

#%% Discussion)

# The MAPE value achieved is 0.11 which is high in performance
# A simple LSTM, Dense, and Dropout layers is implemented in this model.
# The MAPE loss can be further reduced in the future with some suggested approach:
    # 1) Increasing number of samples in the dataset
    # 2) Increasing the number of epochs
    # 3) Introduce different model architectures 

# A simple Long Short-Term Memory (LSTM) model is implemented with
# with an input layer, a single hidden (LSTM) layer, and an output layer 
# that is used to make a prediction. The input layer has neurons equal to 30
# sequence steps (for 30 days COVID-19 data points). 
# The hidden layer is an LSTM layer with 64 hidden units (neurons) 
# and a rectified linear unit (ReLU) as an activation function. 
# The output layer had a dense layer with 1 unit for predicting the output. 
# Moreover, we have set 300 as the number of epochs, Adam as the optimizer, 
# and the mape as the loss function. 
# After that, we fit the model with prepared data to make a prediction. 
# The obtained results may vary given the stochastic nature of the LSTM model; 
# therefore, we have run it several times. 
# Finally, we enter the last sequence with output to forecast the next value in the series.


