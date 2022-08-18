import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

#load training and testing dataset
train = pd.read_csv('mnist_train.csv') 
test = pd.read_csv('mnist_test.csv')

onehotencoder = OneHotEncoder(categories='auto') #encode data into onehotencoder
scaler = MinMaxScaler() #to normalize the data into minmax

X_train = scaler.fit_transform(train.values[:,1:])
y_train = onehotencoder.fit_transform(train.values[:,:1]).toarray()
X_test = scaler.fit_transform(test.values[:,1:])
y_test = onehotencoder.fit_transform(test.values[:,:1]).toarray()

input_size = X_train.shape
output_size = y_train.shape
hidden_size = 1000

print("input size: \n" , input_size)
print("Output size: \n", output_size)
#print ("X_train: \n",  X_train)
#print ("Y_train", y_train)

onehotencoder = OneHotEncoder(categories='auto') #encode data into onehotencoder
scaler = MinMaxScaler() #to normalize the data into minmax

x_train = scaler.fit_transform(train.values[:,1:]/16.0)
y_train = onehotencoder.fit_transform(train.values[:,:1])
X_test = scaler.fit_transform(test.values[:,1:])
y_test = onehotencoder.fit_transform(test.values[:,:1])