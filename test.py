import elm
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split #to split the dataset
from sklearn.datasets import load_iris, load_digits, load_diabetes, make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

stdsc = StandardScaler()
cwd = os.path.dirname(__file__) #get current directory 
onehotencoder = OneHotEncoder(categories='auto') #encode data into onehotencoder
scaler = MinMaxScaler() #to normalize the data into minmax

# **********************************
# mnist dataset classification
# **********************************

print("Cable fault detector dataset>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

# load dataset
train = pd.read_csv(cwd + "/train_20052022.csv")
test = pd.read_csv(cwd + "/test_20052022.csv")
# processing training data
x_train = stdsc.fit_transform(train.values[:,1:]/16.0)
y_train_transpose = np.transpose(train.values[:,:1])
y_train = y_train_transpose[0]
# processing test data
x_test = stdsc.fit_transform(test.values[:,1:]/16.0)
y_test_transpose = np.transpose(test.values[:,:1])
y_test = y_test_transpose[0]
print("Cable fault detector dataset classification>>>>>>>>>>>>>>>>>>>>>>>>")

#training
model = elm.elm(hidden_units=10000, activation_function='sigmoid', random_type='normal', x=x_train, y=y_train, C=0.1, elm_type='clf')
beta, train_accuracy, running_time = model.fit('solution2')
print("classifier beta:\n", beta)
print("classifier train accuracy:", train_accuracy)
print('classifier running time:', running_time)

# test
prediction = model.predict(x_test)
print("classifier test prediction:", prediction)
print('classifier test accuracy:', model.score(x_test, y_test))