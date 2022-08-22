import elm
import os
import pandas as pd
import numpy as np
import pickle 
from sklearn.model_selection import train_test_split #to split the dataset
from sklearn.datasets import load_iris, load_digits, load_diabetes, make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

model_name = "gpm_model.sav"
stdsc = StandardScaler()
cwd = os.path.dirname(__file__)

# load dataset
test = pd.read_csv(cwd + "/test_1.csv")
x_test = stdsc.fit_transform(test.values[:,1:]/16.0)
y_test_transpose = np.transpose(test.values[:,:1])
y_test = y_test_transpose[0]

print("Cable fault detector dataset classification>>>>>>>>>>>>>>>>>>>>>>>>")

# test

load_model = pickle.load(open(model_name, 'rb'))
prediction = load_model.predict(x_test)
print("classifier test prediction:", prediction)
print('classifier test accuracy:', load_model.score(x_test, y_test))