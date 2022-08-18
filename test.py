import elm
import pandas as pd
from sklearn.model_selection import train_test_split #to split the dataset
from sklearn.datasets import load_iris, load_digits, load_diabetes, make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

stdsc = StandardScaler()
# **********************************
# mnist dataset classification
# **********************************

print("mnist dataset>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
# load dataset
train = pd.read_csv('mnist_train.csv')
test = pd.read_csv('mnist_test.csv')

print("handwritten number dataset classification>>>>>>>>>>>>>>>>>>>>>>>>")
# load dataset

digits = load_digits()
dgx, dgy = stdsc.fit_transform(digits.data/16.0), digits.target  #digts.data is the training data; digits.target is the output training
print("dgx shape:", dgx.shape) # give the dimension of the array
print("dgy shape:", dgy.shape)
x_train, x_test, y_train, y_test = train_test_split(dgx, dgy, test_size=0.2)

model = elm.elm(hidden_units=32, activation_function='relu', random_type='normal', x=x_train, y=y_train, C=0.1, elm_type='clf')
beta, train_accuracy, running_time = model.fit('solution2')
print("classifier beta:\n", beta)
print("classifier train accuracy:", train_accuracy)
print('classifier running time:', running_time)

# test
prediction = model.predict(x_test)
print("classifier test prediction:", prediction)
print('classifier test accuracy:', model.score(x_test, y_test))
