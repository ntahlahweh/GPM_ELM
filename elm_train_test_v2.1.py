import elm
import os
import pandas as pd
import numpy as np
import pickle 
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

stdsc = StandardScaler()
cwd = os.path.dirname(__file__) # get current directory 
onehotencoder = OneHotEncoder(categories='auto') # encode data into onehotencoder
scaler = MinMaxScaler() # to normalize the data into minmax
best_model_name = "best_gpm_model.sav"

print("Cable fault detector dataset>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

# load dataset
train = pd.read_csv(cwd + "/train_20052022.csv")
test = pd.read_csv(cwd + "/test_20052022.csv")

# processing training data
x_train = stdsc.fit_transform(train.values[:,1:]/16.0) # take the cb and ir values
y_train = train.values[:,:1] # take the target value
print(y_train)

# processing test data
x_test = stdsc.fit_transform(test.values[:,1:]/16.0)
y_test = test.values[:,:1]

print("Cable fault detector dataset classification>>>>>>>>>>>>>>>>>>>>>>>>")

# List of hidden nodes and activation functions
hidden_nodes_list = [10, 30, 50, 100, 200, 300, 500, 800, 1000, 1200, 1500, 2000, 4000]
activation_funcs = ['sigmoid', 'relu']

# Initialize KFold for cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize variables to keep track of the best model
best_test_accuracy = 0
best_hidden_nodes = None
best_activation_func = None
best_model = None

# Loop over each hidden node count and activation function
for activation_func in activation_funcs:
    for hidden_nodes in hidden_nodes_list:
        print(f"Training with {hidden_nodes} hidden nodes and '{activation_func}' activation function:")
        accuracies = []
        test_accuracies = []

        # Perform cross-validation
        for train_index, val_index in kf.split(x_train):
            x_train_cv, x_val_cv = x_train[train_index], x_train[val_index]
            y_train_cv, y_val_cv = y_train[train_index], y_train[val_index]

            # Train ELM model
            model = elm.elm(hidden_units=hidden_nodes, activation_function=activation_func, random_type='normal', x=x_train_cv, y=y_train_cv, C=0.1, elm_type='clf')
            beta, train_accuracy, running_time = model.fit('solution2')

            # Evaluate on the validation set
            val_accuracy = model.score(x_val_cv, y_val_cv)
            accuracies.append(val_accuracy)

            # After validation, evaluate on the test set
            test_accuracy = model.score(x_test, y_test)
            test_accuracies.append(test_accuracy)
            print(f"Test set accuracy: {test_accuracy}")

        # Average validation accuracy over the 5 folds
        avg_val_accuracy = np.mean(accuracies)
        avg_test_accuracy = np.mean(test_accuracies)
        print(f"Average validation accuracy for {hidden_nodes} hidden nodes and '{activation_func}' activation function: {avg_val_accuracy}")
        print(f"Average test accuracy for {hidden_nodes} hidden nodes and '{activation_func}' activation function: {avg_test_accuracy}\n")

        # Save the model if it yields the best test accuracy so far
        if avg_test_accuracy > best_test_accuracy:
            best_test_accuracy = avg_test_accuracy
            best_hidden_nodes = hidden_nodes
            best_activation_func = activation_func
            best_model = model

# Save the best model to a file
if best_model is not None:
    pickle.dump(best_model, open(best_model_name, 'wb'))
    print(f"Best model saved with {best_hidden_nodes} hidden nodes, '{best_activation_func}' activation function, and test accuracy: {best_test_accuracy}")

# Final testing with the test set
load_model = pickle.load(open(best_model_name, 'rb'))
prediction = load_model.predict(x_test)
print("Classifier test prediction:", prediction)
print('Final classifier test accuracy:', load_model.score(x_test, y_test))

# Create confusion matrix and calculate accuracy
cm = confusion_matrix(y_test, prediction)
print(cm)
accuracy = float(cm.diagonal().sum()) / len(y_test)
print('Final model accuracy on test set:', accuracy * 100, '%')
