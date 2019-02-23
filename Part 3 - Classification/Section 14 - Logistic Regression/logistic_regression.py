# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 15:44:48 2019

@author: Alex

Logistic Regression Part - The Data preprocessing from this section will be used in Deep Learning.
"""
#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data_preprocessing import PreprocessingData

#Preprocessing the daya
preprocessing = PreprocessingData()
preprocessing.import_dataset("Social_Network_Ads.csv")
preprocessing.choose_which_column_to_dataset(indVariables = 2, depVariables = 4)
preprocessing.seperate_training_test_set(test_size_training = 0.25, random_state_training = 0)
preprocessing.feature_scaling()
dataset, independantVariables, dependantVariables = preprocessing.get_imported_variables()
independantVariables_TrainingSet, independantVariables_TestSet, dependantVariables_TrainingSet, dependantVariables_TestSet = preprocessing.get_trainingSet_variables()

#Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(independantVariables_TrainingSet, dependantVariables_TrainingSet)

#Predicting the Test set results
independantVariables_prediction = classifier.predict(independantVariables_TestSet)

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(dependantVariables_TestSet, independantVariables_prediction)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = independantVariables_TrainingSet, dependantVariables_TrainingSet
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = independantVariables_TestSet, dependantVariables_TestSet
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()