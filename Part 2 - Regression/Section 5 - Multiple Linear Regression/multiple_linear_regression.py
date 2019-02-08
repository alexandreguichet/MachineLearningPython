# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 17:49:25 2019

@author: Alex
Multible Linear Regression
"""

#Import libraries
import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import PreprocessingData

#Preprocess using Preprocessing Custom Class
preprocessing = PreprocessingData()
preprocessing.import_dataset("50_Startups.csv")
preprocessing.adjust_missing_data()
preprocessing.encode_categorical_data()
preprocessing.seperate_training_test_set(test_size_training = 0.2, random_state_training = 0)
dataset, independantVariables, dependantVariables = preprocessing.get_imported_variables()
independantVariables_TrainingSet, independantVariables_TestSet, dependantVariables_TrainingSet, dependantVariables_TestSet = preprocessing.get_trainingSet_variables()

#Fitting Multiple LinearRegression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(independantVariables_TrainingSet, dependantVariables_TrainingSet)

#Predicting the Test set results
prediction_vector = regressor.predict(independantVariables_TestSet)

#Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
independantVariables = np.append(arr = np.ones(50,1).astype(int), values = independantVariables, axis = 1)