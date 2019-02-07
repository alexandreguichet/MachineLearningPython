# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 07:48:17 2019

@author: Alex

Simple Linear Regression
"""
#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data_preprocessing import PreprocessingData

preprocessing = PreprocessingData()
preprocessing.import_dataset("Salary_Data.csv")
preprocessing.seperate_training_test_set(test_size = 1/3, random_state = 0)
dataset, independantVariables, dependantVariables = preprocessing.get_imported_variables()
independantVariables_TrainingSet, independantVariables_TestSet, dependantVariables_TrainingSet, dependantVariables_TestSet = preprocessing.get_trainingSet_variables()

#Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(independantVariables_TrainingSet, dependantVariables_TrainingSet)

#Predicting the Test set results
dependantVariable_prediction = regressor.predict(independantVariables_TestSet)

#Visualising the Training set results
plt.scatter(independantVariables_TrainingSet, dependantVariables_TrainingSet, color = "red")
plt.plot(independantVariables_TrainingSet, regressor.predict(independantVariables_TrainingSet), color = 'blue')
plt.title("Salary vs Experience (Training set)")
plt.xlabel("Years of Experience (years)")
plt.ylabel("Salary ($)")
plt.show()

plt.scatter(independantVariables_TestSet, dependantVariables_TestSet, color = "red")
plt.plot(independantVariables_TrainingSet, regressor.predict(independantVariables_TrainingSet), color = 'blue')
plt.title("Salary vs Experience (Test set)")
plt.xlabel("Years of Experience (years)")
plt.ylabel("Salary ($)")
plt.show()