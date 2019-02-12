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

#Backward Elimination fonction with P-Values and Adjusted R Squared
def backwardElimination(x,y, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x

#Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
independantVariables = np.append(arr = np.ones((50,1)).astype(int), values = independantVariables, axis = 1)

SL = 0.05 #Significance levels
independantVariables_optimal = independantVariables[:, [0, 1, 2, 3, 4, 5]]
variables_Modeled = backwardElimination(independantVariables_optimal, dependantVariables, SL)

