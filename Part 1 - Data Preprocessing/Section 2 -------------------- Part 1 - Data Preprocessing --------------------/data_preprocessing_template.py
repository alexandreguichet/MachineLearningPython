# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 14:57:23 2019

@author: Alex
"""
#Data Preprocessing

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Data.csv')
independantVariables = dataset.iloc[:, :-1].values # The independant variables of the dataset imported [Country, Age, Salary]
dependantVariables = dataset.iloc[:, 3].values # The dependant variables of the dataset imported [Purchased]

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
independantVariables_TrainingSet, independantVariables_TestSet, dependantVariables_TrainingSet, dependantVariables_TestSet = train_test_split(independantVariables, 
                                                                                                                                              dependantVariables, 
                                                                                                                                              test_size = 0.2, 
                                                                                                                                              random_state = 0)

#IF NEEDED  [[FEATURE SCALING, TAKING CARE OF MISSING DATA, ENCODING CATEGORICAL DATA]
'''
#Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(independantVariables[:, 1:3])
independantVariables[:, 1:3] = imputer.transform(independantVariables[:, 1:3])

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelEncoderIndependantVariables = LabelEncoder()
independantVariables[:, 0] = labelEncoderIndependantVariables.fit_transform(independantVariables[:, 0])
oneHotEncoderIndependantVariables = OneHotEncoder(categorical_features= [0])
independantVariables = oneHotEncoderIndependantVariables.fit_transform(independantVariables).toarray()

labelEncoderDependantVariables = LabelEncoder()
dependantVariables = labelEncoderDependantVariables.fit_transform(dependantVariables) #No need for OneHotEncoder for this one because there is no more than 2 outputs: (Yes/No)

#Feature scaling
from sklearn.preprocessing import StandardScaler
scale_independantVariables= StandardScaler()
independantVariables_TrainingSet = scale_independantVariables.fit_transform(independantVariables_TrainingSet)
independantVariables_TestSet = scale_independantVariables.transform(independantVariables_TestSet)
'''