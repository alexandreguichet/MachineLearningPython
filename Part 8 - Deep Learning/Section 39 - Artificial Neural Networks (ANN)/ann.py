# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 14:41:28 2019

@author: Alex

Artificial Neural Network
"""

#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from data_preprocessing import PreprocessingData

#Preprocessing the data
preprocessing = PreprocessingData() #Create instance of PreprocessingData class
preprocessing.import_dataset("Churn_Modelling.csv") #Import the dataset and pre-separate the dataset into dependant and independant variables
preprocessing.separate_independant_dependant_variables(indVariables = 3, depVariables = 13) #Separate dataset into independant and dependant variables by selecting which column each cathegory starts from
preprocessing.encode_categorical_data() #Encode names into categorical data and remove one dummy variable
preprocessing.seperate_training_test_set(test_size_training = 0.2, random_state_training = 0) #Separate test and training set
preprocessing.feature_scaling() #scale features 
dataset, independantVariables, dependantVariables = preprocessing.get_imported_variables()
independantVariables_TrainingSet, independantVariables_TestSet, dependantVariables_TrainingSet, dependantVariables_TestSet = preprocessing.get_trainingSet_variables()

#Create Artificial Neural Network
    #Initialising the ANN
classifier = Sequential()
    #Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer = 'uniform', input_dim = 11))
    #Adding a new hidden layer
classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer = 'uniform'))
    #Adding the output layer
classifier.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer = 'uniform'))

#Compile ANN (Apply stochastic gradiant decent)
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the ANN to the Training set
classifier.fit(independantVariables_TrainingSet, dependantVariables_TrainingSet, batch_size = 10, epochs = 100)

#Making the predictions and evaluating the model
    #Prediction and evaluating the model
dependant_variable_prediction = classifier.predict(independantVariables_TestSet)
dependant_variable_prediction = (dependant_variable_prediction > 0.5)

    #Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(dependantVariables_TestSet, dependant_variable_prediction)