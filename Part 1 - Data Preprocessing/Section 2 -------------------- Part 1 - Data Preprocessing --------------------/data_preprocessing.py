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

class PreprocessingData: 
  
    def __init__(self):
        print("I instantiated my class successfully")
        
    #Importing the dataset
    #TODO: make sure the number of column of independant variables is not just the last column but maybe more? Add interactivity
    def import_dataset(self, data):
        self.dataset = pd.read_csv(data)
        self.independantVariables = self.dataset.iloc[:, :-1].values # The independant variables of the dataset imported [Country, Age, Salary]
        self.dependantVariables = self.dataset.iloc[:, -1:].values # The dependant variables of the dataset imported [Purchased]

    #Splitting the dataset into the Training set and Test set
    def seperate_training_test_set(self, test_size = 0.2, random_state = 0):
        from sklearn.model_selection import train_test_split
        self.independantVariables_TrainingSet, self.independantVariables_TestSet, self.dependantVariables_TrainingSet, self.dependantVariables_TestSet = train_test_split(self.independantVariables, 
                                                                                                                                              self.dependantVariables, 
                                                                                                                                              test_size = 1/3, 
                                                                                                                                              random_state = 0)

    #Taking care of missing data
    def adjust_missing_data(self):
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
        imputer.fit(self.independantVariables[:, 1:3])
        self.independantVariables[:, 1:3] = imputer.transform(self.independantVariables[:, 1:3])


    #Encoding categorical data
    def encode_categorical_data(self):       
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        labelEncoderIndependantVariables = LabelEncoder()
        self.independantVariables[:, 0] = labelEncoderIndependantVariables.fit_transform(self.independantVariables[:, 0])
        oneHotEncoderIndependantVariables = OneHotEncoder(categorical_features= [0])
        self.independantVariables = oneHotEncoderIndependantVariables.fit_transform(self.independantVariables).toarray()

    #TODO: change the line bellow if we need more than two variables output (Else than a yes/no answer)
        labelEncoderDependantVariables = LabelEncoder()
        self.dependantVariables = labelEncoderDependantVariables.fit_transform(self.dependantVariables) #No need for OneHotEncoder for this one because there is no more than 2 outputs: (Yes/No)

    #Feature scaling
    def feature_scaling(self):
        from sklearn.preprocessing import StandardScaler
        scale_independantVariables= StandardScaler()
        self.independantVariables_TrainingSet = scale_independantVariables.fit_transform(self.independantVariables_TrainingSet)
        self.independantVariables_TestSet = scale_independantVariables.transform(self.lindependantVariables_TestSet)
        
    def get_imported_variables(self):
        return self.dataset, self.independantVariables, self.independantVariables
    
    def get_trainingSet_variables(self):
        return self.independantVariables_TrainingSet, self.independantVariables_TestSet, self.dependantVariables_TrainingSet, self.dependantVariables_TestSet
    