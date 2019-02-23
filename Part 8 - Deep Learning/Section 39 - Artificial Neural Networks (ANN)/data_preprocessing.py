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

    #Splitting the dataset into the Training set and Test set
    def seperate_training_test_set(self, test_size_training = 0.2, random_state_training = 0):
        from sklearn.model_selection import train_test_split
        self.independantVariables_TrainingSet, self.independantVariables_TestSet, self.dependantVariables_TrainingSet, self.dependantVariables_TestSet = train_test_split(self.independantVariables, 
                                                                                                                                              self.dependantVariables, 
                                                                                                                                              test_size = test_size_training,
                                                                                                                                              random_state = random_state_training)
   
    def separate_independant_dependant_variables(self, indVariables, depVariables): #Separate the dataset into independant and dependant variables by selecting which colmn goes where
        self.independantVariables = self.dataset.iloc[:, indVariables:depVariables - 1].values # The independant variables of the dataset imported [Country, Age, Salary]
        self.dependantVariables = self.dataset.iloc[:, depVariables].values # The dependant variables of the dataset impo                                                                                                                       test_size = test_size_training, 
                                                                                                                                          
    #Taking care of missing data
    def adjust_missing_data(self):
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
        imputer.fit(self.independantVariables[:, 1:3])
        self.independantVariables[:, 1:3] = imputer.transform(self.independantVariables[:, 1:3])


    #Encoding categorical data
    def encode_categorical_data(self, has_dependant_categorical = False):       
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer
        X = []
        for x in range (0, self.independantVariables.shape[1]):
            if type(self.independantVariables[0, x]) == str:
                X.append(x)
        print(X)
        ct = ColumnTransformer([('oh_enc', OneHotEncoder(sparse=False), X),],  # the column numbers I want to apply this to
    remainder='passthrough'  # This leaves the rest of my columns in place
)
        self.independantVariables = ct.fit_transform(self.independantVariables)
        self.avoid_dummy_variable_trap()

    #TODO: change the line bellow if we need more than two variables output (Else than a yes/no answer)
        if has_dependant_categorical:
            X2 = []
            for x in range (0, self.dependantVariables.shape[1]):
                if type(self.dependantVariables[1, x]) == str:
                    X.append(x)
            ct = ColumnTransformer([('oh_enc', OneHotEncoder(sparse=False), X2),],  # the column numbers I want to apply this to
    remainder='passthrough'  # This leaves the rest of my columns in place
)
            self.dependantVariables = ct.fit_transform(self.dependantVariables)

    #Feature scaling
    def feature_scaling(self):
        from sklearn.preprocessing import StandardScaler
        scale_independantVariables= StandardScaler()
        self.independantVariables_TrainingSet = scale_independantVariables.fit_transform(self.independantVariables_TrainingSet)
        self.independantVariables_TestSet = scale_independantVariables.transform(self.independantVariables_TestSet)
        
    #Avoiding the Dummy Variable Trap
    def avoid_dummy_variable_trap(self):
        self.independantVariables = self.independantVariables[:, 1:] 
        
    #Getters for variables
    def get_imported_variables(self):
        return self.dataset, self.independantVariables, self.dependantVariables
    
    def get_trainingSet_variables(self):
        return self.independantVariables_TrainingSet, self.independantVariables_TestSet, self.dependantVariables_TrainingSet, self.dependantVariables_TestSet
    
