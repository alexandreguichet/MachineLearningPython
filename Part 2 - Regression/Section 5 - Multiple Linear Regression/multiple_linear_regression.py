# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 17:49:25 2019

@author: Alex
Multible Linear Regression
"""

#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data_preprocessing import PreprocessingData

preprocessing = PreprocessingData()
preprocessing.import_dataset("50_Startups.csv")
preprocessing.adjust_missing_data()
preprocessing.encode_categorical_data()
preprocessing.seperate_training_test_set(test_size = 1/3, random_state = 0)
dataset, independantVariables, dependantVariables = preprocessing.get_imported_variables()
independantVariables_TrainingSet, independantVariables_TestSet, dependantVariables_TrainingSet, dependantVariables_TestSet = preprocessing.get_trainingSet_variables()