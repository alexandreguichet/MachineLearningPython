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

dataset = preprocessing.dataset 
independantVariables = preprocessing.independantVariables #Years of experience
dependantVariables = preprocessing.dependantVariables #Salary dependant of the years of experience 