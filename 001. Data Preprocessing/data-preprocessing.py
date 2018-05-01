#Data Pre-processing

#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Uncomment the lines below if python can't find the csv file
"""
import os
currentFile = __file__
realPath = os.path.realpath(currentFile)
dirPath = os.path.dirname(realPath)
dataset = pd.read_csv(dirPath+'/Data.csv')
"""

#Import dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values
print ("The dataset is", dataset)
print ("The X values are", X)
print ("The y values are", y)

#Handle missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print ("X after handling missing data", X)

#Encode categorical data (data that represents category e.g. states, yes/no, countries etc.)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
labelEncoder_y = LabelEncoder()
X[:, 0] = labelEncoder_X.fit_transform(X[:, 0])
oneHotEncoder = OneHotEncoder(categorical_features=[0])  #use onehotencoding for multiple categories
X = oneHotEncoder.fit_transform(X).toarray()
y = labelEncoder_y.fit_transform(y).reshape(-1, 1) #encode and then reshape (so dimensions are not missing)
X = X[:,1:]  #ignore one column of onehotencoding, to avoid dummy variable trap
print ("X after encoding", X)

#Split the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print ("X_train\n",X_train, "\nX_test\n", X_test, "\ny_train\n",y_train, "\ny_test\n", y_test)

#Scale features - so that all features are on the same scale
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.transform(y_test)
print ("After scaling\n X_train\n",X_train, "\nX_test\n", X_test, "\ny_train\n",y_train, "\ny_test\n", y_test)