#Simple Linear Regression
#sourced from - superdatascience.com

#----------------------------- Data Preprocessing -------------------
#import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Uncomment the lines below if csv file is not found
"""
import os
currentFile = __file__
realPath = os.path.realpath(currentFile)
dirPath = os.path.dirname(realPath)
dataset = pd.read_csv(dirPath+'/Salary_Data.csv')
"""

#import the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values
print('dataset \n',dataset)
print('features X \n', X)
print('labels y \n',y)

#handle the missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X)
X = imputer.transform(X)
print('X after handling missing data \n',X)

#encode categorical data -- not needed for this dataset


#split the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
print('X_train \n', X_train, '\n y_train \n', y_train, '\n X_test \n', X_test, '\n y_test \n', y_test)

#feature scaling -- not needed since there is only a single feature


#----------------------------------END---------------------------------


#--------------------------------Build the model-----------------------

#create the regressor
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train,y_train)
print('regressor has been fitted with training data')

y_pred = regressor.predict(X_test)
print('predicted values y_pred\n',y_pred, '\n actual values y_test\n', y_test);

#-----------------------------------END--------------------------------


#-----------------------------------GRAPHS--------------------------------

#Plotting the training set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Experience vs Salary - Training set')
plt.xlabel('Experience')
plt.ylabel('Salary ($)')
plt.show()

#Plotting the test set results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Experience vs Salary - Test set')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

#------------------------------------END----------------------------------
