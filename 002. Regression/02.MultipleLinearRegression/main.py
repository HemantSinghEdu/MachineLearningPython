#Multiple Linear Regression - multiple features, one label
#sourced from superdatascience.com

#-------------------------------- Preprocessing -----------------------

#import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values.reshape(-1,1)
print("features X\n",X, "\n labels y \n",y)

#handle missing data
from sklearn.preprocessing import Imputer
imputer = Imputer()
imputer = imputer.fit(X[:,0:3]) #handle only first three columns
X[:,0:3] = imputer.transform(X[:,0:3])
print("X after handling missing data",X)

#Encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder = LabelEncoder()
X[:,3] = labelEncoder.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])  #column to be one-hot encoded
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:] #ignore column 0 so as to avoid dummy variable trap
print("X after encoding categorical data",X)

#Split dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
print("Splitting dataset into training and test sets \n X_train \n",X_train, '\n X_test \n', X_test, '\n y_train \n', y_train, '\n y_test \n', y_test)
#-------------------------------------END------------------------------


#------------------------------------ Model ---------------------------

#Create the regressor and fit it to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X,y)

#predict test set results
y_pred = regressor.predict(X_test)
print('y_pred for X_test\n',y_pred)
#-------------------------------------END------------------------------


#----------------------------------- Graphs ---------------------------
#Since there are multiple features, we can't show a feature vs . label graph
#For now, we will show the predicted vs. actual value graph

#Predicted vs. actual graph for training set
y_pred_train = regressor.predict(X_train)
plt.figure("train")
plt.scatter(y_pred_train,y_train)
plt.title("Predicted vs. Actual Profit: Training set")
plt.xlabel("Predicted Profit")
plt.ylabel("Actual Profit")
plt.show()
plt.savefig("train.png")

#Predicted vs. actual graph for training set
plt.figure("test")
plt.scatter(y_pred,y_test)
plt.title("Predicted vs. Actual Profit: Test set")
plt.xlabel("Predicted Profit")
plt.ylabel("Actual Profit")
plt.show()
plt.savefig("test.png")

#-------------------------------------END------------------------------
