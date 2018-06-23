#Polynomial Linear Regression - used when data cannot be fit by a straight line, but by a curve
#General Equation is of a polynomial curve: y = b0 + b1 x1 + b2 x1^2  + b3 x1^3 + ... + bn x1^n
#sourced from superdatascience.com

#-------------------------------- Preprocessing -----------------------

#import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1].values.reshape(-1,1)  
y = dataset.iloc[:,2].values.reshape(-1,1)
print("features X \n",X,"\n labels y \n",y)

#handle missing data
#not needed as there is no missing data
"""
from sklearn.preprocessing import Imputer
imputer = Imputer()
imputer = imputer.fit(X[:,0:2])  #handle only first two columns
X[:,0:2] = imputer.transform(X[:,0:2])
print("X after handling missing data",X)
"""

#Encode categorical data
#not needed as X does not have any categorical data
"""
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelEncoder = LabelEncoder()
X[:,0] = labelEncoder.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])  #Column to be one-hot encoded
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]  #ignore column 0 so as to avoid dummy variable trap
print("X after encoding categorical data", X)
"""

#Splitting the dataset into training and test set
#not needed as the data set is too small this time
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
"""
 
#Feature Scaling
#not needed as LinearRegression class internally scales the features
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""
#-------------------------------------END------------------------------


#------------------------------------ Model ---------------------------

#Fitting polynomial regression to the dataset
    #Model Selection
        #We will iterate the degree of polynomial features until we get a best-fit model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

#Iteration 1 - degree=2
poly_reg = PolynomialFeatures(degree=2)  # y = b0 + b1 x + b2 x^2
X_poly = poly_reg.fit_transform(X)
regressor = LinearRegression()
regressor.fit(X_poly, y)
#graph
plt.scatter(X,y,color='red')
plt.plot(X, regressor.predict(X_poly), color='blue')
plt.title("level vs salary - Polynomial Regression - degree=2")
plt.xlabel("Position level")
plt.ylabel("salary")
plt.show()
plt.savefig('poly_degree2.png')


#Iteration 2 - degree=3
poly_reg = PolynomialFeatures(degree=3)
X_poly = poly_reg.fit_transform(X)
regressor = LinearRegression()
regressor.fit(X_poly, y)
#graph
plt.scatter(X,y,color='red')
plt.plot(X, regressor.predict(X_poly), color='blue')
plt.title("level vs salary - Polynomial Regression - degree=3")
plt.xlabel("Position level")
plt.ylabel("salary")
plt.show()
plt.savefig("poly_degree3.png")

#Iteration 3 - degree=4 - the graph seems a pretty good fit. Going for higher degrees 
#may lead to overfitting.
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
regressor = LinearRegression()
regressor.fit(X_poly, y)
#graph
plt.scatter(X,y,color='red')
plt.plot(X, regressor.predict(X_poly), color='blue')
plt.title("level vs salary - Polynomial Regression - degree=4")
plt.xlabel("Position level")
plt.ylabel("salary")
plt.show()
plt.savefig("poly_degree4.png")

#-------------------------------------END------------------------------


#----------------------------------- Graphs ---------------------------

#Let's show a smoother curve for the graph by adding intermediate points
X_grid = np.arange(min(X), max(X), 0.1).reshape(-1,1)  #all points between min(X) and max(X), with 0.1 interval
X_grid_poly = poly_reg.fit_transform(X_grid)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid_poly), color = 'blue')
plt.title("level vs salary - Polynomial Regression")
plt.xlabel("Position level")
plt.ylabel("salary")
plt.show()
plt.savefig("poly_final.png")

#-------------------------------------END------------------------------
