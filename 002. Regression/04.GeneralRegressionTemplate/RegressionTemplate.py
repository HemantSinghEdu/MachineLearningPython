#Regression Template - use this as a starting point for your regression models

#-------------------------------- Preprocessing -----------------------
#Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#Split dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)  #80-20 split

#Feature Scaling
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.transform(y_test)
"""
#-------------------------------------END------------------------------



#------------------------------------ Model ---------------------------

#Create your regressor here

#Predict a new result
y_pred = regressor.predict(8.2)

#-------------------------------------END------------------------------



#----------------------------------- Graphs ---------------------------

#Visualizing the regression results
plt.scatter(X, y, color='red')  #shows real values as dots
plt.plot(X, regressor.predict(X), color='blue')  #shows predicted values as a curve
plt.title("Truth or Bluff (Regression Model)")
plt.xlabel("Position level")
plt.ylabel('salary')
plt.show()
plt.savefig("regression.png")

#Visualizing the regression results (with higher resolution & smoother curve)
X_grid = np.arrange(min(X),max(X),0.1)  #get all points between min and max X, with interval 0.1
X_grid = X_grid.reshape((len(X_grid),1)) #reshape the array

plt.scatter(X, y, color='red')  #shows real values as dots
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title("Truth or Bluff (Regression Model)")
plt.xlabel("Position level")
plt.ylabel('salary')
plt.show()
plt.savefig("regression_highRes.png")

#-------------------------------------END------------------------------
