#Support Vector Regression
#sourced from superdatascience.com

#-------------------------------- Preprocessing -----------------------

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1].values.reshape(-1,1)
y = dataset.iloc[:,2].values.reshape(-1,1)

#Feature scaling : because SVR class used for Support Vector Regression doesn't have auto-scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

#-------------------------------------END------------------------------


#------------------------------------ Model ---------------------------

#Create the SVR regressor
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X,y)

#predicting a new result
y_pred = regressor.predict(sc_X.transform(6.5))
y_pred = sc_y.inverse_transform(y_pred) #get the unscaled value

#-------------------------------------END------------------------------


#----------------------------------- Graphs ---------------------------

#Visualizing the SVR results
plt.scatter(X,y,color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title("Truth or bluff (SVR)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
plt.savefig("SVR.png")

#Visualizing the SVR results in high res
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(-1,1)
plt.scatter(X,y,color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title("Truth or bluff (SVR)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
plt.savefig("SVR_highRes.png")

#-------------------------------------END------------------------------
