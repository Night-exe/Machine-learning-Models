# A new file for Machine learning model

# Libraries to be used 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/delaney_solubility_with_descriptors.csv")
# This perticular dataset has a logS function
# print(df)

# Defining X and Y
y = df['logS']
x = df.drop('logS', axis=1)

# Assigning train and test variables
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=100)

# Using Linear Regression model for evaulation
lr = LinearRegression()
lr.fit(X_train, y_train)

# Defining the prediction variables
y_lr_train_pred = lr.predict(X_train)
y_lr_test_pred = lr.predict(X_test)

# Taking the error and r2 scores
lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

# Making the results in a dataframe for clean viewing
lr_results = pd.DataFrame(["Linear Regression", lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ['Method', 'Training MSE', 'Training R2', 'Testing MSE', 'Testing R2']

# Random Forest Regression

# Using Random Forest Regressor
rf = RandomForestRegressor()
rf.fit(X_train, y_train)

# Defining the predicting variables
y_rf_train_pred = rf.predict(X_train)
y_rf_test_pred = rf.predict(X_test)

# Taking the error and r2 scores
rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)

rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)

# Organising the data in a dataframe for cleaniness
rf_results = pd.DataFrame(["Random Forest Regression", rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
rf_results.columns = ['Method', 'Training MSE', 'Training R2', 'Testing MSE', 'Testing R2']
# print(lr_results)

# Comparision of both the models
df_models = pd.concat([lr_results, rf_results], axis=0).reset_index(drop=True)
print(df_models)

# Ploting the graph and showing it

plt.scatter(x=y_train, y=y_lr_train_pred, c='#7CAE00', alpha=0.3)

z= np.polyfit(y_train, y_lr_train_pred, 1)
p = np.poly1d(z)

plt.plot(y_train, p(y_train), '#FA2030')

plt.xlabel('Experimental LogS')
plt.ylabel('Predict LogS')

# Showing the Graph


##

plt.show()