# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load California housing data, select features and targets, and split into training and testing sets.

2.Scale both X (features) and Y (targets) using StandardScaler.

3.Use SGDRegressor wrapped in MultiOutputRegressor to train on the scaled training data.

4.Predict on test data, inverse transform the results, and calculate the mean squared error.1.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: LATHIKA LJ
RegisterNumber:  212223220050


import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

data = fetch_california_housing()

X = data.data[:, :3]

Y=np.column_stack((data.target,data.data[:, 6]))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler_X = StandardScaler()

scaler_X = StandardScaler()

X_train=scaler_X.fit_transform(X_train)
X_test=scaler_X.transform(X_test)
Y_train=scaler_Y.fit_transform(Y_train)
Y_test=scaler_Y.transform(Y_test)

sgd=SGDRegressor(max_iter=1000, tol=1e-3)

multi_output_sgd=MultiOutputRegressor(sgd)

multi_output_sgd.fit(X_train,Y_train)

Y_pred=multi_output_sgd.predict(X_test)

Y_pred=scaler_Y.inverse_transform(Y_pred)
Y_test=scaler_Y.inverse_transform(Y_test)

print("\nPredictions:\n",Y_pred[:5])*/
```

## Output:


![Screenshot 2024-09-09 112116](https://github.com/user-attachments/assets/ff5fd012-8c2c-43a4-84b7-236f7bc2b4b1)
![Screenshot 2024-09-09 112112](https://github.com/user-attachments/assets/572366a6-dcfa-4b3a-a583-0e9ff740f15e)
![Screenshot 2024-09-09 112106](https://github.com/user-attachments/assets/65aa92c8-b84a-4312-b350-c78177260db1)



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
