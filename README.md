# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the house dataset containing features such as area, bedrooms, location, etc., along with house price and number of occupants.
2. Preprocess the data and split it into training and testing sets.
3. Train an SGD Regressor model using the training data to learn the relationship between house features, price, and occupants.
4. Test the model with new house details and predict the house price and number of occupants.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Jayasree T S
RegisterNumber:  212224040135
*/
```
```
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
import matplotlib.pyplot as plt
 
X = np.array([
    [1, 2],
    [2, 1],
    [3, 4],
    [4, 3],
    [5, 5],
    [6, 7],
    [7, 6]
])
 
 
Y = np.array([
    [5, 8],
    [6, 9],
    [9,12],
    [10,13],
    [13,16],
    [16,20],
    [17,21]
])
 
sgd = SGDRegressor(
    max_iter=1000,
    eta0=0.01,
    learning_rate='constant',
    random_state=42
)
 
model = MultiOutputRegressor(sgd)
 
model.fit(X, Y)
 
Y_pred = model.predict(X)
 
print("\nActual Outputs")
print(Y)
 
print("\nPredicted Outputs")
print(np.round(Y_pred,2))
 
new_sample = np.array([[8, 7]])
prediction = model.predict(new_sample)
 
print("\nPrediction for", new_sample)
print(prediction)
 
 
plt.figure(figsize=(6,4))
plt.scatter(Y[:,0], Y_pred[:,0], color='blue')
plt.plot([Y[:,0].min(), Y[:,0].max()],
         [Y[:,0].min(), Y[:,0].max()],
         'r--')
plt.xlabel("Actual Output 1")
plt.ylabel("Predicted Output 1")
plt.title("Output 1: Actual vs Predicted")
plt.grid(True)
plt.show()
plt.figure(figsize=(6,4))
plt.scatter(Y[:,1], Y_pred[:,1], color='green')
plt.plot([Y[:,1].min(), Y[:,1].max()],
         [Y[:,1].min(), Y[:,1].max()],
'r--')
plt.xlabel("Actual Output 2")
plt.ylabel("Predicted Output 2")
plt.title("Output 2: Actual vs Predicted")
plt.grid(True)
plt.show()
Actual Outputs
[[ 5  8]
[ 6  9]
[ 9 12]
[10 13]
[13 16]
[16 20]
[17 21]]
Predicted Outputs
[[ 4.05  7.02]
[ 4.41  8.  ]
[ 9.18 12.24]
[ 9.53 13.22]
[13.2  16.64]
[16.87 20.06]
[17.22 21.04]
```

## Output:
![multivariate linear regression model for predicting the price of the house and number of occupants in the house](sam.png)
<img width="587" height="904" alt="image" src="https://github.com/user-attachments/assets/b49fff90-a5b3-49f4-8569-e244e7661500" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
