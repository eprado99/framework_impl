import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Gradient Descent will be our optimization algorithm, we will need a cost function to gauge accuracy in each iteration.
# We require direction (go to local minima or local maximum) and learning rate (alpha)
# Learning rate is the size of the steps that are taken to reach the min.
# We will use a cost func to measure error to provide feedback, in this case we will use mse

# Use height - weight dataset provided in statistics class
df = pd.read_csv('EstaturaPeso_HM.csv')
# drop data corresponding to female columns to focus in only one
df = df.drop(['M_estat', 'M_peso'], axis = 1)

# Cost function
def mse(y, pred, y_size):
    # we calculate cost and "get rid of negatives"
    err = np.sum((y - pred)**2)/y_size
    return err

# Move to another file
def grad_descent(curr_slope, curr_intercept, data, alpha):
    calculated_m = 0
    calculated_b = 0

    for i in range(len(data)):
        
        x = data.iloc[i].H_estat
        y = data.iloc[i].H_peso
        # print(x, y)
        # Our y hat
        pred = curr_slope * x + curr_intercept
        # Use derivatives of mse
        calculated_m += -(2 / len(data)) * x * (y - (pred))
        calculated_b += -(2 / len(data)) * (y - (pred))

    new_m = curr_slope - calculated_m * alpha
    new_b = curr_intercept - calculated_b * alpha
    return new_m, new_b

def abline(slope, intercept):
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')

m = 0
b = 0
for i in range(1000):
    print(i)
    m, b = grad_descent(m, b, df, 0.01)

plt.scatter(df.H_estat, df.H_peso, color = 'blue')
abline(m, b)
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = np.array(df['H_estat']).reshape(-1, 1)
y = np.array(df['H_peso']).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

regr = LinearRegression()

regr.fit(X_train, y_train)

y_pred = regr.predict(X_test)
plt.scatter(X_test, y_test, color ='b')
plt.plot(X_test, y_pred, color ='k')

plt.show()


