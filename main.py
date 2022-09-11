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
print(df)

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
        print(x, y)
        # Our y hat
        pred = curr_slope * x + curr_intercept
        # Use derivatives of mse
        calculated_m += -(2 / len(data)) * x * (y - (pred))
        calculated_b += -(2 / len(data)) * (y - (pred))

    new_m = curr_slope - calculated_m * alpha
    new_b = curr_intercept - calculated_b * alpha
    return new_m, new_b


m, b = grad_descent(0, 0, df, 0.01)
print(m, b)

plt.scatter(df.H_estat, df.H_peso, color = 'blue')
plt.show()