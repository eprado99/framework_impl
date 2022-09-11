import numpy as np

# Gradient Descent will be our optimization algorithm, we will need a cost function to gauge accuracy in each iteration.
# We require direction and learning rate (alpha)
# Learning rate is the size of the steps that are taken to reach the min.
# We will use a cost func to measure error to provide feedback, in this case we will use mse


# Cost function
def mse(y, pred, y_size):
    # we calculate cost and "get rid of negatives"
    err = np.sum((y - pred)**2)/y_size
    return err