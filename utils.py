import numpy as np
import matplotlib.pyplot as plt

def load_data(filename, start = 0, end = -1):
    data = np.loadtxt(filename, delimiter = ',')
    rows, cols = data.shape

    X = data[:, :-1].reshape(rows, cols - 1)
    Y = data[:, end].reshape(rows, 1)
    return X, Y

def scatter_plot(X, Y):
    plt.scatter(X, Y)
    plt.show()

def r_squared(Ypred, Y):
    SSres = ((Ypred - Y)**2).sum()
    SStot = ((Ypred - Y.mean())**2).sum()
    return 1 - SSres/SStot
