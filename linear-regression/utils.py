import numpy as np
import matplotlib.pyplot as plt

def load_data(filename, start = 0, end = -1, delim = ','):
	'''
	Loads the data from a csv file

	Args:
		filename: The file to load data from
		start: Start column for extracting data
		end: End column for data
		delim: The character to separate the file by

	Returns:
		X, Y: The features and the outputs
	'''
	data = np.loadtxt(filename, delimiter = delim)
	rows, cols = data.shape

	X = data[:, :-1].reshape(rows, cols - 1)
	Y = data[:, end].reshape(rows, 1)
	return X, Y

def scatter_plot(X, Y):
    plt.scatter(X, Y)
    plt.show()

def r_squared(Ypred, Y):
	'''
	Calculates the r-squared metric to check the fit of data

	Args:
		Ypred: The predicted outputs
		Y: The actual outputs

	Returns:
		The r-squared value
	'''
	SSres = ((Ypred - Y)**2).sum()
	SStot = ((Ypred - Y.mean())**2).sum()
	return 1 - SSres/SStot
