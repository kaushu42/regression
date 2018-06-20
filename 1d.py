import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('data_1d.csv', delimiter = ',')
X = data[:, 0].reshape(data.shape[0], 1)
Y = data[:, 1].reshape(data.shape[0], 1)
plt.scatter(X, Y)
plt.show()
Xm = X.mean()
Ym = Y.mean()
XYm = (X * Y).mean()
XXm = (X * X).mean()

denominator = XXm - Xm * Xm

b = (Ym * XXm - Xm * XYm)/denominator
a = (XYm - Xm * Ym)/denominator

Yhat = a*X + b

plt.scatter(X, Y)
plt.plot(X, Yhat, color = 'r')
plt.show()
