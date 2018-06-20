from utils import *

FILENAME = 'data_2d.csv'

def fit(X, Y):
    Xt = X.transpose()
    weights = np.linalg.solve(Xt.dot(X), Xt.dot(Y))
    return weights

def main():
    # Load the data
    X, Y = load_data(FILENAME)
    # We need to append a column containing all 1's to account for the bias term
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    # Fit the data with linear regression
    w = fit(X, Y)
    # Predict using the learnt model
    Yhat = X.dot(w)
    # Calculate the R-squared metric
    print('R^2 is ', r_squared(Yhat, Y))

if __name__ == '__main__':
    main()
