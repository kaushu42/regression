from utils import *

FILENAME = 'data_1d.csv'

def fit(X, Y):
    # Calculate the necessary means
    Xm = X.mean()
    Ym = Y.mean()
    XYm = (X * Y).mean()
    XXm = (X * X).mean()
    # The denominator is common for calculation of both weights
    denominator = XXm - Xm * Xm
    # Calculate the weights
    b = (Ym * XXm - Xm * XYm)/denominator
    a = (XYm - Xm * Ym)/denominator

    return a, b

def plot_fit(X, Y, Yhat):
    # Visuzlize the fit of the data
    plt.scatter(X, Y)
    plt.plot(X, Yhat, color = 'r')
    plt.show()

def main():
    # Load the data from file
    X, Y = load_data(FILENAME)
    # Visualize the data
    scatter_plot(X, Y)
    # Fit the data to a linear regression model
    a, b = fit(X, Y)
    # Make predictions based on the learned weights
    Yhat = a*X + b
    # Calculate the R-squared metric
    print('R^2 is ', r_squared(Yhat, Y))
    plot_fit(X, Y, Yhat)

if __name__ == '__main__':
    main()
