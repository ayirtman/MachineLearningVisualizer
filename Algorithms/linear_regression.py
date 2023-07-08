from sklearn.linear_model import LinearRegression

def linear_regression(X, y, params):
    model = LinearRegression(**params)
    model.fit(X, y)
    return model
