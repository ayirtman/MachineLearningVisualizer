from sklearn.linear_model import LogisticRegression

def logistic_regression(X, y, params):
    model = LogisticRegression(**params)
    model.fit(X, y)
    return model
