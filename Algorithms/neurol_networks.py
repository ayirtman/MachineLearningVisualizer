from sklearn.neural_network import MLPClassifier

def neural_network(X, y, params):
    model = MLPClassifier(**params)
    model.fit(X, y)
    return model
