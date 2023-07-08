from sklearn.ensemble import RandomForestClassifier

def random_forest(X, y, params):
    model = RandomForestClassifier(**params)
    model.fit(X, y)
    return model
