from sklearn.neighbors import KNeighborsClassifier

def k_nearest_neighbors(X, y, params):
    model = KNeighborsClassifier(**params)
    model.fit(X, y)
    return model
