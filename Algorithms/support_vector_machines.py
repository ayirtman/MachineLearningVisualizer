from sklearn.svm import SVC

def support_vector_machines(X, y, params):
    model = SVC(**params)
    model.fit(X, y)
    return model
