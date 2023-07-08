from sklearn.tree import DecisionTreeClassifier

def decision_tree(X, y, params):
    model = DecisionTreeClassifier(**params)
    model.fit(X, y)
    return model
