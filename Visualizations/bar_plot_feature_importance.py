import matplotlib.pyplot as plt

def bar_plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)

    plt.figure()
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()
