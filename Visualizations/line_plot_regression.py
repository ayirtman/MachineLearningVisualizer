import matplotlib.pyplot as plt

def line_plot_regression(model, X, y):
    plt.scatter(X, y, color='blue')
    plt.plot(X, model.predict(X), color='red')
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.show()
