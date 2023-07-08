from flask import render_template, request
from WebApp.app import app
from Algorithms import *
from Visualizations import *
import pandas as pd

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/visualize', methods=['POST'])
def visualize():
    # Get the selected dataset, algorithm, and parameters from the form
    dataset_name = request.form['dataset']
    algorithm_name = request.form['algorithm']
    params = request.form['params']

    # Load the dataset
    dataset = pd.read_csv(f'Data/{dataset_name}.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Apply the algorithm
    algorithm = getattr(Algorithms, algorithm_name)
    model = algorithm(X, y, params)

    # Generate the visualization
    visualization = getattr(Visualizations, algorithm_name)
    img = visualization(model, X, y)

    # Render the visualization page with the generated image
    return render_template('visualization.html', img=img)
