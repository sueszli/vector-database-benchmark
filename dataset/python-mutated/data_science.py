import numpy as np
import pandas as pd
from prefect import Flow, Parameter, task
from prefect.engine.results import LocalResult

@task(log_stdout=True)
def train_model(train_x: pd.DataFrame, train_y: pd.DataFrame, num_train_iter: int, learning_rate: float) -> np.ndarray:
    if False:
        while True:
            i = 10
    num_iter = num_train_iter
    lr = learning_rate
    X = train_x.to_numpy()
    Y = train_y.to_numpy()
    bias = np.ones((X.shape[0], 1))
    X = np.concatenate((bias, X), axis=1)
    weights = []
    for k in range(Y.shape[1]):
        theta = np.zeros(X.shape[1])
        y = Y[:, k]
        for _ in range(num_iter):
            z = np.dot(X, theta)
            h = _sigmoid(z)
            gradient = np.dot(X.T, h - y) / y.size
            theta -= lr * gradient
        weights.append(theta)
    print('Finish training the model.')
    return np.vstack(weights).transpose()

def _sigmoid(z):
    if False:
        while True:
            i = 10
    'A helper sigmoid function used by the training and the scoring tasks.'
    return 1 / (1 + np.exp(-z))

@task
def predict(model: np.ndarray, test_x: pd.DataFrame) -> np.ndarray:
    if False:
        while True:
            i = 10
    'Task for making predictions given a pre-trained model and a test set.'
    X = test_x.to_numpy()
    bias = np.ones((X.shape[0], 1))
    X = np.concatenate((bias, X), axis=1)
    result = _sigmoid(np.dot(X, model))
    return np.argmax(result, axis=1)

@task(log_stdout=True)
def report_accuracy(predictions: np.ndarray, test_y: pd.DataFrame) -> None:
    if False:
        return 10
    'Task for reporting the accuracy of the predictions performed by the\n    previous task. Notice that this function has no outputs, except logging.\n    '
    target = np.argmax(test_y.to_numpy(), axis=1)
    accuracy = np.sum(predictions == target) / target.shape[0]
    print(f'Model accuracy on test set: {round(accuracy * 100, 2)}')
with Flow('data-science') as flow:
    train_test_dict = LocalResult(dir='data/processed/Mon_Dec_20_2021_20:55:20').read(location='split_data_output').value
    train_x = train_test_dict['train_x']
    train_y = train_test_dict['train_y']
    test_x = train_test_dict['test_x']
    test_y = train_test_dict['test_y']
    num_train_iter = Parameter('num_train_iter', default=10000)
    learning_rate = Parameter('learning_rate', default=0.01)
    model = train_model(train_x, train_y, num_train_iter, learning_rate)
    predictions = predict(model, test_x)
    report_accuracy(predictions, test_y)
flow.run()