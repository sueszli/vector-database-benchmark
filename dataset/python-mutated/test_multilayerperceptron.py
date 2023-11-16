import numpy as np
from sklearn.base import clone
from mlxtend.classifier import MultiLayerPerceptron as MLP
from mlxtend.data import iris_data
from mlxtend.utils import assert_raises
(X, y) = iris_data()
X = X[:, [0, 3]]
X_bin = X[0:100]
y_bin = y[0:100]
X_bin[:, 0] = (X_bin[:, 0] - X_bin[:, 0].mean()) / X_bin[:, 0].std()
X_bin[:, 1] = (X_bin[:, 1] - X_bin[:, 1].mean()) / X_bin[:, 1].std()
X[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

def test_multiclass_gd_acc():
    if False:
        while True:
            i = 10
    mlp = MLP(epochs=20, eta=0.05, hidden_layers=[10], minibatches=1, random_seed=1)
    mlp.fit(X, y)
    assert round(mlp.cost_[0], 2) == 0.55, mlp.cost_[0]
    assert round(mlp.cost_[-1], 2) == 0.01, mlp.cost_[-1]
    assert (y == mlp.predict(X)).all()

def test_progress_1():
    if False:
        return 10
    mlp = MLP(epochs=1, eta=0.05, hidden_layers=[10], minibatches=1, print_progress=1, random_seed=1)
    mlp.fit(X, y)

def test_progress_2():
    if False:
        return 10
    mlp = MLP(epochs=1, eta=0.05, hidden_layers=[10], minibatches=1, print_progress=2, random_seed=1)
    mlp.fit(X, y)

def test_progress_3():
    if False:
        return 10
    mlp = MLP(epochs=1, eta=0.05, hidden_layers=[10], minibatches=1, print_progress=3, random_seed=1)
    mlp.fit(X, y)

def test_predict_proba():
    if False:
        while True:
            i = 10
    mlp = MLP(epochs=20, eta=0.05, hidden_layers=[10], minibatches=1, random_seed=1)
    mlp.fit(X, y)
    pred = mlp.predict_proba(X[0, np.newaxis])
    exp = np.array([[0.6, 0.2, 0.2]])
    np.testing.assert_almost_equal(pred, exp, decimal=1)

def test_multiclass_sgd_acc():
    if False:
        i = 10
        return i + 15
    mlp = MLP(epochs=20, eta=0.05, hidden_layers=[25], minibatches=len(y), random_seed=1)
    mlp.fit(X, y)
    assert round(mlp.cost_[-1], 3) == 0.023, mlp.cost_[-1]
    assert (y == mlp.predict(X)).all()

def test_multiclass_minibatch_acc():
    if False:
        for i in range(10):
            print('nop')
    mlp = MLP(epochs=20, eta=0.05, hidden_layers=[25], minibatches=5, random_seed=1)
    mlp.fit(X, y)
    assert round(mlp.cost_[-1], 3) == 0.024, mlp.cost_[-1]
    assert (y == mlp.predict(X)).all()

def test_num_hidden_layers():
    if False:
        for i in range(10):
            print('nop')
    assert_raises(AttributeError, 'Currently, only 1 hidden layer is supported', MLP, 20, 0.05, [25, 10])

def test_binary_gd():
    if False:
        for i in range(10):
            print('nop')
    mlp = MLP(epochs=20, eta=0.05, hidden_layers=[25], minibatches=5, random_seed=1)
    mlp.fit(X_bin, y_bin)
    assert (y_bin == mlp.predict(X_bin)).all()

def test_score_function():
    if False:
        return 10
    mlp = MLP(epochs=20, eta=0.05, hidden_layers=[25], minibatches=5, random_seed=1)
    mlp.fit(X, y)
    acc = mlp.score(X, y)
    assert acc == 1.0, acc

def test_decay_function():
    if False:
        while True:
            i = 10
    mlp = MLP(epochs=20, eta=0.05, decrease_const=0.01, hidden_layers=[25], minibatches=5, random_seed=1)
    mlp.fit(X, y)
    assert mlp._decr_eta < mlp.eta
    acc = mlp.score(X, y)
    assert round(acc, 2) == 0.98, acc

def test_momentum_1():
    if False:
        while True:
            i = 10
    mlp = MLP(epochs=20, eta=0.05, momentum=0.1, hidden_layers=[25], minibatches=len(y), random_seed=1)
    mlp.fit(X, y)
    assert round(mlp.cost_[-1], 4) == 0.0057, mlp.cost_[-1]
    assert (y == mlp.predict(X)).all()

def test_retrain():
    if False:
        print('Hello World!')
    mlp = MLP(epochs=5, eta=0.05, hidden_layers=[10], minibatches=len(y), random_seed=1)
    mlp.fit(X, y)
    cost_1 = mlp.cost_[-1]
    mlp.fit(X, y)
    cost_2 = mlp.cost_[-1]
    mlp.fit(X, y, init_params=False)
    cost_3 = mlp.cost_[-1]
    assert cost_2 == cost_1
    assert cost_3 < cost_2 / 2.0

def test_clone():
    if False:
        i = 10
        return i + 15
    mlp = MLP(epochs=5, eta=0.05, hidden_layers=[10], minibatches=len(y), random_seed=1)
    clone(mlp)