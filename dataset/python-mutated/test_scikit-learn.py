import pytest
from pytest_pyodide.fixture import selenium_context_manager

@pytest.mark.driver_timeout(40)
@pytest.mark.xfail_browsers(chrome='Times out in chrome', firefox='Times out in firefox')
def test_scikit_learn(selenium_module_scope):
    if False:
        i = 10
        return i + 15
    with selenium_context_manager(selenium_module_scope) as selenium:
        selenium.load_package('scikit-learn')
        assert selenium.run("\n                import numpy as np\n                import sklearn\n                from sklearn.linear_model import LogisticRegression\n\n                rng = np.random.RandomState(42)\n                X = rng.rand(100, 20)\n                y = rng.randint(5, size=100)\n\n                estimator = LogisticRegression(solver='liblinear')\n                estimator.fit(X, y)\n                print(estimator.predict(X))\n                estimator.score(X, y)\n                ") > 0

@pytest.mark.driver_timeout(40)
@pytest.mark.xfail_browsers(chrome='Times out in chrome', firefox='Times out in firefox')
def test_logistic_regression(selenium_module_scope):
    if False:
        i = 10
        return i + 15
    with selenium_context_manager(selenium_module_scope) as selenium:
        selenium.load_package('scikit-learn')
        selenium.run('\n            from sklearn.datasets import load_iris\n            from sklearn.linear_model import LogisticRegression\n            X, y = load_iris(return_X_y=True)\n            clf = LogisticRegression(random_state=0).fit(X, y)\n            print(clf.predict(X[:2, :]))\n            print(clf.predict_proba(X[:2, :]))\n            print(clf.score(X, y))\n            ')