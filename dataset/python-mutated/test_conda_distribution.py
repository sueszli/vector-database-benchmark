import pytest
from pathlib import Path
from flaml import AutoML
from sklearn.datasets import load_iris

@pytest.mark.conda
def test_package_minimum():
    if False:
        return 10
    automl = AutoML()
    automl_settings = {'time_budget': 10, 'metric': 'accuracy', 'task': 'classification', 'log_file_name': 'iris.log'}
    (X_train, y_train) = load_iris(return_X_y=True)
    automl.fit(X_train=X_train, y_train=y_train, **automl_settings)
    assert hasattr(automl, 'best_config')
    assert Path('iris.log').exists()
    assert automl.model is not None
    print(automl.model)
    preds = automl.predict_proba(X_train)
    assert preds.shape == (150, 3)
    print(preds)