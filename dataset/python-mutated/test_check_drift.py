import os
import pytest
from sklearn.model_selection import train_test_split
import pycaret.classification
import pycaret.datasets

def test_check_drift(tmpdir):
    if False:
        while True:
            i = 10
    data = pycaret.datasets.get_data('blood')
    experiment = pycaret.classification.ClassificationExperiment()
    experiment.setup(data, target='Class', html=False, n_jobs=1)
    file = experiment.check_drift()
    assert os.path.exists(file)

def test_check_drift_no_setup(tmpdir):
    if False:
        i = 10
        return i + 15
    data = pycaret.datasets.get_data('blood')
    (reference_data, current_data) = train_test_split(data, test_size=0.2, shuffle=False)
    experiment = pycaret.classification.ClassificationExperiment()
    with pytest.raises(ValueError):
        experiment.check_drift()
    file = experiment.check_drift(reference_data=reference_data, current_data=current_data, target='Class', categorical_features=['Recency'])
    assert os.path.exists(file)