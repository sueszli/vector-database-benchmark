from unittest import mock
import numpy as np
import autokeras as ak
from autokeras import test_utils

@mock.patch('autokeras.AutoModel.fit')
def test_txt_clf_fit_call_auto_model_fit(fit, tmp_path):
    if False:
        print('Hello World!')
    auto_model = ak.TextClassifier(directory=tmp_path, seed=test_utils.SEED)
    auto_model.fit(x=np.array(['a b c', 'b b c']), y=np.array([1, 2]))
    assert fit.is_called

@mock.patch('autokeras.AutoModel.fit')
def test_txt_reg_fit_call_auto_model_fit(fit, tmp_path):
    if False:
        print('Hello World!')
    auto_model = ak.TextRegressor(directory=tmp_path, seed=test_utils.SEED)
    auto_model.fit(x=np.array(['a b c', 'b b c']), y=np.array([1.0, 2.0]))
    assert fit.is_called