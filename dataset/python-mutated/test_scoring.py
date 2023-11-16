import numpy as np
from mlxtend.evaluate import scoring

def test_metric_argument():
    if False:
        i = 10
        return i + 15
    'Test exception is raised when user provides invalid metric argument'
    try:
        scoring(y_target=[1], y_predicted=[1], metric='test')
        assert False
    except AttributeError:
        assert True

def test_y_arguments():
    if False:
        while True:
            i = 10
    'Test exception is raised when user provides invalid vectors'
    try:
        scoring(y_target=[1, 2], y_predicted=[1])
        assert False
    except AttributeError:
        assert True

def test_accuracy():
    if False:
        for i in range(10):
            print('nop')
    'Test accuracy metric'
    y_targ = [1, 1, 1, 0, 0, 2, 0, 3]
    y_pred = [1, 0, 1, 0, 0, 2, 1, 3]
    res = scoring(y_target=y_targ, y_predicted=y_pred, metric='accuracy')
    assert res == 0.75

def test_error():
    if False:
        i = 10
        return i + 15
    'Test error metric'
    y_targ = [1, 1, 1, 0, 0, 2, 0, 3]
    y_pred = [1, 0, 1, 0, 0, 2, 1, 3]
    res = scoring(y_target=y_targ, y_predicted=y_pred, metric='error')
    assert res == 0.25

def test_binary():
    if False:
        while True:
            i = 10
    'Test exception is raised if label is not binary in f1'
    try:
        y_targ = [1, 1, 1, 0, 0, 2, 0, 3]
        y_pred = [1, 0, 1, 0, 0, 2, 1, 3]
        scoring(y_target=y_targ, y_predicted=y_pred, metric='f1')
        assert False
    except AttributeError:
        assert True

def test_precision():
    if False:
        i = 10
        return i + 15
    y_targ = [1, 1, 1, 0, 0, 1, 0, 1]
    y_pred = [1, 0, 1, 0, 0, 0, 1, 1]
    res = scoring(y_target=y_targ, y_predicted=y_pred, metric='precision')
    assert round(res, 3) == 0.75, res

def test_recall():
    if False:
        i = 10
        return i + 15
    y_targ = [1, 1, 1, 0, 0, 1, 0, 1]
    y_pred = [1, 0, 1, 0, 0, 0, 1, 1]
    res = scoring(y_target=y_targ, y_predicted=y_pred, metric='recall')
    assert round(res, 3) == 0.6, res

def test_truepositiverate():
    if False:
        print('Hello World!')
    y_targ = [1, 1, 1, 0, 0, 1, 0, 1]
    y_pred = [1, 0, 1, 0, 0, 0, 1, 1]
    res = scoring(y_target=y_targ, y_predicted=y_pred, metric='true_positive_rate')
    assert round(res, 3) == 0.6, res

def test_falsepositiverate():
    if False:
        return 10
    y_targ = [1, 1, 1, 0, 0, 1, 0, 1]
    y_pred = [1, 0, 1, 0, 0, 0, 1, 1]
    res = scoring(y_target=y_targ, y_predicted=y_pred, metric='false_positive_rate')
    assert round(res, 3) == 0.333, res

def test_specificity():
    if False:
        print('Hello World!')
    y_targ = [1, 1, 1, 0, 0, 1, 0, 1]
    y_pred = [1, 0, 1, 0, 0, 0, 1, 1]
    res = scoring(y_target=y_targ, y_predicted=y_pred, metric='specificity')
    assert round(res, 3) == 0.667, res

def test_sensitivity():
    if False:
        while True:
            i = 10
    y_targ = [1, 1, 1, 0, 0, 1, 0, 1]
    y_pred = [1, 0, 1, 0, 0, 0, 1, 1]
    res = scoring(y_target=y_targ, y_predicted=y_pred, metric='sensitivity')
    assert round(res, 3) == 0.6, res

def test_f1():
    if False:
        i = 10
        return i + 15
    y_targ = [1, 1, 1, 0, 0, 1, 0, 1]
    y_pred = [1, 0, 1, 0, 0, 0, 1, 1]
    res = scoring(y_target=y_targ, y_predicted=y_pred, metric='f1')
    assert round(res, 3) == 0.667, res

def test_matthews_corr_coef():
    if False:
        return 10
    y_targ = [1, 1, 1, 0, 0, 1, 0, 1]
    y_pred = [1, 0, 1, 0, 0, 0, 1, 1]
    res = scoring(y_target=y_targ, y_predicted=y_pred, metric='matthews_corr_coef')
    assert round(res, 3) == 0.258, res

def test_balanced_accuracy():
    if False:
        while True:
            i = 10
    y_targ = np.array([0, 0, 0, 1, 1, 1, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 0, 1, 1, 2, 2, 2, 2])
    res = scoring(y_target=y_targ, y_predicted=y_pred, metric='balanced accuracy')
    assert round(res, 3) == 0.578, res

def test_avg_perclass_accuracy():
    if False:
        for i in range(10):
            print('nop')
    y_targ = np.array([0, 0, 0, 1, 1, 1, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 0, 1, 1, 2, 2, 2, 2])
    res = scoring(y_target=y_targ, y_predicted=y_pred, metric='average per-class accuracy')
    assert round(res, 3) == 0.667, res

def test_avg_perclass_error():
    if False:
        i = 10
        return i + 15
    y_targ = np.array([0, 0, 0, 1, 1, 1, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 0, 1, 1, 2, 2, 2, 2])
    res = scoring(y_target=y_targ, y_predicted=y_pred, metric='average per-class error')
    assert round(res, 3) == 0.333, res