from __future__ import annotations
import copy
import math

def check_predict_proba_one(classifier, dataset):
    if False:
        while True:
            i = 10
    'predict_proba_one should return a valid probability distribution and be pure.'
    from river import utils
    if not hasattr(classifier, 'predict_proba_one'):
        return
    for (x, y) in dataset:
        (xx, yy) = (copy.deepcopy(x), copy.deepcopy(y))
        classifier = classifier.learn_one(x, y)
        y_pred = classifier.predict_proba_one(x)
        if utils.inspect.isactivelearner(classifier):
            (y_pred, _) = y_pred
        assert isinstance(y_pred, dict)
        for proba in y_pred.values():
            assert 0.0 <= proba <= 1.0
        assert math.isclose(sum(y_pred.values()), 1.0)
        assert x == xx
        assert y == yy

def check_predict_proba_one_binary(classifier, dataset):
    if False:
        return 10
    'predict_proba_one should return a dict with True and False keys.'
    for (x, y) in dataset:
        y_pred = classifier.predict_proba_one(x)
        classifier = classifier.learn_one(x, y)
        assert set(y_pred.keys()) == {False, True}

def check_multiclass_is_bool(model):
    if False:
        for i in range(10):
            print('nop')
    assert isinstance(model._multiclass, bool)