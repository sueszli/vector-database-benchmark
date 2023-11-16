from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import pytest
import numpy as np
from art.attacks.inference.attribute_inference.white_box_decision_tree import AttributeInferenceWhiteBoxDecisionTree
from art.estimators.classification.scikitlearn import ScikitlearnDecisionTreeClassifier
from tests.attacks.utils import backend_test_classifier_type_check_fail
from tests.utils import ARTTestException
logger = logging.getLogger(__name__)

@pytest.mark.skip_framework('dl_frameworks')
def test_white_box(art_warning, decision_tree_estimator, get_iris_dataset):
    if False:
        for i in range(10):
            print('nop')
    try:
        attack_feature = 2
        values = [0.14, 0.42, 0.71]
        priors = [50 / 150, 54 / 150, 46 / 150]
        ((x_train_iris, y_train_iris), (x_test_iris, y_test_iris)) = get_iris_dataset
        x_train_for_attack = np.delete(x_train_iris, attack_feature, 1)
        x_train_feature = x_train_iris[:, attack_feature]
        x_test_for_attack = np.delete(x_test_iris, attack_feature, 1)
        x_test_feature = x_test_iris[:, attack_feature]
        classifier = decision_tree_estimator()
        attack = AttributeInferenceWhiteBoxDecisionTree(classifier, attack_feature=attack_feature)
        x_train_predictions = np.array([np.argmax(arr) for arr in classifier.predict(x_train_iris)]).reshape(-1, 1)
        x_test_predictions = np.array([np.argmax(arr) for arr in classifier.predict(x_test_iris)]).reshape(-1, 1)
        inferred_train = attack.infer(x_train_for_attack, x_train_predictions, values=values, priors=priors)
        inferred_test = attack.infer(x_test_for_attack, x_test_predictions, values=values, priors=priors)
        train_diff = np.abs(inferred_train - x_train_feature.reshape(1, -1))
        test_diff = np.abs(inferred_test - x_test_feature.reshape(1, -1))
        assert np.sum(train_diff) / len(inferred_train) == pytest.approx(0.2108, abs=0.03)
        assert np.sum(test_diff) / len(inferred_test) == pytest.approx(0.1988, abs=0.03)
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.skip_framework('dl_frameworks')
def test_check_params(art_warning, decision_tree_estimator):
    if False:
        for i in range(10):
            print('nop')
    try:
        classifier = decision_tree_estimator()
        with pytest.raises(ValueError):
            _ = AttributeInferenceWhiteBoxDecisionTree(classifier, attack_feature=-5)
    except ARTTestException as e:
        art_warning(e)

def test_classifier_type_check_fail():
    if False:
        print('Hello World!')
    backend_test_classifier_type_check_fail(AttributeInferenceWhiteBoxDecisionTree, (ScikitlearnDecisionTreeClassifier,))