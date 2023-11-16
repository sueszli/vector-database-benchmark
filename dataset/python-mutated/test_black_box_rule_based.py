from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import pytest
import numpy as np
from art.attacks.inference.membership_inference.black_box_rule_based import MembershipInferenceBlackBoxRuleBased
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin
from tests.attacks.utils import backend_test_classifier_type_check_fail
from tests.utils import ARTTestException
logger = logging.getLogger(__name__)
attack_train_ratio = 0.5
num_classes_iris = 3
num_classes_mnist = 10

def test_rule_based_image(art_warning, get_default_mnist_subset, image_dl_estimator_for_attack):
    if False:
        return 10
    try:
        classifier = image_dl_estimator_for_attack(MembershipInferenceBlackBoxRuleBased)
        attack = MembershipInferenceBlackBoxRuleBased(classifier)
        backend_check_membership_accuracy_no_fit(attack, get_default_mnist_subset, 0.8)
    except ARTTestException as e:
        art_warning(e)

def test_rule_based_tabular(art_warning, get_iris_dataset, tabular_dl_estimator_for_attack):
    if False:
        while True:
            i = 10
    try:
        classifier = tabular_dl_estimator_for_attack(MembershipInferenceBlackBoxRuleBased)
        attack = MembershipInferenceBlackBoxRuleBased(classifier)
        backend_check_membership_accuracy_no_fit(attack, get_iris_dataset, 0.06)
    except ARTTestException as e:
        art_warning(e)

def test_rule_based_tabular_prob(art_warning, get_iris_dataset, tabular_dl_estimator_for_attack):
    if False:
        for i in range(10):
            print('nop')
    try:
        classifier = tabular_dl_estimator_for_attack(MembershipInferenceBlackBoxRuleBased)
        attack = MembershipInferenceBlackBoxRuleBased(classifier)
        backend_check_membership_probabilities(attack, get_iris_dataset)
    except ARTTestException as e:
        art_warning(e)

def test_classifier_type_check_fail(art_warning):
    if False:
        return 10
    try:
        backend_test_classifier_type_check_fail(MembershipInferenceBlackBoxRuleBased, [BaseEstimator, ClassifierMixin])
    except ARTTestException as e:
        art_warning(e)

def backend_check_membership_accuracy_no_fit(attack, dataset, approx):
    if False:
        for i in range(10):
            print('nop')
    ((x_train, y_train), (x_test, y_test)) = dataset
    inferred_train = attack.infer(x_train, y_train)
    inferred_test = attack.infer(x_test, y_test)
    backend_check_accuracy(inferred_train, inferred_test, approx)

def backend_check_accuracy(inferred_train, inferred_test, approx):
    if False:
        for i in range(10):
            print('nop')
    train_pos = sum(inferred_train) / len(inferred_train)
    test_pos = sum(inferred_test) / len(inferred_test)
    assert train_pos > test_pos or train_pos == pytest.approx(test_pos, abs=approx) or test_pos == 1

def backend_check_membership_probabilities(attack, dataset):
    if False:
        return 10
    ((x_train, y_train), _) = dataset
    inferred_train_pred = attack.infer(x_train, y_train)
    inferred_train_prob = attack.infer(x_train, y_train, probabilities=True)
    backend_check_probabilities(inferred_train_pred, inferred_train_prob)

def backend_check_probabilities(pred, prob):
    if False:
        for i in range(10):
            print('nop')
    assert prob.shape[1] == 2
    assert np.all(np.sum(prob, axis=1) == 1)
    assert np.all(np.argmax(prob, axis=1) == pred.astype(int))