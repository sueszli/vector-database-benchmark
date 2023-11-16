from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import pytest
from art.attacks.inference.membership_inference.label_only_gap_attack import LabelOnlyGapAttack
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
        classifier = image_dl_estimator_for_attack(LabelOnlyGapAttack)
        attack = LabelOnlyGapAttack(classifier)
        backend_check_membership_accuracy_no_fit(attack, get_default_mnist_subset, 0.8)
    except ARTTestException as e:
        art_warning(e)

def test_rule_based_tabular(art_warning, get_iris_dataset, tabular_dl_estimator_for_attack):
    if False:
        while True:
            i = 10
    try:
        classifier = tabular_dl_estimator_for_attack(LabelOnlyGapAttack)
        attack = LabelOnlyGapAttack(classifier)
        backend_check_membership_accuracy_no_fit(attack, get_iris_dataset, 0.06)
    except ARTTestException as e:
        art_warning(e)

def test_classifier_type_check_fail(art_warning):
    if False:
        i = 10
        return i + 15
    try:
        backend_test_classifier_type_check_fail(LabelOnlyGapAttack, [BaseEstimator, ClassifierMixin])
    except ARTTestException as e:
        art_warning(e)

def backend_check_membership_accuracy_no_fit(attack, dataset, approx):
    if False:
        while True:
            i = 10
    ((x_train, y_train), (x_test, y_test)) = dataset
    inferred_train = attack.infer(x_train, y_train)
    inferred_test = attack.infer(x_test, y_test)
    backend_check_accuracy(inferred_train, inferred_test, approx)

def backend_check_accuracy(inferred_train, inferred_test, approx):
    if False:
        i = 10
        return i + 15
    train_pos = sum(inferred_train) / len(inferred_train)
    test_pos = sum(inferred_test) / len(inferred_test)
    assert train_pos > test_pos or train_pos == pytest.approx(test_pos, abs=approx) or test_pos == 1