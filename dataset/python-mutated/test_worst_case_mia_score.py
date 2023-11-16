from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
import pytest
from art.metrics.privacy import get_roc_for_fpr, get_roc_for_multi_fprs
from tests.utils import ARTTestException
logger = logging.getLogger(__name__)

@pytest.mark.framework_agnostic
def test_worst_case_accuracy(art_warning):
    if False:
        for i in range(10):
            print('nop')
    try:
        tpr = 1.0
        thr = 0.33
        fpr = 0.0
        y_true = np.array([1, 0, 1, 1])
        y_proba = np.array([0.35, 0.3, 0.33, 0.6])
        res = get_roc_for_fpr(attack_proba=y_proba, attack_true=y_true)[0]
        assert res[0] == fpr
        assert res[1] == tpr
        assert res[2] == thr
        print(res)
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.framework_agnostic
def test_worst_case_targeted_fpr_1(art_warning):
    if False:
        print('Hello World!')
    try:
        tpr = 1.0
        thr = 0.32
        fpr = 0.5
        y_true = np.array([1, 0, 1, 1, 1, 0])
        y_proba = np.array([0.35, 0.33, 0.32, 0.6, 0.6, 0.2])
        res = get_roc_for_fpr(attack_proba=y_proba, attack_true=y_true, targeted_fpr=0.5)[0]
        assert res[0] == fpr
        assert res[1] == tpr
        assert res[2] == thr
        print(res)
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.framework_agnostic
def test_worst_case_targeted_fpr_2(art_warning):
    if False:
        print('Hello World!')
    try:
        tpr = 0.75
        thr = 0.35
        fpr = 0.0
        y_true = np.array([1, 0, 1, 1, 1, 0])
        y_proba = np.array([0.35, 0.33, 0.32, 0.6, 0.6, 0.2])
        res = get_roc_for_fpr(attack_proba=y_proba, attack_true=y_true, targeted_fpr=0.0)[0]
        assert res[0] == fpr
        assert res[1] == tpr
        assert res[2] == thr
        print(res)
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.framework_agnostic
def test_worst_case_multiple_targeted_fpr(art_warning):
    if False:
        i = 10
        return i + 15
    try:
        y_true = np.array([1, 0, 1, 1, 1, 0])
        y_proba = np.array([0.35, 0.33, 0.32, 0.6, 0.6, 0.2])
        res = get_roc_for_multi_fprs(attack_proba=y_proba, attack_true=y_true, targeted_fprs=[0.0, 0.5])
        assert res[0][0] == 0.0
        assert res[1][0] == 0.75
        assert res[2][0] == 0.35
        assert res[0][1] == 0.5
        assert res[1][1] == 1.0
        assert res[2][1] == 0.32
        print(res)
    except ARTTestException as e:
        art_warning(e)

@pytest.mark.framework_agnostic
def test_worst_case_score_per_class(art_warning):
    if False:
        i = 10
        return i + 15
    try:
        y_true = np.array([1, 0, 1, 1, 1, 0, 1, 0])
        y_proba = np.array([0.35, 0.33, 0.32, 0.6, 0.6, 0.2, 0.9, 0.1])
        target_model_labels = np.array([1, 1, 1, 1, 2, 1, 2, 2])
        res = get_roc_for_fpr(attack_proba=y_proba, attack_true=y_true, target_model_labels=target_model_labels)
        print(res)
    except ARTTestException as e:
        art_warning(e)