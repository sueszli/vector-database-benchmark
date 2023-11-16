"""
This module implements a metric for inference attack worst case accuracy measurement.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import Optional, List, Tuple, Union
import numpy as np
from sklearn.metrics import roc_curve
TPR = float
FPR = float
THR = float

def _calculate_roc_for_fpr(y_true: np.ndarray, y_proba: np.ndarray, targeted_fpr: float) -> Tuple[FPR, TPR, THR]:
    if False:
        i = 10
        return i + 15
    '\n    Get FPR, TPR and, THRESHOLD based on the targeted_fpr (such that FPR <= targeted_fpr)\n    :param y_true: True attack labels.\n    :param y_proba: Predicted attack probabilities.\n    :param targeted_fpr: the targeted False Positive Rate, ROC will be calculated based on this FPR.\n    :return: tuple that contains (Achieved FPR, TPR, Threshold).\n    '
    (fpr, tpr, thr) = roc_curve(y_true=y_true, y_score=y_proba)
    if np.isnan(fpr).all() or np.isnan(tpr).all():
        logging.error('TPR or FPR values are NaN')
        raise ValueError("The targeted FPR can't be achieved.")
    targeted_fpr_idx = np.where(fpr <= targeted_fpr)[0][-1]
    return (fpr[targeted_fpr_idx], tpr[targeted_fpr_idx], thr[targeted_fpr_idx])

def get_roc_for_fpr(attack_proba: np.ndarray, attack_true: np.ndarray, target_model_labels: Optional[np.ndarray]=None, targeted_fpr: float=0.001) -> Union[List[Tuple[FPR, TPR, THR]], List[Tuple[int, FPR, TPR, THR]]]:
    if False:
        print('Hello World!')
    '\n    Compute the attack TPR, THRESHOLD and achieved FPR based on the targeted FPR. This implementation supports only\n    binary attack prediction labels {0,1}. The returned THRESHOLD defines the decision threshold on the attack\n    probabilities (meaning if p < THRESHOLD predict 0, otherwise predict 1)\n    | Related paper link: https://arxiv.org/abs/2112.03570\n\n    :param attack_proba: Predicted attack probabilities.\n    :param attack_true: True attack labels.\n    :param targeted_fpr: the targeted False Positive Rate, attack accuracy will be calculated based on this FPRs.\n     If not supplied, get_roc_for_fpr will be computed for `0.001` FPR.\n    :param target_model_labels: Original labels, if provided the Accuracy and threshold will be calculated per each\n     class separately.\n    :return: list of tuples the contains (original label (if target_model_labels is provided),\n    Achieved FPR, TPR, Threshold).\n    '
    if attack_proba.shape[0] != attack_true.shape[0]:
        raise ValueError('Number of rows in attack_pred and attack_true do not match')
    if target_model_labels is not None and attack_proba.shape[0] != target_model_labels.shape[0]:
        raise ValueError('Number of rows in target_model_labels and attack_pred do not match')
    results = []
    if target_model_labels is not None:
        (values, _) = np.unique(target_model_labels, return_counts=True)
        for value in values:
            idxs = np.where(target_model_labels == value)[0]
            (fpr, tpr, thr) = _calculate_roc_for_fpr(y_proba=attack_proba[idxs], y_true=attack_true[idxs], targeted_fpr=targeted_fpr)
            results.append((value, fpr, tpr, thr))
        return results
    (fpr, tpr, thr) = _calculate_roc_for_fpr(y_proba=attack_proba, y_true=attack_true, targeted_fpr=targeted_fpr)
    return [(fpr, tpr, thr)]

def get_roc_for_multi_fprs(attack_proba: np.ndarray, attack_true: np.ndarray, targeted_fprs: np.ndarray) -> Tuple[List[FPR], List[TPR], List[THR]]:
    if False:
        return 10
    '\n    Compute the attack ROC based on the targeted FPRs. This implementation supports only binary\n    attack prediction labels. The returned list of THRESHOLDs defines the decision threshold on the attack\n    probabilities (meaning if p < THRESHOLD predict 0, otherwise predict 1) for each provided fpr\n\n    | Related paper link: https://arxiv.org/abs/2112.03570\n\n    :param attack_proba: Predicted attack probabilities.\n    :param attack_true: True attack labels.\n    :param targeted_fprs: the set of targeted FPR (False Positive Rates), attack accuracy will be calculated based on\n    these FPRs.\n    :return: list of tuples that  (TPR, Threshold, Achieved FPR).\n    '
    if attack_proba.shape[0] != attack_true.shape[0]:
        raise ValueError('Number of rows in attack_pred and attack_true do not match')
    tpr = []
    thr = []
    fpr = []
    for t_fpr in targeted_fprs:
        res = _calculate_roc_for_fpr(y_proba=attack_proba, y_true=attack_true, targeted_fpr=t_fpr)
        fpr.append(res[0])
        tpr.append(res[1])
        thr.append(res[2])
    return (fpr, tpr, thr)