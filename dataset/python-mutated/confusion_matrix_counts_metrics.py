"""Module for confusion matrix counts metrics."""
import numpy as np

def _calc_recall(tp: float, fp: float, fn: float) -> float:
    if False:
        while True:
            i = 10
    'Calculate recall for given matches and number of positives.'
    if tp + fn == 0:
        return -1
    rc = tp / (tp + fn + np.finfo(float).eps)
    return rc

def _calc_precision(tp: float, fp: float, fn: float) -> float:
    if False:
        print('Hello World!')
    'Calculate precision for given matches and number of positives.'
    if tp + fn == 0:
        return -1
    if tp + fp == 0:
        return 0
    pr = tp / (tp + fp + np.finfo(float).eps)
    return pr

def _calc_f1(tp: float, fp: float, fn: float) -> float:
    if False:
        return 10
    'Calculate F1 for given matches and number of positives.'
    if tp + fn == 0:
        return -1
    if tp + fp == 0:
        return 0
    rc = tp / (tp + fn + np.finfo(float).eps)
    pr = tp / (tp + fp + np.finfo(float).eps)
    f1 = 2 * rc * pr / (rc + pr + np.finfo(float).eps)
    return f1

def _calc_fpr(tp: float, fp: float, fn: float) -> float:
    if False:
        for i in range(10):
            print('nop')
    'Calculate FPR for given matches and number of positives.'
    if tp + fn == 0:
        return -1
    if tp + fp == 0:
        return 0
    return fp / (tp + fn + np.finfo(float).eps)

def _calc_fnr(tp: float, fp: float, fn: float) -> float:
    if False:
        for i in range(10):
            print('nop')
    'Calculate FNR for given matches and number of positives.'
    if tp + fn == 0:
        return -1
    if tp + fp == 0:
        return 1
    return fn / (tp + fn + np.finfo(float).eps)
AVAILABLE_EVALUATING_FUNCTIONS = {'recall': _calc_recall, 'fpr': _calc_fpr, 'fnr': _calc_fnr, 'precision': _calc_precision, 'f1': _calc_f1}