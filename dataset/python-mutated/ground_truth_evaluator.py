"""
This module implements classes to evaluate the performance of poison detection methods.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import json
import logging
from typing import Tuple, Union, List
import numpy as np
logger = logging.getLogger(__name__)

class GroundTruthEvaluator:
    """
    Class to evaluate the performance of the poison detection method.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        '\n        Evaluates ground truth constructor\n        '

    def analyze_correctness(self, assigned_clean_by_class: Union[np.ndarray, List[int], List[np.ndarray]], is_clean_by_class: list) -> Tuple[np.ndarray, str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        For each training sample, determine whether the activation clustering method was correct.\n\n        :param assigned_clean_by_class: Result of clustering.\n        :param is_clean_by_class: is clean separated by class.\n        :return: Two variables are returned:\n                 1) all_errors_by_class[i]: an array indicating the correctness of each assignment\n                 in the ith class. Such that:\n                 all_errors_by_class[i] = 0 if marked poison, is poison\n                 all_errors_by_class[i] = 1 if marked clean, is clean\n                 all_errors_by_class[i] = 2 if marked poison, is clean\n                 all_errors_by_class[i] = 3 marked clean, is poison\n                 2) Json object with confusion matrix per-class.\n        '
        all_errors_by_class = []
        poison = 0
        clean = 1
        dic_json = {}
        logger.debug('Error rates per class:')
        for (class_i, (assigned_clean, is_clean)) in enumerate(zip(assigned_clean_by_class, is_clean_by_class)):
            errors = []
            for (assignment, bl_var) in zip(assigned_clean, is_clean):
                bl_var = int(bl_var)
                if assignment == poison and bl_var == poison:
                    errors.append(0)
                elif assignment == clean and bl_var == clean:
                    errors.append(1)
                elif assignment == poison and bl_var == clean:
                    errors.append(2)
                elif assignment == clean and bl_var == poison:
                    errors.append(3)
                else:
                    raise Exception('Analyze_correctness entered wrong class')
            errors_array = np.asarray(errors)
            logger.debug('-------------------%d---------------', class_i)
            key_i = 'class_' + str(class_i)
            matrix_i = self.get_confusion_matrix(errors_array)
            dic_json.update({key_i: matrix_i})
            all_errors_by_class.append(errors_array)
        all_errors_by_class_array = np.asarray(all_errors_by_class, dtype=object)
        conf_matrix_json = json.dumps(dic_json)
        return (all_errors_by_class_array, conf_matrix_json)

    def get_confusion_matrix(self, values: np.ndarray) -> dict:
        if False:
            i = 10
            return i + 15
        '\n        Computes and returns a json object that contains the confusion matrix for each class.\n\n        :param values: Array indicating the correctness of each assignment in the ith class.\n        :return: Json object with confusion matrix per-class.\n        '
        dic_class = {}
        true_positive = np.where(values == 0)[0].shape[0]
        true_negative = np.where(values == 1)[0].shape[0]
        false_positive = np.where(values == 2)[0].shape[0]
        false_negative = np.where(values == 3)[0].shape[0]
        tp_rate = self.calculate_and_print(true_positive, true_positive + false_negative, 'true-positive rate')
        tn_rate = self.calculate_and_print(true_negative, false_positive + true_negative, 'true-negative rate')
        fp_rate = self.calculate_and_print(false_positive, false_positive + true_negative, 'false-positive rate')
        fn_rate = self.calculate_and_print(false_negative, true_positive + false_negative, 'false-negative rate')
        dic_tp = dict(rate=round(tp_rate, 2), numerator=true_positive, denominator=true_positive + false_negative)
        if true_positive + false_negative == 0:
            dic_tp = dict(rate='N/A', numerator=true_positive, denominator=true_positive + false_negative)
        dic_tn = dict(rate=round(tn_rate, 2), numerator=true_negative, denominator=false_positive + true_negative)
        if false_positive + true_negative == 0:
            dic_tn = dict(rate='N/A', numerator=true_negative, denominator=false_positive + true_negative)
        dic_fp = dict(rate=round(fp_rate, 2), numerator=false_positive, denominator=false_positive + true_negative)
        if false_positive + true_negative == 0:
            dic_fp = dict(rate='N/A', numerator=false_positive, denominator=false_positive + true_negative)
        dic_fn = dict(rate=round(fn_rate, 2), numerator=false_negative, denominator=true_positive + false_negative)
        if true_positive + false_negative == 0:
            dic_fn = dict(rate='N/A', numerator=false_negative, denominator=true_positive + false_negative)
        dic_class.update(dict(TruePositive=dic_tp))
        dic_class.update(dict(TrueNegative=dic_tn))
        dic_class.update(dict(FalsePositive=dic_fp))
        dic_class.update(dict(FalseNegative=dic_fn))
        return dic_class

    @staticmethod
    def calculate_and_print(numerator: int, denominator: int, name: str) -> float:
        if False:
            return 10
        '\n        Computes and prints the rates based on the denominator provided.\n\n        :param numerator: number used to compute the rate.\n        :param denominator: number used to compute the rate.\n        :param name: Rate name being computed e.g., false-positive rate.\n        :return: Computed rate\n        '
        try:
            res = 100 * (numerator / float(denominator))
            logger.debug('%s: %d/%d=%.3g', name, numerator, denominator, res)
            return res
        except ZeroDivisionError:
            logger.debug("%s: couldn't calculate %d/%d", name, numerator, denominator)
            return 0.0