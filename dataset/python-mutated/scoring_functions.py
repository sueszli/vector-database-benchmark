"""
Scanner scoring functions.
"""
import numpy as np

class ScoringFunctions:
    """
    Scanner scoring functions. These functions are used in the scanner to determine the score of a subset.
    """

    @staticmethod
    def get_score_bj_fast(n_alpha: np.ndarray, no_records: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        if False:
            while True:
                i = 10
        '\n        BerkJones\n\n        :param n_alpha: Number of records less than alpha.\n        :param no_records: Number of records.\n        :param alpha: Alpha threshold.\n        :return: Score.\n        '
        score = np.zeros(alpha.shape[0])
        inds_tie = n_alpha == no_records
        inds_not_tie = np.logical_not(inds_tie)
        inds_pos = n_alpha > no_records * alpha
        inds_pos_not_tie = np.logical_and(inds_pos, inds_not_tie)
        score[inds_tie] = no_records[inds_tie] * np.log(np.true_divide(1, alpha[inds_tie]))
        factor1 = n_alpha[inds_pos_not_tie] * np.log(np.true_divide(n_alpha[inds_pos_not_tie], no_records[inds_pos_not_tie] * alpha[inds_pos_not_tie]))
        factor2 = no_records[inds_pos_not_tie] - n_alpha[inds_pos_not_tie]
        factor3 = np.log(np.true_divide(no_records[inds_pos_not_tie] - n_alpha[inds_pos_not_tie], no_records[inds_pos_not_tie] * (1 - alpha[inds_pos_not_tie])))
        score[inds_pos_not_tie] = factor1 + factor2 * factor3
        return score

    @staticmethod
    def get_score_hc_fast(n_alpha: np.ndarray, no_records: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        if False:
            print('Hello World!')
        '\n        HigherCriticism\n        Similar to a traditional wald test statistic: (Observed - expected) / standard deviation.\n        In this case we use the binomial distribution. The observed is N_a. The expected (under null) is N*a\n        and the standard deviation is sqrt(N*a(1-a)).\n\n        :param n_alpha: Number of records less than alpha.\n        :param no_records: Number of records.\n        :param alpha: Alpha threshold.\n        :return: Score.\n        '
        score = np.zeros(alpha.shape[0])
        inds = n_alpha > no_records * alpha
        factor1 = n_alpha[inds] - no_records[inds] * alpha[inds]
        factor2 = np.sqrt(no_records[inds] * alpha[inds] * (1.0 - alpha[inds]))
        score[inds] = np.true_divide(factor1, factor2)
        return score

    @staticmethod
    def get_score_ks_fast(n_alpha: np.ndarray, no_records: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        '\n        KolmarovSmirnov\n\n        :param n_alpha: Number of records less than alpha.\n        :param no_records: Number of records.\n        :param alpha: Alpha threshold.\n        :return: Score.\n        '
        score = np.zeros(alpha.shape[0])
        inds = n_alpha > no_records * alpha
        score[inds] = np.true_divide(n_alpha[inds] - no_records[inds] * alpha[inds], np.sqrt(no_records[inds]))
        return score