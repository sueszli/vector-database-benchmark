"""
NLL Laplace Scorer
------------------

Laplace distribution negative log-likelihood Scorer.

The anomaly score is the negative log likelihood of the actual series values
under a Laplace distribution estimated from the stochastic prediction.
"""
import numpy as np
from scipy.stats import laplace
from darts.ad.scorers.scorers import NLLScorer

class LaplaceNLLScorer(NLLScorer):

    def __init__(self, window: int=1) -> None:
        if False:
            while True:
                i = 10
        super().__init__(window=window)

    def __str__(self):
        if False:
            while True:
                i = 10
        return 'LaplaceNLLScorer'

    def _score_core_nllikelihood(self, deterministic_values: np.ndarray, probabilistic_estimations: np.ndarray) -> np.ndarray:
        if False:
            while True:
                i = 10
        loc = np.median(probabilistic_estimations, axis=1)
        scale = np.sum(np.abs(probabilistic_estimations.T - loc), axis=0).T / probabilistic_estimations.shape[1]
        return -laplace.logpdf(deterministic_values, loc=loc, scale=scale)