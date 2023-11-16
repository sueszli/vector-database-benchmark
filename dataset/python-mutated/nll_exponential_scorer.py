"""
NLL Exponential Scorer
----------------------

Exponential distribution negative log-likelihood Scorer.

The anomaly score is the negative log likelihood of the actual series values
under an Exponential distribution estimated from the stochastic prediction.
"""
import numpy as np
from scipy.stats import expon
from darts.ad.scorers.scorers import NLLScorer

class ExponentialNLLScorer(NLLScorer):

    def __init__(self, window: int=1) -> None:
        if False:
            return 10
        super().__init__(window=window)

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'ExponentialNLLScorer'

    def _score_core_nllikelihood(self, deterministic_values: np.ndarray, probabilistic_estimations: np.ndarray) -> np.ndarray:
        if False:
            print('Hello World!')
        mu = np.mean(probabilistic_estimations, axis=1)
        loc = np.min(probabilistic_estimations, axis=1)
        return -expon.logpdf(deterministic_values, scale=mu, loc=loc)