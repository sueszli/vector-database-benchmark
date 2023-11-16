"""
SIF module
"""
import numpy as np
from .base import Scoring

class SIF(Scoring):
    """
    Smooth Inverse Frequency (SIF) scoring.
    """

    def __init__(self, config=None):
        if False:
            i = 10
            return i + 15
        super().__init__(config)
        self.a = self.config.get('a', 0.001)

    def computefreq(self, tokens):
        if False:
            print('Hello World!')
        return {token: self.wordfreq[token] for token in tokens}

    def score(self, freq, idf, length):
        if False:
            i = 10
            return i + 15
        if isinstance(freq, np.ndarray) and freq.shape != idf.shape:
            freq.fill(freq.sum())
        return self.a / (self.a + freq / self.tokens)