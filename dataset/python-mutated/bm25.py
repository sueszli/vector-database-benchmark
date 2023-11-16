"""
BM25 module
"""
import numpy as np
from .base import Scoring

class BM25(Scoring):
    """
    Best matching (BM25) scoring.
    """

    def __init__(self, config=None):
        if False:
            while True:
                i = 10
        super().__init__(config)
        self.k1 = self.config.get('k1', 1.2)
        self.b = self.config.get('b', 0.75)

    def computeidf(self, freq):
        if False:
            i = 10
            return i + 15
        return np.log(1 + (self.total - freq + 0.5) / (freq + 0.5))

    def score(self, freq, idf, length):
        if False:
            i = 10
            return i + 15
        k = self.k1 * (1 - self.b + self.b * length / self.avgdl)
        return idf * (freq * (self.k1 + 1)) / (freq + k)