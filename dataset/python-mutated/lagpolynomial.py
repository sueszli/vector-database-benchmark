"""
Created on Fri Oct 22 08:13:38 2010

Author: josef-pktd
License: BSD (3-clause)
"""
import numpy as np
from numpy import polynomial as npp

class LagPolynomial(npp.Polynomial):

    def pad(self, maxlag):
        if False:
            return 10
        return LagPolynomial(np.r_[self.coef, np.zeros(maxlag - len(self.coef))])

    def padflip(self, maxlag):
        if False:
            print('Hello World!')
        return LagPolynomial(np.r_[self.coef, np.zeros(maxlag - len(self.coef))][::-1])

    def flip(self):
        if False:
            while True:
                i = 10
        'reverse polynomial coefficients\n        '
        return LagPolynomial(self.coef[::-1])

    def div(self, other, maxlag=None):
        if False:
            i = 10
            return i + 15
        'padded division, pads numerator with zeros to maxlag\n        '
        if maxlag is None:
            maxlag = max(len(self.coef), len(other.coef)) + 1
        return (self.padflip(maxlag) / other.flip()).flip()

    def filter(self, arr):
        if False:
            while True:
                i = 10
        return (self * arr).coef[:-len(self.coef)]
ar = LagPolynomial([1, -0.8])
arpad = ar.pad(10)
ma = LagPolynomial([1, 0.1])
mapad = ma.pad(10)
unit = LagPolynomial([1])