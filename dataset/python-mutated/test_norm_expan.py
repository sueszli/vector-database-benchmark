"""Unit tests for Gram-Charlier exansion

No reference results, test based on consistency and normal case.

Created on Wed Feb 19 12:39:49 2014

Author: Josef Perktold
"""
import pytest
import numpy as np
from scipy import stats
from numpy.testing import assert_allclose, assert_array_less
from statsmodels.sandbox.distributions.extras import NormExpan_gen

class CheckDistribution:

    @pytest.mark.smoke
    def test_dist1(self):
        if False:
            print('Hello World!')
        self.dist1.rvs(size=10)
        self.dist1.pdf(np.linspace(-4, 4, 11))

    def test_cdf_ppf_roundtrip(self):
        if False:
            print('Hello World!')
        probs = np.linspace(0.001, 0.999, 6)
        ppf = self.dist2.ppf(probs)
        cdf = self.dist2.cdf(ppf)
        assert_allclose(cdf, probs, rtol=1e-06)
        sf = self.dist2.sf(ppf)
        assert_allclose(sf, 1 - probs, rtol=1e-06)

class CheckExpandNorm(CheckDistribution):

    def test_pdf(self):
        if False:
            for i in range(10):
                print('nop')
        scale = getattr(self, 'scale', 1)
        x = np.linspace(-4, 4, 11) * scale
        pdf2 = self.dist2.pdf(x)
        pdf1 = self.dist1.pdf(x)
        atol_pdf = getattr(self, 'atol_pdf', 0)
        assert_allclose(((pdf2 - pdf1) ** 2).mean(), 0, rtol=1e-06, atol=atol_pdf)
        assert_allclose(pdf2, pdf1, rtol=1e-06, atol=atol_pdf)

    def test_mvsk(self):
        if False:
            return 10
        mvsk2 = self.dist2.mvsk
        mvsk1 = self.dist2.stats(moments='mvsk')
        assert_allclose(mvsk2, mvsk1, rtol=1e-06, atol=1e-13)
        assert_allclose(self.dist2.mvsk, self.mvsk, rtol=1e-12)

class TestExpandNormMom(CheckExpandNorm):

    @classmethod
    def setup_class(cls):
        if False:
            print('Hello World!')
        cls.scale = 2
        cls.dist1 = stats.norm(1, 2)
        cls.mvsk = [1.0, 2 ** 2, 0, 0]
        cls.dist2 = NormExpan_gen(cls.mvsk, mode='mvsk')

class TestExpandNormSample:

    @classmethod
    def setup_class(cls):
        if False:
            while True:
                i = 10
        cls.dist1 = dist1 = stats.norm(1, 2)
        np.random.seed(5999)
        cls.rvs = dist1.rvs(size=200)
        cls.dist2 = NormExpan_gen(cls.rvs, mode='sample')
        cls.scale = 2
        cls.atol_pdf = 0.001

    def test_ks(self):
        if False:
            print('Hello World!')
        (stat, pvalue) = stats.kstest(self.rvs, self.dist2.cdf)
        assert_array_less(0.25, pvalue)

    def test_mvsk(self):
        if False:
            print('Hello World!')
        mvsk = stats.describe(self.rvs)[-4:]
        assert_allclose(self.dist2.mvsk, mvsk, rtol=1e-12)