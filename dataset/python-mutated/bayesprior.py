try:
    import pymc
    pymc_installed = 1
except:
    print('pymc not imported')
    pymc_installed = 0
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats, integrate
from scipy.stats import rv_continuous
from scipy.special import gammaln, gammaincinv, gammainc
from numpy import log, exp

class igamma_gen(rv_continuous):

    def _pdf(self, x, a, b):
        if False:
            while True:
                i = 10
        return exp(self._logpdf(x, a, b))

    def _logpdf(self, x, a, b):
        if False:
            return 10
        return a * log(b) - gammaln(a) - (a + 1) * log(x) - b / x

    def _cdf(self, x, a, b):
        if False:
            while True:
                i = 10
        return 1.0 - gammainc(a, b / x)

    def _ppf(self, q, a, b):
        if False:
            while True:
                i = 10
        return b / gammaincinv(a, 1 - q)

    def _munp(self, n, a, b):
        if False:
            while True:
                i = 10
        args = (a, b)
        super(igamma_gen, self)._munp(self, n, *args)

    def _entropy(self, *args):
        if False:
            print('Hello World!')

        def integ(x):
            if False:
                while True:
                    i = 10
            val = self._pdf(x, *args)
            return val * log(val)
        entr = -integrate.quad(integ, self.a, self.b)[0]
        if not np.isnan(entr):
            return entr
        else:
            raise ValueError('Problem with integration.  Returned nan.')
igamma = igamma_gen(a=0.0, name='invgamma', longname='An inverted gamma', shapes='a,b', extradoc='\n\nInverted gamma distribution\n\ninvgamma.pdf(x,a,b) = b**a*x**(-a-1)/gamma(a) * exp(-b/x)\nfor x > 0, a > 0, b>0.\n')
palpha = np.random.gamma(400.0, 0.005, size=10000)
print('First moment: %s\nSecond moment: %s' % (palpha.mean(), palpha.std()))
palpha = palpha[0]
prho = np.random.beta(49.5, 49.5, size=100000.0)
print('Beta Distribution')
print('First moment: %s\nSecond moment: %s' % (prho.mean(), prho.std()))
prho = prho[0]
psigma = igamma.rvs(1.0, 4.0 ** 2 / 2, size=100000.0)
print('Inverse Gamma Distribution')
print('First moment: %s\nSecond moment: %s' % (psigma.mean(), psigma.std()))
draws = 400
(mu_, lambda_) = (1.0, 2.0)
y1y2 = np.zeros((draws, 2))
for draw in range(draws):
    theta = np.random.normal(mu_, lambda_ ** 2)
    y1 = theta + np.random.normal()
    y2 = theta + np.random.normal()
    y1y2[draw] = (y1, y2)
lnp1p2_mod1 = stats.norm.pdf(y1, loc=mu_, scale=lambda_ ** 2 + 1) * stats.norm.pdf(y2, mu_, scale=lambda_ ** 2 + 1)
pmu_pairsp1 = np.zeros((draws, 2))
y1y2pairsp1 = np.zeros((draws, 2))
for draw in range(draws):
    theta1 = np.random.uniform(0, 1)
    theta2 = np.random.normal(mu_, lambda_ ** 2)
    y1 = theta2
    pmu_pairsp1[draw] = (theta2, theta1)
    y2 = theta2 + theta1 * y1 + np.random.normal()
    y1y2pairsp1[draw] = (y1, y2)
pmu_pairsp2 = np.zeros((draws, 2))
y1y2pairsp2 = np.zeros((draws, 2))
theta12_2 = []
for draw in range(draws):
    theta1 = np.random.uniform(0, 1)
    theta2 = np.random.normal(mu_ * (1 - theta1), lambda_ ** 2 * (1 - theta1) ** 2)
    theta12_2.append([theta1, theta2])
    mu = theta2 / (1 - theta1)
    y1 = np.random.normal(mu_, lambda_ ** 2)
    y2 = theta2 + theta1 * y1 + np.random.normal()
    pmu_pairsp2[draw] = (mu, theta1)
    y1y2pairsp2[draw] = (y1, y2)
fig = plt.figure()
fsp = fig.add_subplot(221)
fsp.scatter(pmu_pairsp1[:, 0], pmu_pairsp1[:, 1], color='b', facecolor='none')
fsp.set_ylabel('Autocorrelation (Y)')
fsp.set_xlabel('Mean (Y)')
fsp.set_title('Model 2 (P1)')
fsp.axis([-20, 20, 0, 1])
fsp = fig.add_subplot(222)
fsp.scatter(pmu_pairsp2[:, 0], pmu_pairsp2[:, 1], color='b', facecolor='none')
fsp.set_title('Model 2 (P2)')
fsp.set_ylabel('Autocorrelation (Y)')
fsp.set_xlabel('Mean (Y)')
fsp.set_title('Model 2 (P2)')
fsp.axis([-20, 20, 0, 1])
fsp = fig.add_subplot(223)
fsp.scatter(y1y2pairsp1[:, 0], y1y2pairsp1[:, 1], color='b', marker='o', facecolor='none')
fsp.scatter(y1y2[:, 0], y1y2[:, 1], color='g', marker='+')
fsp.set_title('Model 1 vs. Model 2 (P1)')
fsp.set_ylabel('Y(2)')
fsp.set_xlabel('Y(1)')
fsp.axis([-20, 20, -20, 20])
fsp = fig.add_subplot(224)
fsp.scatter(y1y2pairsp2[:, 0], y1y2pairsp2[:, 1], color='b', marker='o')
fsp.scatter(y1y2[:, 0], y1y2[:, 1], color='g', marker='+')
fsp.set_title('Model 1 vs. Model 2 (P2)')
fsp.set_ylabel('Y(2)')
fsp.set_xlabel('Y(1)')
fsp.axis([-20, 20, -20, 20])
palpha = np.random.gamma(400, 0.005)
pi = np.random.beta(49.5, 49.5)
psigma = igamma.rvs(1.0, 4.0, size=1000000.0)
if pymc_installed:
    psigma2 = pymc.rinverse_gamma(1.0, 4.0, size=1000000.0)
else:
    psigma2 = stats.invgamma.rvs(1.0, scale=4.0, size=1000000.0)
nsims = 500
y = np.zeros(nsims)