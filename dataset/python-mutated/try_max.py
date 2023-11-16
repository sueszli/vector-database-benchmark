"""

adjusted from Denis on pystatsmodels mailing list

there might still be problems with loc and scale,

"""
from scipy import stats
__date__ = '2010-12-29 dec'

class MaxDist(stats.rv_continuous):
    """ max of n of scipy.stats normal expon ...
        Example:
            maxnormal10 = RVmax( scipy.stats.norm, 10 )
            sample = maxnormal10( size=1000 )
            sample.cdf = cdf ^ n,  ppf ^ (1/n)
    """

    def __init__(self, dist, n):
        if False:
            for i in range(10):
                print('nop')
        self.dist = dist
        self.n = n
        extradoc = 'maximumdistribution is the distribution of the ' + 'maximum of n i.i.d. random variable'
        super(MaxDist, self).__init__(name='maxdist', a=dist.a, b=dist.b, longname='A maximumdistribution')

    def _pdf(self, x, *args, **kw):
        if False:
            for i in range(10):
                print('nop')
        return self.n * self.dist.pdf(x, *args, **kw) * self.dist.cdf(x, *args, **kw) ** (self.n - 1)

    def _cdf(self, x, *args, **kw):
        if False:
            print('Hello World!')
        return self.dist.cdf(x, *args, **kw) ** self.n

    def _ppf(self, q, *args, **kw):
        if False:
            return 10
        return self.dist.ppf(q ** (1.0 / self.n), *args, **kw)
maxdistr = MaxDist(stats.norm, 10)
print(maxdistr.rvs(size=10))
print(maxdistr.stats(moments='mvsk'))
"\n>>> print maxdistr.stats(moments = 'mvsk')\n(array(1.5387527308351818), array(0.34434382328492852), array(0.40990510188513779), array(0.33139861783918922))\n>>> rvs = np.random.randn(1000,10)\n>>> stats.describe(rvs.max(1))\n(1000, (-0.028558517753519492, 3.6134958002753685), 1.5560520428553426, 0.34965234046170773, 0.48504309950278557, 0.17691859056779258)\n>>> rvs2 = maxdistr.rvs(size=1000)\n>>> stats.describe(rvs2)\n(1000, (-0.015290995091401905, 3.3227019151170931), 1.5248146840651813, 0.32827518543128631, 0.23998620901199566, -0.080555658370268013)\n>>> rvs2 = maxdistr.rvs(size=10000)\n>>> stats.describe(rvs2)\n(10000, (-0.15855091764294812, 4.1898138060896937), 1.532862047388899, 0.34361316060467512, 0.43128838106936973, 0.41115043864619061)\n\n>>> maxdistr.pdf(1.5)\n0.69513824417156755\n\n#integrating the pdf\n>>> maxdistr.expect()\n1.5387527308351729\n>>> maxdistr.expect(lambda x:1)\n0.99999999999999956\n\n\n"