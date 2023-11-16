import warnings
import pytest

@pytest.mark.xfail(strict=True)
def test_regression_summary():
    if False:
        return 10
    from statsmodels.regression.tests.test_regression import TestOLS
    import time
    from string import Template
    t = time.localtime()
    desired = Template('     Summary of Regression Results\n=======================================\n| Dependent Variable:                y|\n| Model:                           OLS|\n| Method:                Least Squares|\n| Date:               $XXcurrentXdateXX|\n| Time:                       $XXtimeXXX|\n| # obs:                          16.0|\n| Df residuals:                    9.0|\n| Df model:                        6.0|\n==============================================================================\n|                   coefficient     std. error    t-statistic          prob. |\n------------------------------------------------------------------------------\n| x1                      15.06          84.91         0.1774         0.8631 |\n| x2                   -0.03582        0.03349        -1.0695         0.3127 |\n| x3                     -2.020         0.4884        -4.1364         0.0025 |\n| x4                     -1.033         0.2143        -4.8220         0.0009 |\n| x5                   -0.05110         0.2261        -0.2261         0.8262 |\n| x6                      1829.          455.5         4.0159         0.0030 |\n| const              -3.482e+06      8.904e+05        -3.9108         0.0036 |\n==============================================================================\n|                          Models stats                      Residual stats  |\n------------------------------------------------------------------------------\n| R-squared:                     0.9955   Durbin-Watson:              2.559  |\n| Adjusted R-squared:            0.9925   Omnibus:                   0.7486  |\n| F-statistic:                    330.3   Prob(Omnibus):             0.6878  |\n| Prob (F-statistic):         4.984e-10   JB:                        0.6841  |\n| Log likelihood:                -109.6   Prob(JB):                  0.7103  |\n| AIC criterion:                  233.2   Skew:                      0.4200  |\n| BIC criterion:                  238.6   Kurtosis:                   2.434  |\n------------------------------------------------------------------------------').substitute(XXcurrentXdateXX=str(time.strftime('%a, %d %b %Y', t)), XXtimeXXX=str(time.strftime('%H:%M:%S', t)))
    desired = str(desired)
    aregression = TestOLS()
    TestOLS.setup_class()
    results = aregression.res1
    original_filters = warnings.filters[:]
    warnings.simplefilter('ignore')
    try:
        r_summary = str(results.summary_old())
    finally:
        warnings.filters = original_filters
    actual = r_summary
    import numpy as np
    actual = '\n'.join((line.rstrip() for line in actual.split('\n')))
    np.testing.assert_(actual == desired)