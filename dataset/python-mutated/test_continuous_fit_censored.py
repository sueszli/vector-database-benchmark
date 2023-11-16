import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import fmin
from scipy.stats import CensoredData, beta, cauchy, chi2, expon, gamma, gumbel_l, gumbel_r, invgauss, invweibull, laplace, logistic, lognorm, nct, ncx2, norm, weibull_max, weibull_min

def optimizer(func, x0, args=(), disp=0):
    if False:
        print('Hello World!')
    return fmin(func, x0, args=args, disp=disp, xtol=1e-12, ftol=1e-12)

def test_beta():
    if False:
        print('Hello World!')
    "\n    Test fitting beta shape parameters to interval-censored data.\n\n    Calculation in R:\n\n    > library(fitdistrplus)\n    > data <- data.frame(left=c(0.10, 0.50, 0.75, 0.80),\n    +                    right=c(0.20, 0.55, 0.90, 0.95))\n    > result = fitdistcens(data, 'beta', control=list(reltol=1e-14))\n\n    > result\n    Fitting of the distribution ' beta ' on censored data by maximum likelihood\n    Parameters:\n           estimate\n    shape1 1.419941\n    shape2 1.027066\n    > result$sd\n       shape1    shape2\n    0.9914177 0.6866565\n    "
    data = CensoredData(interval=[[0.1, 0.2], [0.5, 0.55], [0.75, 0.9], [0.8, 0.95]])
    (a, b, loc, scale) = beta.fit(data, floc=0, fscale=1, optimizer=optimizer)
    assert_allclose(a, 1.419941, rtol=5e-06)
    assert_allclose(b, 1.027066, rtol=5e-06)
    assert loc == 0
    assert scale == 1

def test_cauchy_right_censored():
    if False:
        print('Hello World!')
    "\n    Test fitting the Cauchy distribution to right-censored data.\n\n    Calculation in R, with two values not censored [1, 10] and\n    one right-censored value [30].\n\n    > library(fitdistrplus)\n    > data <- data.frame(left=c(1, 10, 30), right=c(1, 10, NA))\n    > result = fitdistcens(data, 'cauchy', control=list(reltol=1e-14))\n    > result\n    Fitting of the distribution ' cauchy ' on censored data by maximum\n    likelihood\n    Parameters:\n             estimate\n    location 7.100001\n    scale    7.455866\n    "
    data = CensoredData(uncensored=[1, 10], right=[30])
    (loc, scale) = cauchy.fit(data, optimizer=optimizer)
    assert_allclose(loc, 7.10001, rtol=5e-06)
    assert_allclose(scale, 7.455866, rtol=5e-06)

def test_cauchy_mixed():
    if False:
        while True:
            i = 10
    "\n    Test fitting the Cauchy distribution to data with mixed censoring.\n\n    Calculation in R, with:\n    * two values not censored [1, 10],\n    * one left-censored [1],\n    * one right-censored [30], and\n    * one interval-censored [[4, 8]].\n\n    > library(fitdistrplus)\n    > data <- data.frame(left=c(NA, 1, 4, 10, 30), right=c(1, 1, 8, 10, NA))\n    > result = fitdistcens(data, 'cauchy', control=list(reltol=1e-14))\n    > result\n    Fitting of the distribution ' cauchy ' on censored data by maximum\n    likelihood\n    Parameters:\n             estimate\n    location 4.605150\n    scale    5.900852\n    "
    data = CensoredData(uncensored=[1, 10], left=[1], right=[30], interval=[[4, 8]])
    (loc, scale) = cauchy.fit(data, optimizer=optimizer)
    assert_allclose(loc, 4.60515, rtol=5e-06)
    assert_allclose(scale, 5.900852, rtol=5e-06)

def test_chi2_mixed():
    if False:
        print('Hello World!')
    "\n    Test fitting just the shape parameter (df) of chi2 to mixed data.\n\n    Calculation in R, with:\n    * two values not censored [1, 10],\n    * one left-censored [1],\n    * one right-censored [30], and\n    * one interval-censored [[4, 8]].\n\n    > library(fitdistrplus)\n    > data <- data.frame(left=c(NA, 1, 4, 10, 30), right=c(1, 1, 8, 10, NA))\n    > result = fitdistcens(data, 'chisq', control=list(reltol=1e-14))\n    > result\n    Fitting of the distribution ' chisq ' on censored data by maximum\n    likelihood\n    Parameters:\n             estimate\n    df 5.060329\n    "
    data = CensoredData(uncensored=[1, 10], left=[1], right=[30], interval=[[4, 8]])
    (df, loc, scale) = chi2.fit(data, floc=0, fscale=1, optimizer=optimizer)
    assert_allclose(df, 5.060329, rtol=5e-06)
    assert loc == 0
    assert scale == 1

def test_expon_right_censored():
    if False:
        return 10
    "\n    For the exponential distribution with loc=0, the exact solution for\n    fitting n uncensored points x[0]...x[n-1] and m right-censored points\n    x[n]..x[n+m-1] is\n\n        scale = sum(x)/n\n\n    That is, divide the sum of all the values (not censored and\n    right-censored) by the number of uncensored values.  (See, for example,\n    https://en.wikipedia.org/wiki/Censoring_(statistics)#Likelihood.)\n\n    The second derivative of the log-likelihood function is\n\n        n/scale**2 - 2*sum(x)/scale**3\n\n    from which the estimate of the standard error can be computed.\n\n    -----\n\n    Calculation in R, for reference only. The R results are not\n    used in the test.\n\n    > library(fitdistrplus)\n    > dexps <- function(x, scale) {\n    +     return(dexp(x, 1/scale))\n    + }\n    > pexps <- function(q, scale) {\n    +     return(pexp(q, 1/scale))\n    + }\n    > left <- c(1, 2.5, 3, 6, 7.5, 10, 12, 12, 14.5, 15,\n    +                                     16, 16, 20, 20, 21, 22)\n    > right <- c(1, 2.5, 3, 6, 7.5, 10, 12, 12, 14.5, 15,\n    +                                     NA, NA, NA, NA, NA, NA)\n    > result = fitdistcens(data, 'exps', start=list(scale=mean(data$left)),\n    +                      control=list(reltol=1e-14))\n    > result\n    Fitting of the distribution ' exps ' on censored data by maximum likelihood\n    Parameters:\n          estimate\n    scale    19.85\n    > result$sd\n       scale\n    6.277119\n    "
    obs = [1, 2.5, 3, 6, 7.5, 10, 12, 12, 14.5, 15, 16, 16, 20, 20, 21, 22]
    cens = [False] * 10 + [True] * 6
    data = CensoredData.right_censored(obs, cens)
    (loc, scale) = expon.fit(data, floc=0, optimizer=optimizer)
    assert loc == 0
    n = len(data) - data.num_censored()
    total = data._uncensored.sum() + data._right.sum()
    expected = total / n
    assert_allclose(scale, expected, 1e-08)

def test_gamma_right_censored():
    if False:
        return 10
    "\n    Fit gamma shape and scale to data with one right-censored value.\n\n    Calculation in R:\n\n    > library(fitdistrplus)\n    > data <- data.frame(left=c(2.5, 2.9, 3.8, 9.1, 9.3, 12.0, 23.0, 25.0),\n    +                    right=c(2.5, 2.9, 3.8, 9.1, 9.3, 12.0, 23.0, NA))\n    > result = fitdistcens(data, 'gamma', start=list(shape=1, scale=10),\n    +                      control=list(reltol=1e-13))\n    > result\n    Fitting of the distribution ' gamma ' on censored data by maximum\n      likelihood\n    Parameters:\n          estimate\n    shape 1.447623\n    scale 8.360197\n    > result$sd\n        shape     scale\n    0.7053086 5.1016531\n    "
    x = CensoredData.right_censored([2.5, 2.9, 3.8, 9.1, 9.3, 12.0, 23.0, 25.0], [0] * 7 + [1])
    (a, loc, scale) = gamma.fit(x, floc=0, optimizer=optimizer)
    assert_allclose(a, 1.447623, rtol=5e-06)
    assert loc == 0
    assert_allclose(scale, 8.360197, rtol=5e-06)

def test_gumbel():
    if False:
        print('Hello World!')
    "\n    Fit gumbel_l and gumbel_r to censored data.\n\n    This R calculation should match gumbel_r.\n\n    > library(evd)\n    > library(fitdistrplus)\n    > data = data.frame(left=c(0, 2, 3, 9, 10, 10),\n    +                   right=c(1, 2, 3, 9, NA, NA))\n    > result = fitdistcens(data, 'gumbel',\n    +                      control=list(reltol=1e-14),\n    +                      start=list(loc=4, scale=5))\n    > result\n    Fitting of the distribution ' gumbel ' on censored data by maximum\n    likelihood\n    Parameters:\n          estimate\n    loc   4.487853\n    scale 4.843640\n    "
    uncensored = np.array([2, 3, 9])
    right = np.array([10, 10])
    interval = np.array([[0, 1]])
    data = CensoredData(uncensored, right=right, interval=interval)
    (loc, scale) = gumbel_r.fit(data, optimizer=optimizer)
    assert_allclose(loc, 4.487853, rtol=5e-06)
    assert_allclose(scale, 4.84364, rtol=5e-06)
    data2 = CensoredData(-uncensored, left=-right, interval=-interval[:, ::-1])
    (loc2, scale2) = gumbel_l.fit(data2, optimizer=optimizer)
    assert_allclose(loc2, -4.487853, rtol=5e-06)
    assert_allclose(scale2, 4.84364, rtol=5e-06)

def test_invgauss():
    if False:
        return 10
    "\n    Fit just the shape parameter of invgauss to data with one value\n    left-censored and one value right-censored.\n\n    Calculation in R; using a fixed dispersion parameter amounts to fixing\n    the scale to be 1.\n\n    > library(statmod)\n    > library(fitdistrplus)\n    > left <- c(NA, 0.4813096, 0.5571880, 0.5132463, 0.3801414, 0.5904386,\n    +           0.4822340, 0.3478597, 3, 0.7191797, 1.5810902, 0.4442299)\n    > right <- c(0.15, 0.4813096, 0.5571880, 0.5132463, 0.3801414, 0.5904386,\n    +            0.4822340, 0.3478597, NA, 0.7191797, 1.5810902, 0.4442299)\n    > data <- data.frame(left=left, right=right)\n    > result = fitdistcens(data, 'invgauss', control=list(reltol=1e-12),\n    +                      fix.arg=list(dispersion=1), start=list(mean=3))\n    > result\n    Fitting of the distribution ' invgauss ' on censored data by maximum\n      likelihood\n    Parameters:\n         estimate\n    mean 0.853469\n    Fixed parameters:\n               value\n    dispersion     1\n    > result$sd\n        mean\n    0.247636\n\n    Here's the R calculation with the dispersion as a free parameter to\n    be fit.\n\n    > result = fitdistcens(data, 'invgauss', control=list(reltol=1e-12),\n    +                      start=list(mean=3, dispersion=1))\n    > result\n    Fitting of the distribution ' invgauss ' on censored data by maximum\n    likelihood\n    Parameters:\n                estimate\n    mean       0.8699819\n    dispersion 1.2261362\n\n    The parametrization of the inverse Gaussian distribution in the\n    `statmod` package is not the same as in SciPy (see\n        https://arxiv.org/abs/1603.06687\n    for details).  The translation from R to SciPy is\n\n        scale = 1/dispersion\n        mu    = mean * dispersion\n\n    > 1/result$estimate['dispersion']  # 1/dispersion\n    dispersion\n     0.8155701\n    > result$estimate['mean'] * result$estimate['dispersion']\n        mean\n    1.066716\n\n    Those last two values are the SciPy scale and shape parameters.\n    "
    x = [0.4813096, 0.557188, 0.5132463, 0.3801414, 0.5904386, 0.482234, 0.3478597, 0.7191797, 1.5810902, 0.4442299]
    data = CensoredData(uncensored=x, left=[0.15], right=[3])
    (mu, loc, scale) = invgauss.fit(data, floc=0, fscale=1, optimizer=optimizer)
    assert_allclose(mu, 0.853469, rtol=5e-05)
    assert loc == 0
    assert scale == 1
    (mu, loc, scale) = invgauss.fit(data, floc=0, optimizer=optimizer)
    assert_allclose(mu, 1.066716, rtol=5e-05)
    assert loc == 0
    assert_allclose(scale, 0.8155701, rtol=5e-05)

def test_invweibull():
    if False:
        return 10
    "\n    Fit invweibull to censored data.\n\n    Here is the calculation in R.  The 'frechet' distribution from the evd\n    package matches SciPy's invweibull distribution.  The `loc` parameter\n    is fixed at 0.\n\n    > library(evd)\n    > library(fitdistrplus)\n    > data = data.frame(left=c(0, 2, 3, 9, 10, 10),\n    +                   right=c(1, 2, 3, 9, NA, NA))\n    > result = fitdistcens(data, 'frechet',\n    +                      control=list(reltol=1e-14),\n    +                      start=list(loc=4, scale=5))\n    > result\n    Fitting of the distribution ' frechet ' on censored data by maximum\n    likelihood\n    Parameters:\n           estimate\n    scale 2.7902200\n    shape 0.6379845\n    Fixed parameters:\n        value\n    loc     0\n    "
    data = CensoredData(uncensored=[2, 3, 9], right=[10, 10], interval=[[0, 1]])
    (c, loc, scale) = invweibull.fit(data, floc=0, optimizer=optimizer)
    assert_allclose(c, 0.6379845, rtol=5e-06)
    assert loc == 0
    assert_allclose(scale, 2.79022, rtol=5e-06)

def test_laplace():
    if False:
        while True:
            i = 10
    "\n    Fir the Laplace distribution to left- and right-censored data.\n\n    Calculation in R:\n\n    > library(fitdistrplus)\n    > dlaplace <- function(x, location=0, scale=1) {\n    +     return(0.5*exp(-abs((x - location)/scale))/scale)\n    + }\n    > plaplace <- function(q, location=0, scale=1) {\n    +     z <- (q - location)/scale\n    +     s <- sign(z)\n    +     f <- -s*0.5*exp(-abs(z)) + (s+1)/2\n    +     return(f)\n    + }\n    > left <- c(NA, -41.564, 50.0, 15.7384, 50.0, 10.0452, -2.0684,\n    +           -19.5399, 50.0,   9.0005, 27.1227, 4.3113, -3.7372,\n    +           25.3111, 14.7987,  34.0887,  50.0, 42.8496, 18.5862,\n    +           32.8921, 9.0448, -27.4591, NA, 19.5083, -9.7199)\n    > right <- c(-50.0, -41.564,  NA, 15.7384, NA, 10.0452, -2.0684,\n    +            -19.5399, NA, 9.0005, 27.1227, 4.3113, -3.7372,\n    +            25.3111, 14.7987, 34.0887, NA,  42.8496, 18.5862,\n    +            32.8921, 9.0448, -27.4591, -50.0, 19.5083, -9.7199)\n    > data <- data.frame(left=left, right=right)\n    > result <- fitdistcens(data, 'laplace', start=list(location=10, scale=10),\n    +                       control=list(reltol=1e-13))\n    > result\n    Fitting of the distribution ' laplace ' on censored data by maximum\n      likelihood\n    Parameters:\n             estimate\n    location 14.79870\n    scale    30.93601\n    > result$sd\n         location     scale\n    0.1758864 7.0972125\n    "
    obs = np.array([-50.0, -41.564, 50.0, 15.7384, 50.0, 10.0452, -2.0684, -19.5399, 50.0, 9.0005, 27.1227, 4.3113, -3.7372, 25.3111, 14.7987, 34.0887, 50.0, 42.8496, 18.5862, 32.8921, 9.0448, -27.4591, -50.0, 19.5083, -9.7199])
    x = obs[(obs != -50.0) & (obs != 50)]
    left = obs[obs == -50.0]
    right = obs[obs == 50.0]
    data = CensoredData(uncensored=x, left=left, right=right)
    (loc, scale) = laplace.fit(data, loc=10, scale=10, optimizer=optimizer)
    assert_allclose(loc, 14.7987, rtol=5e-06)
    assert_allclose(scale, 30.93601, rtol=5e-06)

def test_logistic():
    if False:
        print('Hello World!')
    "\n    Fit the logistic distribution to left-censored data.\n\n    Calculation in R:\n    > library(fitdistrplus)\n    > left = c(13.5401, 37.4235, 11.906 , 13.998 ,  NA    ,  0.4023,  NA    ,\n    +          10.9044, 21.0629,  9.6985,  NA    , 12.9016, 39.164 , 34.6396,\n    +          NA    , 20.3665, 16.5889, 18.0952, 45.3818, 35.3306,  8.4949,\n    +          3.4041,  NA    ,  7.2828, 37.1265,  6.5969, 17.6868, 17.4977,\n    +          16.3391, 36.0541)\n    > right = c(13.5401, 37.4235, 11.906 , 13.998 ,  0.    ,  0.4023,  0.    ,\n    +           10.9044, 21.0629,  9.6985,  0.    , 12.9016, 39.164 , 34.6396,\n    +           0.    , 20.3665, 16.5889, 18.0952, 45.3818, 35.3306,  8.4949,\n    +           3.4041,  0.    ,  7.2828, 37.1265,  6.5969, 17.6868, 17.4977,\n    +           16.3391, 36.0541)\n    > data = data.frame(left=left, right=right)\n    > result = fitdistcens(data, 'logis', control=list(reltol=1e-14))\n    > result\n    Fitting of the distribution ' logis ' on censored data by maximum\n      likelihood\n    Parameters:\n              estimate\n    location 14.633459\n    scale     9.232736\n    > result$sd\n    location    scale\n    2.931505 1.546879\n    "
    x = np.array([13.5401, 37.4235, 11.906, 13.998, 0.0, 0.4023, 0.0, 10.9044, 21.0629, 9.6985, 0.0, 12.9016, 39.164, 34.6396, 0.0, 20.3665, 16.5889, 18.0952, 45.3818, 35.3306, 8.4949, 3.4041, 0.0, 7.2828, 37.1265, 6.5969, 17.6868, 17.4977, 16.3391, 36.0541])
    data = CensoredData.left_censored(x, censored=x == 0)
    (loc, scale) = logistic.fit(data, optimizer=optimizer)
    assert_allclose(loc, 14.633459, rtol=5e-07)
    assert_allclose(scale, 9.232736, rtol=5e-06)

def test_lognorm():
    if False:
        while True:
            i = 10
    "\n    Ref: https://math.montana.edu/jobo/st528/documents/relc.pdf\n\n    The data is the locomotive control time to failure example that starts\n    on page 8.  That's the 8th page in the PDF; the page number shown in\n    the text is 270).\n    The document includes SAS output for the data.\n    "
    miles_to_fail = [22.5, 37.5, 46.0, 48.5, 51.5, 53.0, 54.5, 57.5, 66.5, 68.0, 69.5, 76.5, 77.0, 78.5, 80.0, 81.5, 82.0, 83.0, 84.0, 91.5, 93.5, 102.5, 107.0, 108.5, 112.5, 113.5, 116.0, 117.0, 118.5, 119.0, 120.0, 122.5, 123.0, 127.5, 131.0, 132.5, 134.0]
    data = CensoredData.right_censored(miles_to_fail + [135] * 59, [0] * len(miles_to_fail) + [1] * 59)
    (sigma, loc, scale) = lognorm.fit(data, floc=0)
    assert loc == 0
    mu = np.log(scale)
    assert_allclose(mu, 5.1169, rtol=0.0005)
    assert_allclose(sigma, 0.7055, rtol=0.005)

def test_nct():
    if False:
        while True:
            i = 10
    "\n    Test fitting the noncentral t distribution to censored data.\n\n    Calculation in R:\n\n    > library(fitdistrplus)\n    > data <- data.frame(left=c(1, 2, 3, 5, 8, 10, 25, 25),\n    +                    right=c(1, 2, 3, 5, 8, 10, NA, NA))\n    > result = fitdistcens(data, 't', control=list(reltol=1e-14),\n    +                      start=list(df=1, ncp=2))\n    > result\n    Fitting of the distribution ' t ' on censored data by maximum likelihood\n    Parameters:\n         estimate\n    df  0.5432336\n    ncp 2.8893565\n\n    "
    data = CensoredData.right_censored([1, 2, 3, 5, 8, 10, 25, 25], [0, 0, 0, 0, 0, 0, 1, 1])
    with np.errstate(over='ignore'):
        (df, nc, loc, scale) = nct.fit(data, floc=0, fscale=1, optimizer=optimizer)
    assert_allclose(df, 0.5432336, rtol=5e-06)
    assert_allclose(nc, 2.8893565, rtol=5e-06)
    assert loc == 0
    assert scale == 1

def test_ncx2():
    if False:
        for i in range(10):
            print('nop')
    "\n    Test fitting the shape parameters (df, ncp) of ncx2 to mixed data.\n\n    Calculation in R, with\n    * 5 not censored values [2.7, 0.2, 6.5, 0.4, 0.1],\n    * 1 interval-censored value [[0.6, 1.0]], and\n    * 2 right-censored values [8, 8].\n\n    > library(fitdistrplus)\n    > data <- data.frame(left=c(2.7, 0.2, 6.5, 0.4, 0.1, 0.6, 8, 8),\n    +                    right=c(2.7, 0.2, 6.5, 0.4, 0.1, 1.0, NA, NA))\n    > result = fitdistcens(data, 'chisq', control=list(reltol=1e-14),\n    +                      start=list(df=1, ncp=2))\n    > result\n    Fitting of the distribution ' chisq ' on censored data by maximum\n    likelihood\n    Parameters:\n        estimate\n    df  1.052871\n    ncp 2.362934\n    "
    data = CensoredData(uncensored=[2.7, 0.2, 6.5, 0.4, 0.1], right=[8, 8], interval=[[0.6, 1.0]])
    with np.errstate(over='ignore'):
        (df, ncp, loc, scale) = ncx2.fit(data, floc=0, fscale=1, optimizer=optimizer)
    assert_allclose(df, 1.052871, rtol=5e-06)
    assert_allclose(ncp, 2.362934, rtol=5e-06)
    assert loc == 0
    assert scale == 1

def test_norm():
    if False:
        i = 10
        return i + 15
    "\n    Test fitting the normal distribution to interval-censored data.\n\n    Calculation in R:\n\n    > library(fitdistrplus)\n    > data <- data.frame(left=c(0.10, 0.50, 0.75, 0.80),\n    +                    right=c(0.20, 0.55, 0.90, 0.95))\n    > result = fitdistcens(data, 'norm', control=list(reltol=1e-14))\n\n    > result\n    Fitting of the distribution ' norm ' on censored data by maximum likelihood\n    Parameters:\n          estimate\n    mean 0.5919990\n    sd   0.2868042\n    > result$sd\n         mean        sd\n    0.1444432 0.1029451\n    "
    data = CensoredData(interval=[[0.1, 0.2], [0.5, 0.55], [0.75, 0.9], [0.8, 0.95]])
    (loc, scale) = norm.fit(data, optimizer=optimizer)
    assert_allclose(loc, 0.591999, rtol=5e-06)
    assert_allclose(scale, 0.2868042, rtol=5e-06)

def test_weibull_censored1():
    if False:
        while True:
            i = 10
    s = '3,5,6*,8,10*,11*,15,20*,22,23,27*,29,32,35,40,26,28,33*,21,24*'
    (times, cens) = zip(*[(float(t[0]), len(t) == 2) for t in [w.split('*') for w in s.split(',')]])
    data = CensoredData.right_censored(times, cens)
    (c, loc, scale) = weibull_min.fit(data, floc=0)
    assert_allclose(c, 2.149, rtol=0.001)
    assert loc == 0
    assert_allclose(scale, 28.99, rtol=0.001)
    data2 = CensoredData.left_censored(-np.array(times), cens)
    (c2, loc2, scale2) = weibull_max.fit(data2, floc=0)
    assert_allclose(c2, 2.149, rtol=0.001)
    assert loc2 == 0
    assert_allclose(scale2, 28.99, rtol=0.001)

def test_weibull_min_sas1():
    if False:
        for i in range(10):
            print('nop')
    text = '\n           450 0    460 1   1150 0   1150 0   1560 1\n          1600 0   1660 1   1850 1   1850 1   1850 1\n          1850 1   1850 1   2030 1   2030 1   2030 1\n          2070 0   2070 0   2080 0   2200 1   3000 1\n          3000 1   3000 1   3000 1   3100 0   3200 1\n          3450 0   3750 1   3750 1   4150 1   4150 1\n          4150 1   4150 1   4300 1   4300 1   4300 1\n          4300 1   4600 0   4850 1   4850 1   4850 1\n          4850 1   5000 1   5000 1   5000 1   6100 1\n          6100 0   6100 1   6100 1   6300 1   6450 1\n          6450 1   6700 1   7450 1   7800 1   7800 1\n          8100 1   8100 1   8200 1   8500 1   8500 1\n          8500 1   8750 1   8750 0   8750 1   9400 1\n          9900 1  10100 1  10100 1  10100 1  11500 1\n    '
    (life, cens) = np.array([int(w) for w in text.split()]).reshape(-1, 2).T
    life = life / 1000.0
    data = CensoredData.right_censored(life, cens)
    (c, loc, scale) = weibull_min.fit(data, floc=0, optimizer=optimizer)
    assert_allclose(c, 1.0584, rtol=0.0001)
    assert_allclose(scale, 26.2968, rtol=1e-05)
    assert loc == 0

def test_weibull_min_sas2():
    if False:
        i = 10
        return i + 15
    days = np.array([143, 164, 188, 188, 190, 192, 206, 209, 213, 216, 220, 227, 230, 234, 246, 265, 304, 216, 244])
    data = CensoredData.right_censored(days, [0] * (len(days) - 2) + [1] * 2)
    (c, loc, scale) = weibull_min.fit(data, 1, loc=100, scale=100, optimizer=optimizer)
    assert_allclose(c, 2.7112, rtol=0.0005)
    assert_allclose(loc, 122.03, rtol=0.0005)
    assert_allclose(scale, 108.37, rtol=0.0005)