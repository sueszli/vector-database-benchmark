"""Asymmetric kernels for R+ and unit interval

References
----------

.. [1] Bouezmarni, Taoufik, and Olivier Scaillet. 2005. “Consistency of
   Asymmetric Kernel Density Estimators and Smoothed Histograms with
   Application to Income Data.” Econometric Theory 21 (2): 390–412.

.. [2] Chen, Song Xi. 1999. “Beta Kernel Estimators for Density Functions.”
   Computational Statistics & Data Analysis 31 (2): 131–45.
   https://doi.org/10.1016/S0167-9473(99)00010-9.

.. [3] Chen, Song Xi. 2000. “Probability Density Function Estimation Using
   Gamma Kernels.”
   Annals of the Institute of Statistical Mathematics 52 (3): 471–80.
   https://doi.org/10.1023/A:1004165218295.

.. [4] Jin, Xiaodong, and Janusz Kawczak. 2003. “Birnbaum-Saunders and
   Lognormal Kernel Estimators for Modelling Durations in High Frequency
   Financial Data.” Annals of Economics and Finance 4: 103–24.

.. [5] Micheaux, Pierre Lafaye de, and Frédéric Ouimet. 2020. “A Study of Seven
   Asymmetric Kernels for the Estimation of Cumulative Distribution Functions,”
   November. https://arxiv.org/abs/2011.14893v1.

.. [6] Mombeni, Habib Allah, B Masouri, and Mohammad Reza Akhoond. 2019.
   “Asymmetric Kernels for Boundary Modification in Distribution Function
   Estimation.” REVSTAT, 1–27.

.. [7] Scaillet, O. 2004. “Density Estimation Using Inverse and Reciprocal
   Inverse Gaussian Kernels.”
   Journal of Nonparametric Statistics 16 (1–2): 217–26.
   https://doi.org/10.1080/10485250310001624819.


Created on Mon Mar  8 11:12:24 2021

Author: Josef Perktold
License: BSD-3

"""
import numpy as np
from scipy import special, stats
doc_params = 'Parameters\n    ----------\n    x : array_like, float\n        Points for which density is evaluated. ``x`` can be scalar or 1-dim.\n    sample : ndarray, 1-d\n        Sample from which kde is computed.\n    bw : float\n        Bandwidth parameter, there is currently no default value for it.\n\n    Returns\n    -------\n    Components for kernel estimation'

def pdf_kernel_asym(x, sample, bw, kernel_type, weights=None, batch_size=10):
    if False:
        for i in range(10):
            print('nop')
    'Density estimate based on asymmetric kernel.\n\n    Parameters\n    ----------\n    x : array_like, float\n        Points for which density is evaluated. ``x`` can be scalar or 1-dim.\n    sample : ndarray, 1-d\n        Sample from which kernel estimate is computed.\n    bw : float\n        Bandwidth parameter, there is currently no default value for it.\n    kernel_type : str or callable\n        Kernel name or kernel function.\n        Currently supported kernel names are "beta", "beta2", "gamma",\n        "gamma2", "bs", "invgamma", "invgauss", "lognorm", "recipinvgauss" and\n        "weibull".\n    weights : None or ndarray\n        If weights is not None, then kernel for sample points are weighted\n        by it. No weights corresponds to uniform weighting of each component\n        with 1 / nobs, where nobs is the size of `sample`.\n    batch_size : float\n        If x is an 1-dim array, then points can be evaluated in vectorized\n        form. To limit the amount of memory, a loop can work in batches.\n        The number of batches is determined so that the intermediate array\n        sizes are limited by\n\n        ``np.size(batch) * len(sample) < batch_size * 1000``.\n\n        Default is to have at most 10000 elements in intermediate arrays.\n\n    Returns\n    -------\n    pdf : float or ndarray\n        Estimate of pdf at points x. ``pdf`` has the same size or shape as x.\n    '
    if callable(kernel_type):
        kfunc = kernel_type
    else:
        kfunc = kernel_dict_pdf[kernel_type]
    batch_size = batch_size * 1000
    if np.size(x) * len(sample) < batch_size:
        if np.size(x) > 1:
            x = np.asarray(x)[:, None]
        pdfi = kfunc(x, sample, bw)
        if weights is None:
            pdf = pdfi.mean(-1)
        else:
            pdf = pdfi @ weights
    else:
        if weights is None:
            weights = np.ones(len(sample)) / len(sample)
        k = batch_size // len(sample)
        n = len(x) // k
        x_split = np.array_split(x, n)
        pdf = np.concatenate([kfunc(xi[:, None], sample, bw) @ weights for xi in x_split])
    return pdf

def cdf_kernel_asym(x, sample, bw, kernel_type, weights=None, batch_size=10):
    if False:
        return 10
    'Estimate of cumulative distribution based on asymmetric kernel.\n\n    Parameters\n    ----------\n    x : array_like, float\n        Points for which density is evaluated. ``x`` can be scalar or 1-dim.\n    sample : ndarray, 1-d\n        Sample from which kernel estimate is computed.\n    bw : float\n        Bandwidth parameter, there is currently no default value for it.\n    kernel_type : str or callable\n        Kernel name or kernel function.\n        Currently supported kernel names are "beta", "beta2", "gamma",\n        "gamma2", "bs", "invgamma", "invgauss", "lognorm", "recipinvgauss" and\n        "weibull".\n    weights : None or ndarray\n        If weights is not None, then kernel for sample points are weighted\n        by it. No weights corresponds to uniform weighting of each component\n        with 1 / nobs, where nobs is the size of `sample`.\n    batch_size : float\n        If x is an 1-dim array, then points can be evaluated in vectorized\n        form. To limit the amount of memory, a loop can work in batches.\n        The number of batches is determined so that the intermediate array\n        sizes are limited by\n\n        ``np.size(batch) * len(sample) < batch_size * 1000``.\n\n        Default is to have at most 10000 elements in intermediate arrays.\n\n    Returns\n    -------\n    cdf : float or ndarray\n        Estimate of cdf at points x. ``cdf`` has the same size or shape as x.\n    '
    if callable(kernel_type):
        kfunc = kernel_type
    else:
        kfunc = kernel_dict_cdf[kernel_type]
    batch_size = batch_size * 1000
    if np.size(x) * len(sample) < batch_size:
        if np.size(x) > 1:
            x = np.asarray(x)[:, None]
        cdfi = kfunc(x, sample, bw)
        if weights is None:
            cdf = cdfi.mean(-1)
        else:
            cdf = cdfi @ weights
    else:
        if weights is None:
            weights = np.ones(len(sample)) / len(sample)
        k = batch_size // len(sample)
        n = len(x) // k
        x_split = np.array_split(x, n)
        cdf = np.concatenate([kfunc(xi[:, None], sample, bw) @ weights for xi in x_split])
    return cdf

def kernel_pdf_beta(x, sample, bw):
    if False:
        i = 10
        return i + 15
    return stats.beta.pdf(sample, x / bw + 1, (1 - x) / bw + 1)
kernel_pdf_beta.__doc__ = '    Beta kernel for density, pdf, estimation.\n\n    {doc_params}\n\n    References\n    ----------\n    .. [1] Bouezmarni, Taoufik, and Olivier Scaillet. 2005. “Consistency of\n       Asymmetric Kernel Density Estimators and Smoothed Histograms with\n       Application to Income Data.” Econometric Theory 21 (2): 390–412.\n\n    .. [2] Chen, Song Xi. 1999. “Beta Kernel Estimators for Density Functions.”\n       Computational Statistics & Data Analysis 31 (2): 131–45.\n       https://doi.org/10.1016/S0167-9473(99)00010-9.\n    '.format(doc_params=doc_params)

def kernel_cdf_beta(x, sample, bw):
    if False:
        print('Hello World!')
    return stats.beta.sf(sample, x / bw + 1, (1 - x) / bw + 1)
kernel_cdf_beta.__doc__ = '    Beta kernel for cumulative distribution, cdf, estimation.\n\n    {doc_params}\n\n    References\n    ----------\n    .. [1] Bouezmarni, Taoufik, and Olivier Scaillet. 2005. “Consistency of\n       Asymmetric Kernel Density Estimators and Smoothed Histograms with\n       Application to Income Data.” Econometric Theory 21 (2): 390–412.\n\n    .. [2] Chen, Song Xi. 1999. “Beta Kernel Estimators for Density Functions.”\n       Computational Statistics & Data Analysis 31 (2): 131–45.\n       https://doi.org/10.1016/S0167-9473(99)00010-9.\n    '.format(doc_params=doc_params)

def kernel_pdf_beta2(x, sample, bw):
    if False:
        print('Hello World!')
    a1 = 2 * bw ** 2 + 2.5
    a2 = 4 * bw ** 4 + 6 * bw ** 2 + 2.25
    if np.size(x) == 1:
        if x < 2 * bw:
            a = a1 - np.sqrt(a2 - x ** 2 - x / bw)
            pdf = stats.beta.pdf(sample, a, (1 - x) / bw)
        elif x > 1 - 2 * bw:
            x_ = 1 - x
            a = a1 - np.sqrt(a2 - x_ ** 2 - x_ / bw)
            pdf = stats.beta.pdf(sample, x / bw, a)
        else:
            pdf = stats.beta.pdf(sample, x / bw, (1 - x) / bw)
    else:
        alpha = x / bw
        beta = (1 - x) / bw
        mask_low = x < 2 * bw
        x_ = x[mask_low]
        alpha[mask_low] = a1 - np.sqrt(a2 - x_ ** 2 - x_ / bw)
        mask_upp = x > 1 - 2 * bw
        x_ = 1 - x[mask_upp]
        beta[mask_upp] = a1 - np.sqrt(a2 - x_ ** 2 - x_ / bw)
        pdf = stats.beta.pdf(sample, alpha, beta)
    return pdf
kernel_pdf_beta2.__doc__ = '    Beta kernel for density, pdf, estimation with boundary corrections.\n\n    {doc_params}\n\n    References\n    ----------\n    .. [1] Bouezmarni, Taoufik, and Olivier Scaillet. 2005. “Consistency of\n       Asymmetric Kernel Density Estimators and Smoothed Histograms with\n       Application to Income Data.” Econometric Theory 21 (2): 390–412.\n\n    .. [2] Chen, Song Xi. 1999. “Beta Kernel Estimators for Density Functions.”\n       Computational Statistics & Data Analysis 31 (2): 131–45.\n       https://doi.org/10.1016/S0167-9473(99)00010-9.\n    '.format(doc_params=doc_params)

def kernel_cdf_beta2(x, sample, bw):
    if False:
        print('Hello World!')
    a1 = 2 * bw ** 2 + 2.5
    a2 = 4 * bw ** 4 + 6 * bw ** 2 + 2.25
    if np.size(x) == 1:
        if x < 2 * bw:
            a = a1 - np.sqrt(a2 - x ** 2 - x / bw)
            pdf = stats.beta.sf(sample, a, (1 - x) / bw)
        elif x > 1 - 2 * bw:
            x_ = 1 - x
            a = a1 - np.sqrt(a2 - x_ ** 2 - x_ / bw)
            pdf = stats.beta.sf(sample, x / bw, a)
        else:
            pdf = stats.beta.sf(sample, x / bw, (1 - x) / bw)
    else:
        alpha = x / bw
        beta = (1 - x) / bw
        mask_low = x < 2 * bw
        x_ = x[mask_low]
        alpha[mask_low] = a1 - np.sqrt(a2 - x_ ** 2 - x_ / bw)
        mask_upp = x > 1 - 2 * bw
        x_ = 1 - x[mask_upp]
        beta[mask_upp] = a1 - np.sqrt(a2 - x_ ** 2 - x_ / bw)
        pdf = stats.beta.sf(sample, alpha, beta)
    return pdf
kernel_cdf_beta2.__doc__ = '    Beta kernel for cdf estimation with boundary correction.\n\n    {doc_params}\n\n    References\n    ----------\n    .. [1] Bouezmarni, Taoufik, and Olivier Scaillet. 2005. “Consistency of\n       Asymmetric Kernel Density Estimators and Smoothed Histograms with\n       Application to Income Data.” Econometric Theory 21 (2): 390–412.\n\n    .. [2] Chen, Song Xi. 1999. “Beta Kernel Estimators for Density Functions.”\n       Computational Statistics & Data Analysis 31 (2): 131–45.\n       https://doi.org/10.1016/S0167-9473(99)00010-9.\n    '.format(doc_params=doc_params)

def kernel_pdf_gamma(x, sample, bw):
    if False:
        print('Hello World!')
    pdfi = stats.gamma.pdf(sample, x / bw + 1, scale=bw)
    return pdfi
kernel_pdf_gamma.__doc__ = '    Gamma kernel for density, pdf, estimation.\n\n    {doc_params}\n\n    References\n    ----------\n    .. [1] Bouezmarni, Taoufik, and Olivier Scaillet. 2005. “Consistency of\n       Asymmetric Kernel Density Estimators and Smoothed Histograms with\n       Application to Income Data.” Econometric Theory 21 (2): 390–412.\n\n    .. [2] Chen, Song Xi. 2000. “Probability Density Function Estimation Using\n       Gamma Krnels.”\n       Annals of the Institute of Statistical Mathematics 52 (3): 471–80.\n       https://doi.org/10.1023/A:1004165218295.\n    '.format(doc_params=doc_params)

def kernel_cdf_gamma(x, sample, bw):
    if False:
        for i in range(10):
            print('nop')
    cdfi = stats.gamma.sf(sample, x / bw + 1, scale=bw)
    return cdfi
kernel_cdf_gamma.__doc__ = '    Gamma kernel for cumulative distribution, cdf, estimation.\n\n    {doc_params}\n\n    References\n    ----------\n    .. [1] Bouezmarni, Taoufik, and Olivier Scaillet. 2005. “Consistency of\n       Asymmetric Kernel Density Estimators and Smoothed Histograms with\n       Application to Income Data.” Econometric Theory 21 (2): 390–412.\n\n    .. [2] Chen, Song Xi. 2000. “Probability Density Function Estimation Using\n       Gamma Krnels.”\n       Annals of the Institute of Statistical Mathematics 52 (3): 471–80.\n       https://doi.org/10.1023/A:1004165218295.\n    '.format(doc_params=doc_params)

def _kernel_pdf_gamma(x, sample, bw):
    if False:
        print('Hello World!')
    'Gamma kernel for pdf, without boundary corrected part.\n\n    drops `+ 1` in shape parameter\n\n    It should be possible to use this if probability in\n    neighborhood of zero boundary is small.\n\n    '
    return stats.gamma.pdf(sample, x / bw, scale=bw)

def _kernel_cdf_gamma(x, sample, bw):
    if False:
        return 10
    'Gamma kernel for cdf, without boundary corrected part.\n\n    drops `+ 1` in shape parameter\n\n    It should be possible to use this if probability in\n    neighborhood of zero boundary is small.\n\n    '
    return stats.gamma.sf(sample, x / bw, scale=bw)

def kernel_pdf_gamma2(x, sample, bw):
    if False:
        print('Hello World!')
    if np.size(x) == 1:
        if x < 2 * bw:
            a = (x / bw) ** 2 + 1
        else:
            a = x / bw
    else:
        a = x / bw
        mask = x < 2 * bw
        a[mask] = a[mask] ** 2 + 1
    pdf = stats.gamma.pdf(sample, a, scale=bw)
    return pdf
kernel_pdf_gamma2.__doc__ = '    Gamma kernel for density, pdf, estimation with boundary correction.\n\n    {doc_params}\n\n    References\n    ----------\n    .. [1] Bouezmarni, Taoufik, and Olivier Scaillet. 2005. “Consistency of\n       Asymmetric Kernel Density Estimators and Smoothed Histograms with\n       Application to Income Data.” Econometric Theory 21 (2): 390–412.\n\n    .. [2] Chen, Song Xi. 2000. “Probability Density Function Estimation Using\n       Gamma Krnels.”\n       Annals of the Institute of Statistical Mathematics 52 (3): 471–80.\n       https://doi.org/10.1023/A:1004165218295.\n    '.format(doc_params=doc_params)

def kernel_cdf_gamma2(x, sample, bw):
    if False:
        while True:
            i = 10
    if np.size(x) == 1:
        if x < 2 * bw:
            a = (x / bw) ** 2 + 1
        else:
            a = x / bw
    else:
        a = x / bw
        mask = x < 2 * bw
        a[mask] = a[mask] ** 2 + 1
    pdf = stats.gamma.sf(sample, a, scale=bw)
    return pdf
kernel_cdf_gamma2.__doc__ = '    Gamma kernel for cdf estimation with boundary correction.\n\n    {doc_params}\n\n    References\n    ----------\n    .. [1] Bouezmarni, Taoufik, and Olivier Scaillet. 2005. “Consistency of\n       Asymmetric Kernel Density Estimators and Smoothed Histograms with\n       Application to Income Data.” Econometric Theory 21 (2): 390–412.\n\n    .. [2] Chen, Song Xi. 2000. “Probability Density Function Estimation Using\n       Gamma Krnels.”\n       Annals of the Institute of Statistical Mathematics 52 (3): 471–80.\n       https://doi.org/10.1023/A:1004165218295.\n    '.format(doc_params=doc_params)

def kernel_pdf_invgamma(x, sample, bw):
    if False:
        while True:
            i = 10
    return stats.invgamma.pdf(sample, 1 / bw + 1, scale=x / bw)
kernel_pdf_invgamma.__doc__ = '    Inverse gamma kernel for density, pdf, estimation.\n\n    Based on cdf kernel by Micheaux and Ouimet (2020)\n\n    {doc_params}\n\n    References\n    ----------\n    .. [1] Micheaux, Pierre Lafaye de, and Frédéric Ouimet. 2020. “A Study of\n       Seven Asymmetric Kernels for the Estimation of Cumulative Distribution\n       Functions,” November. https://arxiv.org/abs/2011.14893v1.\n    '.format(doc_params=doc_params)

def kernel_cdf_invgamma(x, sample, bw):
    if False:
        print('Hello World!')
    return stats.invgamma.sf(sample, 1 / bw + 1, scale=x / bw)
kernel_cdf_invgamma.__doc__ = '    Inverse gamma kernel for cumulative distribution, cdf, estimation.\n\n    {doc_params}\n\n    References\n    ----------\n    .. [1] Micheaux, Pierre Lafaye de, and Frédéric Ouimet. 2020. “A Study of\n       Seven Asymmetric Kernels for the Estimation of Cumulative Distribution\n       Functions,” November. https://arxiv.org/abs/2011.14893v1.\n    '.format(doc_params=doc_params)

def kernel_pdf_invgauss(x, sample, bw):
    if False:
        while True:
            i = 10
    m = x
    lam = 1 / bw
    return stats.invgauss.pdf(sample, m / lam, scale=lam)
kernel_pdf_invgauss.__doc__ = '    Inverse gaussian kernel for density, pdf, estimation.\n\n    {doc_params}\n\n    References\n    ----------\n    .. [1] Scaillet, O. 2004. “Density Estimation Using Inverse and Reciprocal\n       Inverse Gaussian Kernels.”\n       Journal of Nonparametric Statistics 16 (1–2): 217–26.\n       https://doi.org/10.1080/10485250310001624819.\n    '.format(doc_params=doc_params)

def kernel_pdf_invgauss_(x, sample, bw):
    if False:
        while True:
            i = 10
    'Inverse gaussian kernel density, explicit formula.\n\n    Scaillet 2004\n    '
    pdf = 1 / np.sqrt(2 * np.pi * bw * sample ** 3) * np.exp(-1 / (2 * bw * x) * (sample / x - 2 + x / sample))
    return pdf.mean(-1)

def kernel_cdf_invgauss(x, sample, bw):
    if False:
        while True:
            i = 10
    m = x
    lam = 1 / bw
    return stats.invgauss.sf(sample, m / lam, scale=lam)
kernel_cdf_invgauss.__doc__ = '    Inverse gaussian kernel for cumulative distribution, cdf, estimation.\n\n    {doc_params}\n\n    References\n    ----------\n    .. [1] Scaillet, O. 2004. “Density Estimation Using Inverse and Reciprocal\n       Inverse Gaussian Kernels.”\n       Journal of Nonparametric Statistics 16 (1–2): 217–26.\n       https://doi.org/10.1080/10485250310001624819.\n    '.format(doc_params=doc_params)

def kernel_pdf_recipinvgauss(x, sample, bw):
    if False:
        print('Hello World!')
    m = 1 / (x - bw)
    lam = 1 / bw
    return stats.recipinvgauss.pdf(sample, m / lam, scale=1 / lam)
kernel_pdf_recipinvgauss.__doc__ = '    Reciprocal inverse gaussian kernel for density, pdf, estimation.\n\n    {doc_params}\n\n    References\n    ----------\n    .. [1] Scaillet, O. 2004. “Density Estimation Using Inverse and Reciprocal\n       Inverse Gaussian Kernels.”\n       Journal of Nonparametric Statistics 16 (1–2): 217–26.\n       https://doi.org/10.1080/10485250310001624819.\n    '.format(doc_params=doc_params)

def kernel_pdf_recipinvgauss_(x, sample, bw):
    if False:
        while True:
            i = 10
    'Reciprocal inverse gaussian kernel density, explicit formula.\n\n    Scaillet 2004\n    '
    pdf = 1 / np.sqrt(2 * np.pi * bw * sample) * np.exp(-(x - bw) / (2 * bw) * sample / (x - bw) - 2 + (x - bw) / sample)
    return pdf

def kernel_cdf_recipinvgauss(x, sample, bw):
    if False:
        i = 10
        return i + 15
    m = 1 / (x - bw)
    lam = 1 / bw
    return stats.recipinvgauss.sf(sample, m / lam, scale=1 / lam)
kernel_cdf_recipinvgauss.__doc__ = '    Reciprocal inverse gaussian kernel for cdf estimation.\n\n    {doc_params}\n\n    References\n    ----------\n    .. [1] Scaillet, O. 2004. “Density Estimation Using Inverse and Reciprocal\n       Inverse Gaussian Kernels.”\n       Journal of Nonparametric Statistics 16 (1–2): 217–26.\n       https://doi.org/10.1080/10485250310001624819.\n    '.format(doc_params=doc_params)

def kernel_pdf_bs(x, sample, bw):
    if False:
        i = 10
        return i + 15
    return stats.fatiguelife.pdf(sample, bw, scale=x)
kernel_pdf_bs.__doc__ = '    Birnbaum Saunders (normal) kernel for density, pdf, estimation.\n\n    {doc_params}\n\n    References\n    ----------\n    .. [1] Jin, Xiaodong, and Janusz Kawczak. 2003. “Birnbaum-Saunders and\n       Lognormal Kernel Estimators for Modelling Durations in High Frequency\n       Financial Data.” Annals of Economics and Finance 4: 103–24.\n    '.format(doc_params=doc_params)

def kernel_cdf_bs(x, sample, bw):
    if False:
        return 10
    return stats.fatiguelife.sf(sample, bw, scale=x)
kernel_cdf_bs.__doc__ = '    Birnbaum Saunders (normal) kernel for cdf estimation.\n\n    {doc_params}\n\n    References\n    ----------\n    .. [1] Jin, Xiaodong, and Janusz Kawczak. 2003. “Birnbaum-Saunders and\n       Lognormal Kernel Estimators for Modelling Durations in High Frequency\n       Financial Data.” Annals of Economics and Finance 4: 103–24.\n    .. [2] Mombeni, Habib Allah, B Masouri, and Mohammad Reza Akhoond. 2019.\n       “Asymmetric Kernels for Boundary Modification in Distribution Function\n       Estimation.” REVSTAT, 1–27.\n    '.format(doc_params=doc_params)

def kernel_pdf_lognorm(x, sample, bw):
    if False:
        for i in range(10):
            print('nop')
    bw_ = np.sqrt(4 * np.log(1 + bw))
    return stats.lognorm.pdf(sample, bw_, scale=x)
kernel_pdf_lognorm.__doc__ = '    Log-normal kernel for density, pdf, estimation.\n\n    {doc_params}\n\n    Notes\n    -----\n    Warning: parameterization of bandwidth will likely be changed\n\n    References\n    ----------\n    .. [1] Jin, Xiaodong, and Janusz Kawczak. 2003. “Birnbaum-Saunders and\n       Lognormal Kernel Estimators for Modelling Durations in High Frequency\n       Financial Data.” Annals of Economics and Finance 4: 103–24.\n    '.format(doc_params=doc_params)

def kernel_cdf_lognorm(x, sample, bw):
    if False:
        i = 10
        return i + 15
    bw_ = np.sqrt(4 * np.log(1 + bw))
    return stats.lognorm.sf(sample, bw_, scale=x)
kernel_cdf_lognorm.__doc__ = '    Log-normal kernel for cumulative distribution, cdf, estimation.\n\n    {doc_params}\n\n    Notes\n    -----\n    Warning: parameterization of bandwidth will likely be changed\n\n    References\n    ----------\n    .. [1] Jin, Xiaodong, and Janusz Kawczak. 2003. “Birnbaum-Saunders and\n       Lognormal Kernel Estimators for Modelling Durations in High Frequency\n       Financial Data.” Annals of Economics and Finance 4: 103–24.\n    '.format(doc_params=doc_params)

def kernel_pdf_lognorm_(x, sample, bw):
    if False:
        while True:
            i = 10
    'Log-normal kernel for density, pdf, estimation, explicit formula.\n\n    Jin, Kawczak 2003\n    '
    term = 8 * np.log(1 + bw)
    pdf = 1 / np.sqrt(term * np.pi) / sample * np.exp(-(np.log(x) - np.log(sample)) ** 2 / term)
    return pdf.mean(-1)

def kernel_pdf_weibull(x, sample, bw):
    if False:
        i = 10
        return i + 15
    return stats.weibull_min.pdf(sample, 1 / bw, scale=x / special.gamma(1 + bw))
kernel_pdf_weibull.__doc__ = '    Weibull kernel for density, pdf, estimation.\n\n    Based on cdf kernel by Mombeni et al. (2019)\n\n    {doc_params}\n\n    References\n    ----------\n    .. [1] Mombeni, Habib Allah, B Masouri, and Mohammad Reza Akhoond. 2019.\n       “Asymmetric Kernels for Boundary Modification in Distribution Function\n       Estimation.” REVSTAT, 1–27.\n    '.format(doc_params=doc_params)

def kernel_cdf_weibull(x, sample, bw):
    if False:
        while True:
            i = 10
    return stats.weibull_min.sf(sample, 1 / bw, scale=x / special.gamma(1 + bw))
kernel_cdf_weibull.__doc__ = '    Weibull kernel for cumulative distribution, cdf, estimation.\n\n    {doc_params}\n\n    References\n    ----------\n    .. [1] Mombeni, Habib Allah, B Masouri, and Mohammad Reza Akhoond. 2019.\n       “Asymmetric Kernels for Boundary Modification in Distribution Function\n       Estimation.” REVSTAT, 1–27.\n    '.format(doc_params=doc_params)
kernel_dict_cdf = {'beta': kernel_cdf_beta, 'beta2': kernel_cdf_beta2, 'bs': kernel_cdf_bs, 'gamma': kernel_cdf_gamma, 'gamma2': kernel_cdf_gamma2, 'invgamma': kernel_cdf_invgamma, 'invgauss': kernel_cdf_invgauss, 'lognorm': kernel_cdf_lognorm, 'recipinvgauss': kernel_cdf_recipinvgauss, 'weibull': kernel_cdf_weibull}
kernel_dict_pdf = {'beta': kernel_pdf_beta, 'beta2': kernel_pdf_beta2, 'bs': kernel_pdf_bs, 'gamma': kernel_pdf_gamma, 'gamma2': kernel_pdf_gamma2, 'invgamma': kernel_pdf_invgamma, 'invgauss': kernel_pdf_invgauss, 'lognorm': kernel_pdf_lognorm, 'recipinvgauss': kernel_pdf_recipinvgauss, 'weibull': kernel_pdf_weibull}