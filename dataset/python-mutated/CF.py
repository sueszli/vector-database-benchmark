"""
Created on Mon Oct  7 17:57:19 2019

@author: cantaro86
"""
import numpy as np

def cf_normal(u, mu=1, sig=2):
    if False:
        while True:
            i = 10
    '\n    Characteristic function of a Normal random variable\n    '
    return np.exp(1j * u * mu - 0.5 * u ** 2 * sig ** 2)

def cf_gamma(u, a=1, b=2):
    if False:
        i = 10
        return i + 15
    '\n    Characteristic function of a Gamma random variable\n    - shape: a\n    - scale: b\n    '
    return (1 - b * u * 1j) ** (-a)

def cf_poisson(u, lam=1):
    if False:
        i = 10
        return i + 15
    '\n    Characteristic function of a Poisson random variable\n    - rate: lam\n    '
    return np.exp(lam * (np.exp(1j * u) - 1))

def cf_mert(u, t=1, mu=1, sig=2, lam=0.8, muJ=0, sigJ=0.5):
    if False:
        i = 10
        return i + 15
    '\n    Characteristic function of a Merton random variable at time t\n    mu: drift\n    sig: diffusion coefficient\n    lam: jump activity\n    muJ: jump mean size\n    sigJ: jump size standard deviation\n    '
    return np.exp(t * (1j * u * mu - 0.5 * u ** 2 * sig ** 2 + lam * (np.exp(1j * u * muJ - 0.5 * u ** 2 * sigJ ** 2) - 1)))

def cf_VG(u, t=1, mu=0, theta=-0.1, sigma=0.2, kappa=0.1):
    if False:
        i = 10
        return i + 15
    '\n    Characteristic function of a Variance Gamma random variable at time t\n    mu: additional drift\n    theta: Brownian motion drift\n    sigma: Brownian motion diffusion\n    kappa: Gamma process variance\n    '
    return np.exp(t * (1j * mu * u - np.log(1 - 1j * theta * kappa * u + 0.5 * kappa * sigma ** 2 * u ** 2) / kappa))

def cf_NIG(u, t=1, mu=0, theta=-0.1, sigma=0.2, kappa=0.1):
    if False:
        print('Hello World!')
    '\n    Characteristic function of a Normal Inverse Gaussian random variable at time t\n    mu: additional drift\n    theta: Brownian motion drift\n    sigma: Brownian motion diffusion\n    kappa: Inverse Gaussian process variance\n    '
    return np.exp(t * (1j * mu * u + 1 / kappa - np.sqrt(1 - 2j * theta * kappa * u + kappa * sigma ** 2 * u ** 2) / kappa))

def cf_Heston(u, t, v0, mu, kappa, theta, sigma, rho):
    if False:
        for i in range(10):
            print('nop')
    '\n    Heston characteristic function as proposed in the original paper of Heston (1993)\n    '
    xi = kappa - sigma * rho * u * 1j
    d = np.sqrt(xi ** 2 + sigma ** 2 * (u ** 2 + 1j * u))
    g1 = (xi + d) / (xi - d)
    cf = np.exp(1j * u * mu * t + kappa * theta / sigma ** 2 * ((xi + d) * t - 2 * np.log((1 - g1 * np.exp(d * t)) / (1 - g1))) + v0 / sigma ** 2 * (xi + d) * (1 - np.exp(d * t)) / (1 - g1 * np.exp(d * t)))
    return cf

def cf_Heston_good(u, t, v0, mu, kappa, theta, sigma, rho):
    if False:
        while True:
            i = 10
    '\n    Heston characteristic function as proposed by Schoutens (2004)\n    '
    xi = kappa - sigma * rho * u * 1j
    d = np.sqrt(xi ** 2 + sigma ** 2 * (u ** 2 + 1j * u))
    g1 = (xi + d) / (xi - d)
    g2 = 1 / g1
    cf = np.exp(1j * u * mu * t + kappa * theta / sigma ** 2 * ((xi - d) * t - 2 * np.log((1 - g2 * np.exp(-d * t)) / (1 - g2))) + v0 / sigma ** 2 * (xi - d) * (1 - np.exp(-d * t)) / (1 - g2 * np.exp(-d * t)))
    return cf