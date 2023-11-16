"""
Created on Mon Oct  7 18:33:39 2019

@author: cantaro86
"""
import numpy as np
from scipy.integrate import quad
from functools import partial
from FMNM.CF import cf_Heston_good
import scipy.special as scps
from math import factorial

def Q1(k, cf, right_lim):
    if False:
        print('Hello World!')
    '\n    P(X<k) - Probability to be in the money under the stock numeraire.\n    cf: characteristic function\n    right_lim: right limit of integration\n    '

    def integrand(u):
        if False:
            for i in range(10):
                print('nop')
        return np.real(np.exp(-u * k * 1j) / (u * 1j) * cf(u - 1j) / cf(-1.0000000000001j))
    return 1 / 2 + 1 / np.pi * quad(integrand, 1e-15, right_lim, limit=2000)[0]

def Q2(k, cf, right_lim):
    if False:
        for i in range(10):
            print('nop')
    '\n    P(X<k) - Probability to be in the money under the money market numeraire\n    cf: characteristic function\n    right_lim: right limit of integration\n    '

    def integrand(u):
        if False:
            while True:
                i = 10
        return np.real(np.exp(-u * k * 1j) / (u * 1j) * cf(u))
    return 1 / 2 + 1 / np.pi * quad(integrand, 1e-15, right_lim, limit=2000)[0]

def Gil_Pelaez_pdf(x, cf, right_lim):
    if False:
        for i in range(10):
            print('nop')
    '\n    Gil Pelaez formula for the inversion of the characteristic function\n    INPUT\n    - x: is a number\n    - right_lim: is the right extreme of integration\n    - cf: is the characteristic function\n    OUTPUT\n    - the value of the density at x.\n    '

    def integrand(u):
        if False:
            for i in range(10):
                print('nop')
        return np.real(np.exp(-u * x * 1j) * cf(u))
    return 1 / np.pi * quad(integrand, 1e-15, right_lim)[0]

def Heston_pdf(i, t, v0, mu, theta, sigma, kappa, rho):
    if False:
        for i in range(10):
            print('nop')
    '\n    Heston density by Fourier inversion.\n    '
    cf_H_b_good = partial(cf_Heston_good, t=t, v0=v0, mu=mu, theta=theta, sigma=sigma, kappa=kappa, rho=rho)
    return Gil_Pelaez_pdf(i, cf_H_b_good, np.inf)

def VG_pdf(x, T, c, theta, sigma, kappa):
    if False:
        print('Hello World!')
    '\n    Variance Gamma density function\n    '
    return 2 * np.exp(theta * (x - c) / sigma ** 2) / (kappa ** (T / kappa) * np.sqrt(2 * np.pi) * sigma * scps.gamma(T / kappa)) * ((x - c) ** 2 / (2 * sigma ** 2 / kappa + theta ** 2)) ** (T / (2 * kappa) - 1 / 4) * scps.kv(T / kappa - 1 / 2, sigma ** (-2) * np.sqrt((x - c) ** 2 * (2 * sigma ** 2 / kappa + theta ** 2)))

def Merton_pdf(x, T, mu, sig, lam, muJ, sigJ):
    if False:
        for i in range(10):
            print('nop')
    '\n    Merton density function\n    '
    tot = 0
    for k in range(20):
        tot += (lam * T) ** k * np.exp(-(x - mu * T - k * muJ) ** 2 / (2 * (T * sig ** 2 + k * sigJ ** 2))) / (factorial(k) * np.sqrt(2 * np.pi * (sig ** 2 * T + k * sigJ ** 2)))
    return np.exp(-lam * T) * tot

def NIG_pdf(x, T, c, theta, sigma, kappa):
    if False:
        for i in range(10):
            print('nop')
    '\n    Merton density function\n    '
    A = theta / sigma ** 2
    B = np.sqrt(theta ** 2 + sigma ** 2 / kappa) / sigma ** 2
    C = T / np.pi * np.exp(T / kappa) * np.sqrt(theta ** 2 / (kappa * sigma ** 2) + 1 / kappa ** 2)
    return C * np.exp(A * (x - c * T)) * scps.kv(1, B * np.sqrt((x - c * T) ** 2 + T ** 2 * sigma ** 2 / kappa)) / np.sqrt((x - c * T) ** 2 + T ** 2 * sigma ** 2 / kappa)