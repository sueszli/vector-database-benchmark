"""
Created on Sun Apr 19 12:13:10 2020

@author: cantaro86
"""
from time import time
import numpy as np
import scipy as scp
import scipy.stats as ss
from FMNM.CF import cf_Heston_good
from FMNM.cython.heston import Heston_paths
from FMNM.probabilities import Q1, Q2
from functools import partial
from FMNM.FFT import fft_Lewis, IV_from_Lewis

class Heston_pricer:
    """
    Class to price the options with the Heston model by:
    - Fourier-inversion.
    - Monte Carlo.
    """

    def __init__(self, Option_info, Process_info):
        if False:
            for i in range(10):
                print('nop')
        '\n        Process_info:  of type VG_process. It contains the interest rate r\n        and the VG parameters (sigma, theta, kappa)\n\n        Option_info:  of type Option_param. It contains (S0,K,T) i.e. current price,\n        strike, maturity in years\n        '
        self.r = Process_info.mu
        self.sigma = Process_info.sigma
        self.theta = Process_info.theta
        self.kappa = Process_info.kappa
        self.rho = Process_info.rho
        self.S0 = Option_info.S0
        self.v0 = Option_info.v0
        self.K = Option_info.K
        self.T = Option_info.T
        self.exercise = Option_info.exercise
        self.payoff = Option_info.payoff

    def payoff_f(self, S):
        if False:
            i = 10
            return i + 15
        if self.payoff == 'call':
            Payoff = np.maximum(S - self.K, 0)
        elif self.payoff == 'put':
            Payoff = np.maximum(self.K - S, 0)
        return Payoff

    def MC(self, N, paths, Err=False, Time=False):
        if False:
            return 10
        '\n        Heston Monte Carlo\n        N = time steps\n        paths = number of simulated paths\n        Err = return Standard Error if True\n        Time = return execution time if True\n        '
        t_init = time()
        (S_T, _) = Heston_paths(N=N, paths=paths, T=self.T, S0=self.S0, v0=self.v0, mu=self.r, rho=self.rho, kappa=self.kappa, theta=self.theta, sigma=self.sigma)
        S_T = S_T.reshape((paths, 1))
        DiscountedPayoff = np.exp(-self.r * self.T) * self.payoff_f(S_T)
        V = scp.mean(DiscountedPayoff, axis=0)
        std_err = ss.sem(DiscountedPayoff)
        if Err is True:
            if Time is True:
                elapsed = time() - t_init
                return (V, std_err, elapsed)
            else:
                return (V, std_err)
        elif Time is True:
            elapsed = time() - t_init
            return (V, elapsed)
        else:
            return V

    def Fourier_inversion(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Price obtained by inversion of the characteristic function\n        '
        k = np.log(self.K / self.S0)
        cf_H_b_good = partial(cf_Heston_good, t=self.T, v0=self.v0, mu=self.r, theta=self.theta, sigma=self.sigma, kappa=self.kappa, rho=self.rho)
        limit_max = 2000
        if self.payoff == 'call':
            call = self.S0 * Q1(k, cf_H_b_good, limit_max) - self.K * np.exp(-self.r * self.T) * Q2(k, cf_H_b_good, limit_max)
            return call
        elif self.payoff == 'put':
            put = self.K * np.exp(-self.r * self.T) * (1 - Q2(k, cf_H_b_good, limit_max)) - self.S0 * (1 - Q1(k, cf_H_b_good, limit_max))
            return put
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

    def FFT(self, K):
        if False:
            for i in range(10):
                print('nop')
        '\n        FFT method. It returns a vector of prices.\n        K is an array of strikes\n        '
        K = np.array(K)
        cf_H_b_good = partial(cf_Heston_good, t=self.T, v0=self.v0, mu=self.r, theta=self.theta, sigma=self.sigma, kappa=self.kappa, rho=self.rho)
        if self.payoff == 'call':
            return fft_Lewis(K, self.S0, self.r, self.T, cf_H_b_good, interp='cubic')
        elif self.payoff == 'put':
            return fft_Lewis(K, self.S0, self.r, self.T, cf_H_b_good, interp='cubic') - self.S0 + K * np.exp(-self.r * self.T)
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

    def IV_Lewis(self):
        if False:
            print('Hello World!')
        'Implied Volatility from the Lewis formula'
        cf_H_b_good = partial(cf_Heston_good, t=self.T, v0=self.v0, mu=self.r, theta=self.theta, sigma=self.sigma, kappa=self.kappa, rho=self.rho)
        if self.payoff == 'call':
            return IV_from_Lewis(self.K, self.S0, self.T, self.r, cf_H_b_good)
        elif self.payoff == 'put':
            raise NotImplementedError
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")