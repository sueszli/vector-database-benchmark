"""
Created on Thu Jun 13 10:18:39 2019

@author: cantaro86
"""
import numpy as np
import scipy as scp
from scipy.sparse.linalg import spsolve
from scipy import sparse
from scipy.sparse.linalg import splu
import matplotlib.pyplot as plt
from matplotlib import cm
from time import time
import scipy.stats as ss
from FMNM.Solvers import Thomas
from FMNM.cython.solvers import SOR
from FMNM.CF import cf_normal
from FMNM.probabilities import Q1, Q2
from functools import partial
from FMNM.FFT import fft_Lewis, IV_from_Lewis

class BS_pricer:
    """
    Closed Formula.
    Monte Carlo.
    Finite-difference Black-Scholes PDE:
     df/dt + r df/dx + 1/2 sigma^2 d^f/dx^2 -rf = 0
    """

    def __init__(self, Option_info, Process_info):
        if False:
            for i in range(10):
                print('nop')
        '\n        Option_info: of type Option_param. It contains (S0,K,T)\n                i.e. current price, strike, maturity in years\n        Process_info: of type Diffusion_process. It contains (r, mu, sig) i.e.\n                interest rate, drift coefficient, diffusion coefficient\n        '
        self.r = Process_info.r
        self.sig = Process_info.sig
        self.S0 = Option_info.S0
        self.K = Option_info.K
        self.T = Option_info.T
        self.exp_RV = Process_info.exp_RV
        self.price = 0
        self.S_vec = None
        self.price_vec = None
        self.mesh = None
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

    @staticmethod
    def BlackScholes(payoff='call', S0=100.0, K=100.0, T=1.0, r=0.1, sigma=0.2):
        if False:
            i = 10
            return i + 15
        'Black Scholes closed formula:\n        payoff: call or put.\n        S0: float.    initial stock/index level.\n        K: float strike price.\n        T: float maturity (in year fractions).\n        r: float constant risk-free short rate.\n        sigma: volatility factor in diffusion term.'
        d1 = (np.log(S0 / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = (np.log(S0 / K) + (r - sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        if payoff == 'call':
            return S0 * ss.norm.cdf(d1) - K * np.exp(-r * T) * ss.norm.cdf(d2)
        elif payoff == 'put':
            return K * np.exp(-r * T) * ss.norm.cdf(-d2) - S0 * ss.norm.cdf(-d1)
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

    @staticmethod
    def vega(sigma, S0, K, T, r):
        if False:
            print('Hello World!')
        'BS vega: derivative of the price with respect to the volatility'
        d1 = (np.log(S0 / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        return S0 * np.sqrt(T) * ss.norm.pdf(d1)

    def closed_formula(self):
        if False:
            i = 10
            return i + 15
        '\n        Black Scholes closed formula:\n        '
        d1 = (np.log(self.S0 / self.K) + (self.r + self.sig ** 2 / 2) * self.T) / (self.sig * np.sqrt(self.T))
        d2 = (np.log(self.S0 / self.K) + (self.r - self.sig ** 2 / 2) * self.T) / (self.sig * np.sqrt(self.T))
        if self.payoff == 'call':
            return self.S0 * ss.norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * ss.norm.cdf(d2)
        elif self.payoff == 'put':
            return self.K * np.exp(-self.r * self.T) * ss.norm.cdf(-d2) - self.S0 * ss.norm.cdf(-d1)
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

    def Fourier_inversion(self):
        if False:
            i = 10
            return i + 15
        '\n        Price obtained by inversion of the characteristic function\n        '
        k = np.log(self.K / self.S0)
        cf_GBM = partial(cf_normal, mu=(self.r - 0.5 * self.sig ** 2) * self.T, sig=self.sig * np.sqrt(self.T))
        if self.payoff == 'call':
            call = self.S0 * Q1(k, cf_GBM, np.inf) - self.K * np.exp(-self.r * self.T) * Q2(k, cf_GBM, np.inf)
            return call
        elif self.payoff == 'put':
            put = self.K * np.exp(-self.r * self.T) * (1 - Q2(k, cf_GBM, np.inf)) - self.S0 * (1 - Q1(k, cf_GBM, np.inf))
            return put
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

    def FFT(self, K):
        if False:
            for i in range(10):
                print('nop')
        '\n        FFT method. It returns a vector of prices.\n        K is an array of strikes\n        '
        K = np.array(K)
        cf_GBM = partial(cf_normal, mu=(self.r - 0.5 * self.sig ** 2) * self.T, sig=self.sig * np.sqrt(self.T))
        if self.payoff == 'call':
            return fft_Lewis(K, self.S0, self.r, self.T, cf_GBM, interp='cubic')
        elif self.payoff == 'put':
            return fft_Lewis(K, self.S0, self.r, self.T, cf_GBM, interp='cubic') - self.S0 + K * np.exp(-self.r * self.T)
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

    def IV_Lewis(self):
        if False:
            print('Hello World!')
        'Implied Volatility from the Lewis formula'
        cf_GBM = partial(cf_normal, mu=(self.r - 0.5 * self.sig ** 2) * self.T, sig=self.sig * np.sqrt(self.T))
        if self.payoff == 'call':
            return IV_from_Lewis(self.K, self.S0, self.T, self.r, cf_GBM)
        elif self.payoff == 'put':
            raise NotImplementedError
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

    def MC(self, N, Err=False, Time=False):
        if False:
            print('Hello World!')
        '\n        BS Monte Carlo\n        Err = return Standard Error if True\n        Time = return execution time if True\n        '
        t_init = time()
        S_T = self.exp_RV(self.S0, self.T, N)
        PayOff = self.payoff_f(S_T)
        V = scp.mean(np.exp(-self.r * self.T) * PayOff, axis=0)
        if Err is True:
            if Time is True:
                elapsed = time() - t_init
                return (V, ss.sem(np.exp(-self.r * self.T) * self.payoff_f(S_T)), elapsed)
            else:
                return (V, ss.sem(np.exp(-self.r * self.T) * self.payoff_f(S_T)))
        elif Time is True:
            elapsed = time() - t_init
            return (V, elapsed)
        else:
            return V

    def PDE_price(self, steps, Time=False, solver='splu'):
        if False:
            while True:
                i = 10
        '\n        steps = tuple with number of space steps and time steps\n        payoff = "call" or "put"\n        exercise = "European" or "American"\n        Time = Boolean. Execution time.\n        Solver = spsolve or splu or Thomas or SOR\n        '
        t_init = time()
        Nspace = steps[0]
        Ntime = steps[1]
        S_max = 6 * float(self.K)
        S_min = float(self.K) / 6
        x_max = np.log(S_max)
        x_min = np.log(S_min)
        x0 = np.log(self.S0)
        (x, dx) = np.linspace(x_min, x_max, Nspace, retstep=True)
        (t, dt) = np.linspace(0, self.T, Ntime, retstep=True)
        self.S_vec = np.exp(x)
        Payoff = self.payoff_f(self.S_vec)
        V = np.zeros((Nspace, Ntime))
        if self.payoff == 'call':
            V[:, -1] = Payoff
            V[-1, :] = np.exp(x_max) - self.K * np.exp(-self.r * t[::-1])
            V[0, :] = 0
        else:
            V[:, -1] = Payoff
            V[-1, :] = 0
            V[0, :] = Payoff[0] * np.exp(-self.r * t[::-1])
        sig2 = self.sig ** 2
        dxx = dx ** 2
        a = dt / 2 * ((self.r - 0.5 * sig2) / dx - sig2 / dxx)
        b = 1 + dt * (sig2 / dxx + self.r)
        c = -(dt / 2) * ((self.r - 0.5 * sig2) / dx + sig2 / dxx)
        D = sparse.diags([a, b, c], [-1, 0, 1], shape=(Nspace - 2, Nspace - 2)).tocsc()
        offset = np.zeros(Nspace - 2)
        if solver == 'spsolve':
            if self.exercise == 'European':
                for i in range(Ntime - 2, -1, -1):
                    offset[0] = a * V[0, i]
                    offset[-1] = c * V[-1, i]
                    V[1:-1, i] = spsolve(D, V[1:-1, i + 1] - offset)
            elif self.exercise == 'American':
                for i in range(Ntime - 2, -1, -1):
                    offset[0] = a * V[0, i]
                    offset[-1] = c * V[-1, i]
                    V[1:-1, i] = np.maximum(spsolve(D, V[1:-1, i + 1] - offset), Payoff[1:-1])
        elif solver == 'Thomas':
            if self.exercise == 'European':
                for i in range(Ntime - 2, -1, -1):
                    offset[0] = a * V[0, i]
                    offset[-1] = c * V[-1, i]
                    V[1:-1, i] = Thomas(D, V[1:-1, i + 1] - offset)
            elif self.exercise == 'American':
                for i in range(Ntime - 2, -1, -1):
                    offset[0] = a * V[0, i]
                    offset[-1] = c * V[-1, i]
                    V[1:-1, i] = np.maximum(Thomas(D, V[1:-1, i + 1] - offset), Payoff[1:-1])
        elif solver == 'SOR':
            if self.exercise == 'European':
                for i in range(Ntime - 2, -1, -1):
                    offset[0] = a * V[0, i]
                    offset[-1] = c * V[-1, i]
                    V[1:-1, i] = SOR(a, b, c, V[1:-1, i + 1] - offset, w=1.68, eps=1e-10, N_max=600)
            elif self.exercise == 'American':
                for i in range(Ntime - 2, -1, -1):
                    offset[0] = a * V[0, i]
                    offset[-1] = c * V[-1, i]
                    V[1:-1, i] = np.maximum(SOR(a, b, c, V[1:-1, i + 1] - offset, w=1.68, eps=1e-10, N_max=600), Payoff[1:-1])
        elif solver == 'splu':
            DD = splu(D)
            if self.exercise == 'European':
                for i in range(Ntime - 2, -1, -1):
                    offset[0] = a * V[0, i]
                    offset[-1] = c * V[-1, i]
                    V[1:-1, i] = DD.solve(V[1:-1, i + 1] - offset)
            elif self.exercise == 'American':
                for i in range(Ntime - 2, -1, -1):
                    offset[0] = a * V[0, i]
                    offset[-1] = c * V[-1, i]
                    V[1:-1, i] = np.maximum(DD.solve(V[1:-1, i + 1] - offset), Payoff[1:-1])
        else:
            raise ValueError('Solver is splu, spsolve, SOR or Thomas')
        self.price = np.interp(x0, x, V[:, 0])
        self.price_vec = V[:, 0]
        self.mesh = V
        if Time is True:
            elapsed = time() - t_init
            return (self.price, elapsed)
        else:
            return self.price

    def plot(self, axis=None):
        if False:
            for i in range(10):
                print('nop')
        if type(self.S_vec) != np.ndarray or type(self.price_vec) != np.ndarray:
            self.PDE_price((7000, 5000))
        plt.plot(self.S_vec, self.payoff_f(self.S_vec), color='blue', label='Payoff')
        plt.plot(self.S_vec, self.price_vec, color='red', label='BS curve')
        if type(axis) == list:
            plt.axis(axis)
        plt.xlabel('S')
        plt.ylabel('price')
        plt.title(f'{self.exercise} - Black Scholes price')
        plt.legend()
        plt.show()

    def mesh_plt(self):
        if False:
            return 10
        if type(self.S_vec) != np.ndarray or type(self.mesh) != np.ndarray:
            self.PDE_price((7000, 5000))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        (X, Y) = np.meshgrid(np.linspace(0, self.T, self.mesh.shape[1]), self.S_vec)
        ax.plot_surface(Y, X, self.mesh, cmap=cm.ocean)
        ax.set_title(f'{self.exercise} - BS price surface')
        ax.set_xlabel('S')
        ax.set_ylabel('t')
        ax.set_zlabel('V')
        ax.view_init(30, -100)
        plt.show()

    def LSM(self, N=10000, paths=10000, order=2):
        if False:
            print('Hello World!')
        '\n        Longstaff-Schwartz Method for pricing American options\n\n        N = number of time steps\n        paths = number of generated paths\n        order = order of the polynomial for the regression\n        '
        if self.payoff != 'put':
            raise ValueError("invalid type. Set 'call' or 'put'")
        dt = self.T / (N - 1)
        df = np.exp(-self.r * dt)
        X0 = np.zeros((paths, 1))
        increments = ss.norm.rvs(loc=(self.r - self.sig ** 2 / 2) * dt, scale=np.sqrt(dt) * self.sig, size=(paths, N - 1))
        X = np.concatenate((X0, increments), axis=1).cumsum(1)
        S = self.S0 * np.exp(X)
        H = np.maximum(self.K - S, 0)
        V = np.zeros_like(H)
        V[:, -1] = H[:, -1]
        for t in range(N - 2, 0, -1):
            good_paths = H[:, t] > 0
            rg = np.polyfit(S[good_paths, t], V[good_paths, t + 1] * df, 2)
            C = np.polyval(rg, S[good_paths, t])
            exercise = np.zeros(len(good_paths), dtype=bool)
            exercise[good_paths] = H[good_paths, t] > C
            V[exercise, t] = H[exercise, t]
            V[exercise, t + 1:] = 0
            discount_path = V[:, t] == 0
            V[discount_path, t] = V[discount_path, t + 1] * df
        V0 = np.mean(V[:, 1]) * df
        return V0