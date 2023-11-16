"""
Created on Mon Aug 12 18:47:05 2019

@author: cantaro86
"""
from scipy import sparse
from scipy.sparse.linalg import splu
from time import time
import numpy as np
import scipy as scp
from scipy import signal
from scipy.integrate import quad
import scipy.stats as ss
import scipy.special as scps
import matplotlib.pyplot as plt
from matplotlib import cm
from FMNM.CF import cf_VG
from FMNM.probabilities import Q1, Q2
from functools import partial
from FMNM.FFT import fft_Lewis, IV_from_Lewis

class VG_pricer:
    """
    Closed Formula.
    Monte Carlo.
    Finite-difference PIDE: Explicit-implicit scheme, with Brownian approximation

        0 = dV/dt + (r -(1/2)sig^2 -w) dV/dx + (1/2)sig^2 d^V/dx^2
                 + \\int[ V(x+y) nu(dy) ] -(r+lam)V
    """

    def __init__(self, Option_info, Process_info):
        if False:
            i = 10
            return i + 15
        '\n        Process_info:  of type VG_process.\n        It contains the interest rate r and the VG parameters (sigma, theta, kappa)\n\n        Option_info:  of type Option_param.\n        It contains (S0,K,T) i.e. current price, strike, maturity in years\n        '
        self.r = Process_info.r
        self.sigma = Process_info.sigma
        self.theta = Process_info.theta
        self.kappa = Process_info.kappa
        self.exp_RV = Process_info.exp_RV
        self.w = -np.log(1 - self.theta * self.kappa - self.kappa / 2 * self.sigma ** 2) / self.kappa
        self.S0 = Option_info.S0
        self.K = Option_info.K
        self.T = Option_info.T
        self.price = 0
        self.S_vec = None
        self.price_vec = None
        self.mesh = None
        self.exercise = Option_info.exercise
        self.payoff = Option_info.payoff

    def payoff_f(self, S):
        if False:
            return 10
        if self.payoff == 'call':
            Payoff = np.maximum(S - self.K, 0)
        elif self.payoff == 'put':
            Payoff = np.maximum(self.K - S, 0)
        return Payoff

    def closed_formula(self):
        if False:
            print('Hello World!')
        '\n        VG closed formula.  Put is obtained by put/call parity.\n        '

        def Psy(a, b, g):
            if False:
                return 10
            f = lambda u: ss.norm.cdf(a / np.sqrt(u) + b * np.sqrt(u)) * u ** (g - 1) * np.exp(-u) / scps.gamma(g)
            result = quad(f, 0, np.inf)
            return result[0]
        xi = -self.theta / self.sigma ** 2
        s = self.sigma / np.sqrt(1 + (self.theta / self.sigma) ** 2 * (self.kappa / 2))
        alpha = xi * s
        c1 = self.kappa / 2 * (alpha + s) ** 2
        c2 = self.kappa / 2 * alpha ** 2
        d = 1 / s * (np.log(self.S0 / self.K) + self.r * self.T + self.T / self.kappa * np.log((1 - c1) / (1 - c2)))
        call = self.S0 * Psy(d * np.sqrt((1 - c1) / self.kappa), (alpha + s) * np.sqrt(self.kappa / (1 - c1)), self.T / self.kappa) - self.K * np.exp(-self.r * self.T) * Psy(d * np.sqrt((1 - c2) / self.kappa), alpha * np.sqrt(self.kappa / (1 - c2)), self.T / self.kappa)
        if self.payoff == 'call':
            return call
        elif self.payoff == 'put':
            return call - self.S0 + self.K * np.exp(-self.r * self.T)
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

    def Fourier_inversion(self):
        if False:
            print('Hello World!')
        '\n        Price obtained by inversion of the characteristic function\n        '
        k = np.log(self.K / self.S0)
        cf_VG_b = partial(cf_VG, t=self.T, mu=self.r - self.w, theta=self.theta, sigma=self.sigma, kappa=self.kappa)
        right_lim = 5000
        if self.payoff == 'call':
            call = self.S0 * Q1(k, cf_VG_b, right_lim) - self.K * np.exp(-self.r * self.T) * Q2(k, cf_VG_b, right_lim)
            return call
        elif self.payoff == 'put':
            put = self.K * np.exp(-self.r * self.T) * (1 - Q2(k, cf_VG_b, right_lim)) - self.S0 * (1 - Q1(k, cf_VG_b, right_lim))
            return put
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

    def MC(self, N, Err=False, Time=False):
        if False:
            i = 10
            return i + 15
        '\n        Variance Gamma Monte Carlo\n        Err = return Standard Error if True\n        Time = return execution time if True\n        '
        t_init = time()
        S_T = self.exp_RV(self.S0, self.T, N)
        V = scp.mean(np.exp(-self.r * self.T) * self.payoff_f(S_T), axis=0)
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

    def FFT(self, K):
        if False:
            return 10
        '\n        FFT method. It returns a vector of prices.\n        K is an array of strikes\n        '
        K = np.array(K)
        cf_VG_b = partial(cf_VG, t=self.T, mu=self.r - self.w, theta=self.theta, sigma=self.sigma, kappa=self.kappa)
        if self.payoff == 'call':
            return fft_Lewis(K, self.S0, self.r, self.T, cf_VG_b, interp='cubic')
        elif self.payoff == 'put':
            return fft_Lewis(K, self.S0, self.r, self.T, cf_VG_b, interp='cubic') - self.S0 + K * np.exp(-self.r * self.T)
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

    def IV_Lewis(self):
        if False:
            print('Hello World!')
        'Implied Volatility from the Lewis formula'
        cf_VG_b = partial(cf_VG, t=self.T, mu=self.r - self.w, theta=self.theta, sigma=self.sigma, kappa=self.kappa)
        if self.payoff == 'call':
            return IV_from_Lewis(self.K, self.S0, self.T, self.r, cf_VG_b)
        elif self.payoff == 'put':
            raise NotImplementedError
        else:
            raise ValueError("invalid type. Set 'call' or 'put'")

    def PIDE_price(self, steps, Time=False):
        if False:
            print('Hello World!')
        '\n        steps = tuple with number of space steps and time steps\n        payoff = "call" or "put"\n        exercise = "European" or "American"\n        Time = Boolean. Execution time.\n        '
        t_init = time()
        Nspace = steps[0]
        Ntime = steps[1]
        S_max = 6 * float(self.K)
        S_min = float(self.K) / 6
        x_max = np.log(S_max)
        x_min = np.log(S_min)
        dev_X = np.sqrt(self.sigma ** 2 + self.theta ** 2 * self.kappa)
        dx = (x_max - x_min) / (Nspace - 1)
        extraP = int(np.floor(5 * dev_X / dx))
        x = np.linspace(x_min - extraP * dx, x_max + extraP * dx, Nspace + 2 * extraP)
        (t, dt) = np.linspace(0, self.T, Ntime, retstep=True)
        Payoff = self.payoff_f(np.exp(x))
        offset = np.zeros(Nspace - 2)
        V = np.zeros((Nspace + 2 * extraP, Ntime))
        if self.payoff == 'call':
            V[:, -1] = Payoff
            V[-extraP - 1:, :] = np.exp(x[-extraP - 1:]).reshape(extraP + 1, 1) * np.ones((extraP + 1, Ntime)) - self.K * np.exp(-self.r * t[::-1]) * np.ones((extraP + 1, Ntime))
            V[:extraP + 1, :] = 0
        else:
            V[:, -1] = Payoff
            V[-extraP - 1:, :] = 0
            V[:extraP + 1, :] = self.K * np.exp(-self.r * t[::-1]) * np.ones((extraP + 1, Ntime))
        A = self.theta / self.sigma ** 2
        B = np.sqrt(self.theta ** 2 + 2 * self.sigma ** 2 / self.kappa) / self.sigma ** 2

        def levy_m(y):
            if False:
                i = 10
                return i + 15
            'Levy measure VG'
            return np.exp(A * y - B * np.abs(y)) / (self.kappa * np.abs(y))
        eps = 1.5 * dx
        lam = quad(levy_m, -(extraP + 1.5) * dx, -eps)[0] + quad(levy_m, eps, (extraP + 1.5) * dx)[0]

        def int_w(y):
            if False:
                for i in range(10):
                    print('nop')
            'integrator'
            return (np.exp(y) - 1) * levy_m(y)
        int_s = lambda y: np.abs(y) * np.exp(A * y - B * np.abs(y)) / self.kappa
        w = quad(int_w, -(extraP + 1.5) * dx, -eps)[0] + quad(int_w, eps, (extraP + 1.5) * dx)[0]
        sig2 = quad(int_s, -eps, eps)[0]
        dxx = dx * dx
        a = dt / 2 * ((self.r - w - 0.5 * sig2) / dx - sig2 / dxx)
        b = 1 + dt * (sig2 / dxx + self.r + lam)
        c = -(dt / 2) * ((self.r - w - 0.5 * sig2) / dx + sig2 / dxx)
        D = sparse.diags([a, b, c], [-1, 0, 1], shape=(Nspace - 2, Nspace - 2)).tocsc()
        DD = splu(D)
        nu = np.zeros(2 * extraP + 3)
        x_med = extraP + 1
        x_nu = np.linspace(-(extraP + 1 + 0.5) * dx, (extraP + 1 + 0.5) * dx, 2 * (extraP + 2))
        for i in range(len(nu)):
            if i == x_med or i == x_med - 1 or i == x_med + 1:
                continue
            nu[i] = quad(levy_m, x_nu[i], x_nu[i + 1])[0]
        if self.exercise == 'European':
            for i in range(Ntime - 2, -1, -1):
                offset[0] = a * V[extraP, i]
                offset[-1] = c * V[-1 - extraP, i]
                V_jump = V[extraP + 1:-extraP - 1, i + 1] + dt * signal.convolve(V[:, i + 1], nu[::-1], mode='valid', method='auto')
                V[extraP + 1:-extraP - 1, i] = DD.solve(V_jump - offset)
        elif self.exercise == 'American':
            for i in range(Ntime - 2, -1, -1):
                offset[0] = a * V[extraP, i]
                offset[-1] = c * V[-1 - extraP, i]
                V_jump = V[extraP + 1:-extraP - 1, i + 1] + dt * signal.convolve(V[:, i + 1], nu[::-1], mode='valid', method='auto')
                V[extraP + 1:-extraP - 1, i] = np.maximum(DD.solve(V_jump - offset), Payoff[extraP + 1:-extraP - 1])
        X0 = np.log(self.S0)
        self.S_vec = np.exp(x[extraP + 1:-extraP - 1])
        self.price = np.interp(X0, x, V[:, 0])
        self.price_vec = V[extraP + 1:-extraP - 1, 0]
        self.mesh = V[extraP + 1:-extraP - 1, :]
        if Time is True:
            elapsed = time() - t_init
            return (self.price, elapsed)
        else:
            return self.price

    def plot(self, axis=None):
        if False:
            i = 10
            return i + 15
        if type(self.S_vec) != np.ndarray or type(self.price_vec) != np.ndarray:
            self.PIDE_price((5000, 4000))
        plt.plot(self.S_vec, self.payoff_f(self.S_vec), color='blue', label='Payoff')
        plt.plot(self.S_vec, self.price_vec, color='red', label='VG curve')
        if type(axis) == list:
            plt.axis(axis)
        plt.xlabel('S')
        plt.ylabel('price')
        plt.title('VG price')
        plt.legend(loc='upper left')
        plt.show()

    def mesh_plt(self):
        if False:
            print('Hello World!')
        if type(self.S_vec) != np.ndarray or type(self.mesh) != np.ndarray:
            self.PDE_price((7000, 5000))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        (X, Y) = np.meshgrid(np.linspace(0, self.T, self.mesh.shape[1]), self.S_vec)
        ax.plot_surface(Y, X, self.mesh, cmap=cm.ocean)
        ax.set_title('VG price surface')
        ax.set_xlabel('S')
        ax.set_ylabel('t')
        ax.set_zlabel('V')
        ax.view_init(30, -100)
        plt.show()

    def closed_formula_wrong(self):
        if False:
            print('Hello World!')
        '\n        VG closed formula. This implementation seems correct, BUT IT DOES NOT WORK!!\n        Here I use the closed formula of Carr,Madan,Chang 1998.\n        With scps.kv, a modified Bessel function of second kind.\n        You can try to run it, but the output is slightly different from expected.\n        '

        def Phi(alpha, beta, gamm, x, y):
            if False:
                for i in range(10):
                    print('nop')
            f = lambda u: u ** (alpha - 1) * (1 - u) ** (gamm - alpha - 1) * (1 - u * x) ** (-beta) * np.exp(u * y)
            result = quad(f, 1e-08, 0.99999999)
            return scps.gamma(gamm) / (scps.gamma(alpha) * scps.gamma(gamm - alpha)) * result[0]

        def Psy(a, b, g):
            if False:
                i = 10
                return i + 15
            c = np.abs(a) * np.sqrt(2 + b ** 2)
            u = b / np.sqrt(2 + b ** 2)
            value = c ** (g + 0.5) * np.exp(np.sign(a) * c) * (1 + u) ** g / (np.sqrt(2 * np.pi) * g * scps.gamma(g)) * scps.kv(g + 0.5, c) * Phi(g, 1 - g, 1 + g, (1 + u) / 2, -np.sign(a) * c * (1 + u)) - np.sign(a) * (c ** (g + 0.5) * np.exp(np.sign(a) * c) * (1 + u) ** (1 + g)) / (np.sqrt(2 * np.pi) * (g + 1) * scps.gamma(g)) * scps.kv(g - 0.5, c) * Phi(g + 1, 1 - g, 2 + g, (1 + u) / 2, -np.sign(a) * c * (1 + u)) + np.sign(a) * (c ** (g + 0.5) * np.exp(np.sign(a) * c) * (1 + u) ** (1 + g)) / (np.sqrt(2 * np.pi) * (g + 1) * scps.gamma(g)) * scps.kv(g - 0.5, c) * Phi(g, 1 - g, 1 + g, (1 + u) / 2, -np.sign(a) * c * (1 + u))
            return value
        xi = -self.theta / self.sigma ** 2
        s = self.sigma / np.sqrt(1 + (self.theta / self.sigma) ** 2 * (self.kappa / 2))
        alpha = xi * s
        c1 = self.kappa / 2 * (alpha + s) ** 2
        c2 = self.kappa / 2 * alpha ** 2
        d = 1 / s * (np.log(self.S0 / self.K) + self.r * self.T + self.T / self.kappa * np.log((1 - c1) / (1 - c2)))
        call = self.S0 * Psy(d * np.sqrt((1 - c1) / self.kappa), (alpha + s) * np.sqrt(self.kappa / (1 - c1)), self.T / self.kappa) - self.K * np.exp(-self.r * self.T) * Psy(d * np.sqrt((1 - c2) / self.kappa), alpha * np.sqrt(self.kappa / (1 - c2)), self.T / self.kappa)
        return call