"""
Created on Mon Jun 10 09:56:25 2019

@author: cantaro86
"""
from time import time
import numpy as np
import numpy.matlib
import FMNM.cost_utils as cost

class TC_pricer:
    """
    Solver for the option pricing model of Davis-Panas-Zariphopoulou.
    """

    def __init__(self, Option_info, Process_info, cost_b=0, cost_s=0, gamma=0.001):
        if False:
            return 10
        '\n        Option_info:  of type Option_param. It contains (S0,K,T)\n        i.e. current price, strike, maturity in years\n\n        Process_info:  of type Diffusion_process.\n        It contains (r,mu, sig) i.e.  interest rate, drift coefficient, diffusion coeff\n        cost_b:  (lambda in the paper) BUY cost\n        cost_s: (mu in the paper)  SELL cost\n        gamma: risk avversion coefficient\n        '
        if Option_info.payoff == 'put':
            raise ValueError('Not implemented for Put Options')
        self.r = Process_info.r
        self.mu = Process_info.mu
        self.sig = Process_info.sig
        self.S0 = Option_info.S0
        self.K = Option_info.K
        self.T = Option_info.T
        self.cost_b = cost_b
        self.cost_s = cost_s
        self.gamma = gamma

    def price(self, N=500, TYPE='writer', Time=False):
        if False:
            return 10
        '\n        N =  number of time steps\n        TYPE writer or buyer\n        Time: Boolean\n        '
        t = time()
        np.seterr(all='ignore')
        x0 = np.log(self.S0)
        (T_vec, dt) = np.linspace(0, self.T, N + 1, retstep=True)
        delta = np.exp(-self.r * (self.T - T_vec))
        dx = self.sig * np.sqrt(dt)
        dy = dx
        M = int(np.floor(N / 2))
        y = np.linspace(-M * dy, M * dy, 2 * M + 1)
        N_y = len(y)
        med = np.where(y == 0)[0].item()

        def F(xx, ll, nn):
            if False:
                print('Hello World!')
            return np.exp(self.gamma * (1 + self.cost_b) * np.exp(xx) * ll / delta[nn])

        def G(xx, mm, nn):
            if False:
                for i in range(10):
                    print('nop')
            return np.exp(-self.gamma * (1 - self.cost_s) * np.exp(xx) * mm / delta[nn])
        for portfolio in ['no_opt', TYPE]:
            x = np.array([x0 + (self.mu - 0.5 * self.sig ** 2) * dt * N + (2 * i - N) * dx for i in range(N + 1)])
            if portfolio == 'no_opt':
                Q = np.exp(-self.gamma * cost.no_opt(x, y, self.cost_b, self.cost_s))
            elif portfolio == 'writer':
                Q = np.exp(-self.gamma * cost.writer(x, y, self.cost_b, self.cost_s, self.K))
            elif portfolio == 'buyer':
                Q = np.exp(-self.gamma * cost.buyer(x, y, self.cost_b, self.cost_s, self.K))
            else:
                raise ValueError('TYPE can be only writer or buyer')
            for k in range(N - 1, -1, -1):
                Q_new = (Q[:-1, :] + Q[1:, :]) / 2
                x = np.array([x0 + (self.mu - 0.5 * self.sig ** 2) * dt * k + (2 * i - k) * dx for i in range(k + 1)])
                Buy = np.copy(Q_new)
                Buy[:, :-1] = np.matlib.repmat(F(x, dy, k), N_y - 1, 1).T * Q_new[:, 1:]
                Sell = np.copy(Q_new)
                Sell[:, 1:] = np.matlib.repmat(G(x, dy, k), N_y - 1, 1).T * Q_new[:, :-1]
                Q = np.minimum(np.minimum(Buy, Sell), Q_new)
            if portfolio == 'no_opt':
                Q_no = Q[0, med]
            else:
                Q_yes = Q[0, med]
        if TYPE == 'writer':
            price = delta[0] / self.gamma * np.log(Q_yes / Q_no)
        else:
            price = delta[0] / self.gamma * np.log(Q_no / Q_yes)
        if Time is True:
            elapsed = time() - t
            return (price, elapsed)
        else:
            return price