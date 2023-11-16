""" Diffusion 2: jump diffusion, stochastic volatility, stochastic time

Created on Tue Dec 08 15:03:49 2009

Author: josef-pktd   following Meucci
License: BSD

contains:

CIRSubordinatedBrownian
Heston
IG
JumpDiffusionKou
JumpDiffusionMerton
NIG
VG

References
----------

Attilio Meucci, Review of Discrete and Continuous Processes in Finance: Theory and Applications
Bloomberg Portfolio Research Paper No. 2009-02-CLASSROOM July 1, 2009
http://papers.ssrn.com/sol3/papers.cfm?abstract_id=1373102




this is currently mostly a translation from matlab of
http://www.mathworks.com/matlabcentral/fileexchange/23554-review-of-discrete-and-continuous-processes-in-finance
license BSD:

Copyright (c) 2008, Attilio Meucci
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the distribution

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.



TODO:

* vectorize where possible
* which processes are exactly simulated by finite differences ?
* include or exclude (now) the initial observation ?
* convert to and merge with diffusion.py (part 1 of diffusions)
* which processes can be easily estimated ?
  loglike or characteristic function ?
* tests ? check for possible index errors (random indices), graphs look ok
* adjust notation, variable names, more consistent, more pythonic
* delete a few unused lines, cleanup
* docstrings


random bug (showed up only once, need fuzz-testing to replicate)
  File "../diffusion2.py", line 375, in <module>
    x = jd.simulate(mu,sigma,lambd,a,D,ts,nrepl)
  File "../diffusion2.py", line 129, in simulate
    jumps_ts[n] = CumS[Events]
IndexError: index out of bounds

CumS is empty array, Events == -1


"""
import numpy as np
import matplotlib.pyplot as plt

class JumpDiffusionMerton:
    """

    Example
    -------
    mu=.00     # deterministic drift
    sig=.20 # Gaussian component
    l=3.45 # Poisson process arrival rate
    a=0 # drift of log-jump
    D=.2 # st.dev of log-jump

    X = JumpDiffusionMerton().simulate(mu,sig,lambd,a,D,ts,nrepl)

    plt.figure()
    plt.plot(X.T)
    plt.title('Merton jump-diffusion')


    """

    def __init__(self):
        if False:
            while True:
                i = 10
        pass

    def simulate(self, m, s, lambd, a, D, ts, nrepl):
        if False:
            i = 10
            return i + 15
        T = ts[-1]
        n_jumps = np.random.poisson(lambd * T, size=(nrepl, 1))
        jumps = []
        nobs = len(ts)
        jumps = np.zeros((nrepl, nobs))
        for j in range(nrepl):
            t = T * np.random.rand(n_jumps[j])
            t = np.sort(t, 0)
            S = a + D * np.random.randn(n_jumps[j], 1)
            CumS = np.cumsum(S)
            jumps_ts = np.zeros(nobs)
            for n in range(nobs):
                Events = np.sum(t <= ts[n]) - 1
                jumps_ts[n] = 0
                if Events > 0:
                    jumps_ts[n] = CumS[Events]
            jumps[j, :] = jumps_ts
        D_Diff = np.zeros((nrepl, nobs))
        for k in range(nobs):
            Dt = ts[k]
            if k > 1:
                Dt = ts[k] - ts[k - 1]
            D_Diff[:, k] = m * Dt + s * np.sqrt(Dt) * np.random.randn(nrepl)
        x = np.hstack((np.zeros((nrepl, 1)), np.cumsum(D_Diff, 1) + jumps))
        return x

class JumpDiffusionKou:

    def __init__(self):
        if False:
            while True:
                i = 10
        pass

    def simulate(self, m, s, lambd, p, e1, e2, ts, nrepl):
        if False:
            print('Hello World!')
        T = ts[-1]
        N = np.random.poisson(lambd * T, size=(nrepl, 1))
        jumps = []
        nobs = len(ts)
        jumps = np.zeros((nrepl, nobs))
        for j in range(nrepl):
            t = T * np.random.rand(N[j])
            t = np.sort(t)
            ww = np.random.binomial(1, p, size=N[j])
            S = ww * np.random.exponential(e1, size=N[j]) - (1 - ww) * np.random.exponential(e2, N[j])
            CumS = np.cumsum(S)
            jumps_ts = np.zeros(nobs)
            for n in range(nobs):
                Events = sum(t <= ts[n]) - 1
                jumps_ts[n] = 0
                if Events:
                    jumps_ts[n] = CumS[Events]
            jumps[j, :] = jumps_ts
        D_Diff = np.zeros((nrepl, nobs))
        for k in range(nobs):
            Dt = ts[k]
            if k > 1:
                Dt = ts[k] - ts[k - 1]
            D_Diff[:, k] = m * Dt + s * np.sqrt(Dt) * np.random.normal(size=nrepl)
        x = np.hstack((np.zeros((nrepl, 1)), np.cumsum(D_Diff, 1) + jumps))
        return x

class VG:
    """variance gamma process
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        pass

    def simulate(self, m, s, kappa, ts, nrepl):
        if False:
            return 10
        T = len(ts)
        dXs = np.zeros((nrepl, T))
        for t in range(T):
            dt = ts[1] - 0
            if t > 1:
                dt = ts[t] - ts[t - 1]
            d_tau = kappa * np.random.gamma(dt / kappa, 1.0, size=nrepl)
            dX = np.random.normal(loc=m * d_tau, scale=1e-06 + s * np.sqrt(d_tau))
            dXs[:, t] = dX
        x = np.cumsum(dXs, 1)
        return x

class IG:
    """inverse-Gaussian ??? used by NIG
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def simulate(self, l, m, nrepl):
        if False:
            for i in range(10):
                print('nop')
        N = np.random.randn(nrepl, 1)
        Y = N ** 2
        X = m + 0.5 * m * m / l * Y - 0.5 * m / l * np.sqrt(4 * m * l * Y + m * m * Y ** 2)
        U = np.random.rand(nrepl, 1)
        ind = U > m / (X + m)
        X[ind] = m * m / X[ind]
        return X.ravel()

class NIG:
    """normal-inverse-Gaussian
    """

    def __init__(self):
        if False:
            print('Hello World!')
        pass

    def simulate(self, th, k, s, ts, nrepl):
        if False:
            for i in range(10):
                print('nop')
        T = len(ts)
        DXs = np.zeros((nrepl, T))
        for t in range(T):
            Dt = ts[1] - 0
            if t > 1:
                Dt = ts[t] - ts[t - 1]
            lfrac = 1 / k * Dt ** 2
            m = Dt
            DS = IG().simulate(lfrac, m, nrepl)
            N = np.random.randn(nrepl)
            DX = s * N * np.sqrt(DS) + th * DS
            DXs[:, t] = DX
        x = np.cumsum(DXs, 1)
        return x

class Heston:
    """Heston Stochastic Volatility
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        pass

    def simulate(self, m, kappa, eta, lambd, r, ts, nrepl, tratio=1.0):
        if False:
            return 10
        T = ts[-1]
        nobs = len(ts)
        dt = np.zeros(nobs)
        dt[0] = ts[0] - 0
        dt[1:] = np.diff(ts)
        DXs = np.zeros((nrepl, nobs))
        dB_1 = np.sqrt(dt) * np.random.randn(nrepl, nobs)
        dB_2u = np.sqrt(dt) * np.random.randn(nrepl, nobs)
        dB_2 = r * dB_1 + np.sqrt(1 - r ** 2) * dB_2u
        vt = eta * np.ones(nrepl)
        v = []
        dXs = np.zeros((nrepl, nobs))
        vts = np.zeros((nrepl, nobs))
        for t in range(nobs):
            dv = kappa * (eta - vt) * dt[t] + lambd * np.sqrt(vt) * dB_2[:, t]
            dX = m * dt[t] + np.sqrt(vt * dt[t]) * dB_1[:, t]
            vt = vt + dv
            vts[:, t] = vt
            dXs[:, t] = dX
        x = np.cumsum(dXs, 1)
        return (x, vts)

class CIRSubordinatedBrownian:
    """CIR subordinated Brownian Motion
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def simulate(self, m, kappa, T_dot, lambd, sigma, ts, nrepl):
        if False:
            return 10
        T = ts[-1]
        nobs = len(ts)
        dtarr = np.zeros(nobs)
        dtarr[0] = ts[0] - 0
        dtarr[1:] = np.diff(ts)
        DXs = np.zeros((nrepl, nobs))
        dB = np.sqrt(dtarr) * np.random.randn(nrepl, nobs)
        yt = 1.0
        dXs = np.zeros((nrepl, nobs))
        dtaus = np.zeros((nrepl, nobs))
        y = np.zeros((nrepl, nobs))
        for t in range(nobs):
            dt = dtarr[t]
            dy = kappa * (T_dot - yt) * dt + lambd * np.sqrt(yt) * dB[:, t]
            yt = np.maximum(yt + dy, 1e-10)
            dtau = np.maximum(yt * dt, 1e-06)
            dX = np.random.normal(loc=m * dtau, scale=sigma * np.sqrt(dtau))
            y[:, t] = yt
            dtaus[:, t] = dtau
            dXs[:, t] = dX
        tau = np.cumsum(dtaus, 1)
        x = np.cumsum(dXs, 1)
        return (x, tau, y)

def schout2contank(a, b, d):
    if False:
        return 10
    th = d * b / np.sqrt(a ** 2 - b ** 2)
    k = 1 / (d * np.sqrt(a ** 2 - b ** 2))
    s = np.sqrt(d / np.sqrt(a ** 2 - b ** 2))
    return (th, k, s)
if __name__ == '__main__':
    nobs = 252.0
    ts = np.linspace(1.0 / nobs, 1.0, nobs)
    nrepl = 5
    mu = 0.01
    sigma = 0.02
    lambd = 3.45 * 10
    a = 0
    D = 0.2
    jd = JumpDiffusionMerton()
    x = jd.simulate(mu, sigma, lambd, a, D, ts, nrepl)
    plt.figure()
    plt.plot(x.T)
    plt.title('Merton jump-diffusion')
    sigma = 0.2
    lambd = 3.45
    x = jd.simulate(mu, sigma, lambd, a, D, ts, nrepl)
    plt.figure()
    plt.plot(x.T)
    plt.title('Merton jump-diffusion')
    mu = 0.0
    lambd = 4.25
    p = 0.5
    e1 = 0.2
    e2 = 0.3
    sig = 0.2
    x = JumpDiffusionKou().simulate(mu, sig, lambd, p, e1, e2, ts, nrepl)
    plt.figure()
    plt.plot(x.T)
    plt.title('double exponential (Kou jump diffusion)')
    mu = 0.1
    kappa = 1.0
    sig = 0.5
    x = VG().simulate(mu, sig, kappa, ts, nrepl)
    plt.figure()
    plt.plot(x.T)
    plt.title('variance gamma')
    al = 2.1
    be = 0
    de = 1
    (th, k, s) = schout2contank(al, be, de)
    x = NIG().simulate(th, k, s, ts, nrepl)
    plt.figure()
    plt.plot(x.T)
    plt.title('normal-inverse-Gaussian')
    m = 0.0
    kappa = 0.6
    eta = 0.3 ** 2
    lambd = 0.25
    r = -0.7
    T = 20.0
    nobs = 252.0 * T
    tsh = np.linspace(T / nobs, T, nobs)
    (x, vts) = Heston().simulate(m, kappa, eta, lambd, r, tsh, nrepl, tratio=20.0)
    plt.figure()
    plt.plot(x.T)
    plt.title('Heston Stochastic Volatility')
    plt.figure()
    plt.plot(np.sqrt(vts).T)
    plt.title('Heston Stochastic Volatility - CIR Vol.')
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(x[0])
    plt.title('Heston Stochastic Volatility process')
    plt.subplot(2, 1, 2)
    plt.plot(np.sqrt(vts[0]))
    plt.title('CIR Volatility')
    m = 0.1
    sigma = 0.4
    kappa = 0.6
    T_dot = 1
    lambd = 1
    T = 10.0
    nobs = 252.0 * T
    tsh = np.linspace(T / nobs, T, nobs)
    (x, tau, y) = CIRSubordinatedBrownian().simulate(m, kappa, T_dot, lambd, sigma, tsh, nrepl)
    plt.figure()
    plt.plot(tsh, x.T)
    plt.title('CIRSubordinatedBrownian process')
    plt.figure()
    plt.plot(tsh, y.T)
    plt.title('CIRSubordinatedBrownian - CIR')
    plt.figure()
    plt.plot(tsh, tau.T)
    plt.title('CIRSubordinatedBrownian - stochastic time ')
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(tsh, x[0])
    plt.title('CIRSubordinatedBrownian process')
    plt.subplot(2, 1, 2)
    plt.plot(tsh, y[0], label='CIR')
    plt.plot(tsh, tau[0], label='stoch. time')
    plt.legend(loc='upper left')
    plt.title('CIRSubordinatedBrownian')