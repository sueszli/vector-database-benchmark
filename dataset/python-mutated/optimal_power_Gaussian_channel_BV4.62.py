"""
Copyright 2013 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np
import cvxpy as cvx
'\nInput parameters\n  n: number of receivers\n  a_val: Positive bit rate coefficient for each receiver\n  b_val: Positive signal to noise ratio coefficient for each receiver\n  P_tot: Total power available to all channels\n  W_tot: Total bandwidth available to all channels\n'

def optimal_power(n, a_val, b_val, P_tot: float=1.0, W_tot: float=1.0):
    if False:
        i = 10
        return i + 15
    '\nBoyd and Vandenberghe, Convex Optimization, exercise 4.62 page 210\nOptimal power and bandwidth allocation in a Gaussian broadcast channel.\n\nWe consider a communication system in which a central node transmits messages\nto n receivers. Each receiver channel is characterized by its (transmit) power\nlevel Pi ≥ 0 and its bandwidth Wi ≥ 0. The power and bandwidth of a receiver\nchannel determine its bit rate Ri (the rate at which information can be sent)\nvia\n   Ri=αiWi log(1 + βiPi/Wi),\nwhere αi and βi are known positive constants. For Wi=0, we take Ri=0 (which\nis what you get if you take the limit as Wi → 0).  The powers must satisfy a\ntotal power constraint, which has the form\nP1 + · · · + Pn = Ptot,\nwhere Ptot > 0 is a given total power available to allocate among the channels.\nSimilarly, the bandwidths must satisfy\nW1 + · · · +Wn = Wtot,\nwhere Wtot > 0 is the (given) total available bandwidth. The optimization\nvariables in this problem are the powers and bandwidths, i.e.,\nP1, . . . , Pn, W1, . . . ,Wn.\nThe objective is to maximize the total utility, sum(ui(Ri),i=1..n)\nwhere ui: R → R is the utility function associated with the ith receiver.\n  '
    n = len(a_val)
    if n != len(b_val):
        print('alpha and beta vectors must have same length!')
        return ('failed', np.nan, np.nan, np.nan)
    P = cvx.Variable(n)
    W = cvx.Variable(n)
    alpha = cvx.Parameter(n)
    beta = cvx.Parameter(n)
    alpha.value = np.array(a_val)
    beta.value = np.array(b_val)
    R = cvx.kl_div(cvx.multiply(alpha, W), cvx.multiply(alpha, W + cvx.multiply(beta, P))) - cvx.multiply(alpha, cvx.multiply(beta, P))
    objective = cvx.Minimize(cvx.sum(R))
    constraints = [P >= 0.0, W >= 0.0, cvx.sum(P) - P_tot == 0.0, cvx.sum(W) - W_tot == 0.0]
    prob = cvx.Problem(objective, constraints)
    prob.solve()
    return (prob.status, -prob.value, P.value, W.value)
if __name__ == '__main__':
    print(optimal_power.__doc__)
    np.set_printoptions(precision=3)
    n = 5
    a_val = np.arange(10, n + 10) / (1.0 * n)
    b_val = [1.0] * n
    b_val = np.arange(10, n + 10) / (1.0 * n)
    P_tot = 0.5
    W_tot = 1.0
    print('Test problem data:')
    print('n = %d Ptot = %.3f Wtot = %.3f' % (n, P_tot, W_tot))
    print('α =', a_val)
    print('β =', b_val)
    (status, utility, power, bandwidth) = optimal_power(n, a_val, b_val, P_tot, W_tot)
    print('Status =', status)
    print('Optimal utility value = %.4g ' % utility)
    print('Optimal power level:\n', power)
    print('Optimal bandwidth:\n', bandwidth)