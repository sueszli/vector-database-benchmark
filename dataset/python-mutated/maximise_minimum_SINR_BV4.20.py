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
'\nInput parameters\n  G: matrix of path gains from transmitters to receivers\n  P_max: Maximum power that can be transmitted\n  P_received: Maximum power that can be received\n  sigma: Noise at each receiver\n  Group: Power supply groups the transmitters belong to\n  Group_max: The maximum power that can be supplied to each group\n  detail: Detailed output information is printed if True\n  epsilon: Level of precision for the bisection algorithm\n'

def maxmin_sinr(G, P_max, P_received, sigma, Group, Group_max, detail=False, epsilon=0.001):
    if False:
        return 10
    '\nBoyd and Vandenberghe, Convex Optimization, exercise 4.20 page 196\nPower assignment in a wireless communication system.\n\nWe consider n transmitters with powers p1,...,pn ≥ 0, transmitting to\nn receivers. These powers are the optimization variables in the problem.\nWe let G ∈ ℝ(n*n) denote the matrix of path gains from the\ntransmitter to the receiver. Signal is defined as G_(i,i)*P_i, and\ninterference is defined as ∑(G_(i,j)*pj). Then signal to interference plus\nnoise ratio is defined as S_i/(I_i+σ). The objective function is then to\nmaximise the minimum SINR for all receivers. Each transmitter must be below\na given threshold P_max.  Furthermore, the transmitters are partitioned\ninto groups, with each group sharing the same power supply.  Therefore there\nis a power constraint for each group of transmitter powers.\nThe receivers have the constraint that they cannot receiver more than\na given amount of power i.e. a saturation threshold.\n  '
    (n, m) = np.shape(G)
    if m != np.size(P_max):
        print('Error: P_max dimensions do not match gain matrix dimensions\n')
        return ('Error: P_max dimensions do not match gain matrix dimensions\n', np.nan, np.nan, np.nan)
    if n != np.size(P_received):
        print('Error: P_received dimensions do not match gain matrix dimensions\n')
        return ('Error: P_received dimensions do not match gain matrix dimensions', np.nan, np.nan, np.nan)
    if n != np.size(sigma):
        print('Error: σ dimensions do not match gain matrix dimensions\n')
        return ('Error: σ dimensions do not match gain matrix dimensions', np.nan, np.nan, np.nan)
    I = np.zeros((n, m))
    S = np.zeros((n, m))
    delta = np.identity(n)
    S = G * delta
    I = G - S
    num_groups = int(np.size(Group, 0))
    if num_groups != np.size(Group_max):
        print('Error: Number of groups from Group matrix does not match dimensions of Group_max\n')
        return ('Error: Number of groups from Group matrix does not match dimensions of Group_max', np.nan, np.nan, np.nan, np.nan)
    Group_norm = Group / np.sum(Group, axis=1).reshape((num_groups, 1))
    p = cvx.Variable(n)
    best = np.zeros(n)
    u = 10000.0
    l = 0
    alpha = cvx.Parameter(rows=1, cols=1)
    constraints = [I * p + sigma <= alpha * S * p, p <= P_max, p >= 0, G * p <= P_received, Group_norm * p <= Group_max]
    obj = cvx.Minimize(alpha)
    alpha.value = u
    prob = cvx.Problem(obj, constraints)
    prob.solve()
    if prob.status != 'optimal':
        print('No optimal solution within bounds\n')
        return ('Error: no optimal solution within bounds', np.nan, np.nan, np.nan)
    alpha.value = l
    prob = cvx.Problem(obj, constraints)
    prob.solve()
    if prob.status == 'optimal':
        print('No optimal solution within bounds\n')
        return ('Error: no optimal solution within bounds', np.nan, np.nan, np.nan)
    maxLoop = int(10000000.0)
    for i in range(1, maxLoop):
        alpha.value = (u + l) / 2.0
        if u - l <= epsilon:
            break
        prob = cvx.Problem(obj, constraints)
        prob.solve()
        if prob.status == 'optimal':
            u = alpha.value
            best = p.value
        else:
            l = alpha.value
        if u - l > epsilon and i == maxLoop - 1:
            print('Solution not converged to order epsilon')
    if detail:
        print('l = ', l)
        print('u = ', u)
        print('α = ', alpha.value)
        print('Optimal power p = \n', best)
        print('Received power G*p = \n', G * best)
    return (l, u, alpha.value, best)
if __name__ == '__main__':
    print(maxmin_sinr.__doc__)
    np.set_printoptions(precision=3)
    G = np.array([[0.6, 0.1, 0.1, 0.1, 0.1], [0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1], [0.1, 0.1, 0.1, 0.6, 0.1], [0.1, 0.1, 0.1, 0.1, 0.6]])
    (n, m) = np.shape(G)
    P_max = np.array([1.0] * n)
    P_received = np.array([4.0, 4.0, 4.0, 4.0, 4.0]) / n
    sigma = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
    Group = np.array([[1.0, 1.0, 0, 0, 0], [0, 0, 1.0, 1.0, 1.0]])
    Group_max = np.array([[1.8], [1.8]])
    print('Test problem data')
    print('G=%s' % G)
    print('P_max=%s' % P_max)
    print('P_received=%s' % P_received)
    print('Grouping=%s' % Group)
    print('Max group output=%s' % Group_max)
    (l, u, alpha, best) = maxmin_sinr(G, P_max, P_received, sigma, Group, Group_max, detail=False)
    print('Max SINR=%.4g' % (1 / alpha))
    print('Power=%s' % best)