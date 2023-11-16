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
from __future__ import division
import numpy
from cvxpy import Minimize, Parameter, Problem, Variable, sum_squares

def cvx_relax(prob):
    if False:
        return 10
    new_constr = []
    for var in prob.variables():
        if getattr(var, 'boolean', False):
            new_constr += [0 <= var, var <= 1]
    return Problem(prob.objective, prob.constraints + new_constr)

def round_and_fix(prob):
    if False:
        while True:
            i = 10
    prob.solve()
    new_constr = []
    for var in prob.variables():
        if getattr(var, 'boolean', False):
            new_constr += [var == numpy.round(var.value)]
    return Problem(prob.objective, prob.constraints + new_constr)

def branch_and_bound(n, A, B, c):
    if False:
        print('Hello World!')
    from queue import PriorityQueue
    x = Variable(n)
    z = Variable(n)
    L = Parameter(n)
    U = Parameter(n)
    prob = Problem(Minimize(sum_squares(A * x + B * z - c)), [L <= z, z <= U])
    visited = 0
    best_z = None
    f_best = numpy.inf
    nodes = PriorityQueue()
    nodes.put((numpy.inf, 0, numpy.zeros(n), numpy.ones(n), 0))
    while not nodes.empty():
        visited += 1
        (_, _, L_val, U_val, idx) = nodes.get()
        L.value = L_val
        U.value = U_val
        lower_bound = prob.solve()
        z_star = numpy.round(z.value)
        upper_bound = Problem(prob.objective, [z == z_star]).solve()
        f_best = min(f_best, upper_bound)
        if upper_bound == f_best:
            best_z = z_star
        if idx < n and lower_bound < f_best:
            for i in [0, 1]:
                L_val[idx] = U_val[idx] = i
                nodes.put((lower_bound, i, L_val.copy(), U_val.copy(), idx + 1))
    return (f_best, best_z)
numpy.random.seed(1)

def example(n, get_vals: bool=False):
    if False:
        return 10
    print('n = %d #################' % n)
    m = 2 * n
    A = numpy.matrix(numpy.random.randn(m, n))
    B = numpy.matrix(numpy.random.randn(m, n))
    sltn = (numpy.random.randn(n, 1), numpy.random.randint(2, size=(n, 1)))
    noise = numpy.random.normal(size=(m, 1))
    c = A.dot(sltn[0]) + B.dot(sltn[1]) + noise
    x = Variable(n)
    z = Variable(n)
    z.boolean = True
    obj = sum_squares(A * x + B * z - c)
    prob = Problem(Minimize(obj))
    relaxation = cvx_relax(prob)
    print('relaxation', relaxation.solve())
    rel_z = z.value
    rounded = round_and_fix(relaxation)
    rounded.solve()
    print('relax and round', rounded.value)
    (truth, true_z) = branch_and_bound(n, A, B, c)
    print('true optimum', truth)
    if get_vals:
        return (rel_z, z.value, true_z)
    return (relaxation.value, rounded.value, truth)
import matplotlib.pyplot as plt
n = 20
vals = range(1, n + 1)
(relaxed, rounded, truth) = map(numpy.asarray, example(n, True))
plt.figure(figsize=(6, 4))
plt.plot(vals, relaxed, 'ro')
plt.axhline(y=0.5, color='k', ls='dashed')
plt.xlabel('$i$')
plt.ylabel('$z^\\mathrm{rel}_i$')
plt.show()
import matplotlib.pyplot as plt
relaxed = []
rounded = []
truth = []
vals = range(1, 36)
for n in vals:
    results = example(n)
    results = list(map(lambda x: numpy.around(x, 3), results))
    relaxed.append(results[0])
    rounded.append(results[1])
    truth.append(results[2])
plt.figure(figsize=(6, 4))
plt.plot(vals, rounded, vals, truth, vals, relaxed)
plt.xlabel('n')
plt.ylabel('Objective value')
plt.legend(['Relax and round value', 'Global optimum', 'Lower bound'], loc=2)
plt.show()