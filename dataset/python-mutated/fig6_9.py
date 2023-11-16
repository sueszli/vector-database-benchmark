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
from multiprocessing import Pool
import cvxopt
import numpy as np
from pylab import plot, show, title, xlabel, ylabel
from scipy import sparse
from cvxpy import Minimize, Parameter, Problem, Variable, norm
n = 400
t = np.array(range(0, n))
exact = 0.5 * np.sin(2 * np.pi * t / n) * np.sin(0.01 * t)
corrupt = exact + 0.05 * np.random.randn(len(exact))
corrupt = cvxopt.matrix(corrupt)
e = np.ones(n).T
ee = np.column_stack((-e, e)).T
D = sparse.spdiags(ee, range(-1, 1), n, n)
D = D.todense()
D = cvxopt.matrix(D)
nopts = 10
lambdas = np.linspace(0, 50, nopts)
lamb = Parameter(nonneg=True)
x = Variable(n)
p = Problem(Minimize(norm(x - corrupt) + norm(D * x) * lamb))

def get_value(g):
    if False:
        return 10
    lamb.value = g
    result = p.solve()
    return [np.linalg.norm(x.value - corrupt), np.linalg.norm(D * x.value)]
pool = Pool(processes=4)
(norms1, norms2) = zip(*pool.map(get_value, lambdas))
plot(norms1, norms2)
xlabel('||x - x_{cor}||_2')
ylabel('||Dx||_2')
title('Optimal trade-off curve')
show()