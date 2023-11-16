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
from cvxpy import Minimize, Parameter, Problem, Variable, norm
m = 6
n = 3
A = cvxopt.matrix([-2, 7, 1, -5, -1, 3, -7, 3, -5, -1, 4, -4, 1, 5, 5, 2, -5, -1], (m, n))
b = cvxopt.matrix([-4, 3, 9, 0, -11, 5], (m, 1))
d = cvxopt.matrix([-10, -13, -27, -10, -7, 14], (m, 1))
epsilon = Parameter()
x = Variable(n)
objective = Minimize(norm(A * x + b + epsilon * d, 1))
p = Problem(objective, [])

def get_p(e_value):
    if False:
        return 10
    epsilon.value = e_value
    result = p.solve()
    return result
e_values = np.linspace(-1, 1, 41)
print('Computing p*(epsilon) for -1 <= epsilon <= 1 ...')
pool = Pool(processes=4)
p_values = pool.map(get_p, e_values)
print('Done!')
plot(e_values, p_values)
title('p*($\\epsilon$) vs $\\epsilon$')
xlabel('$\\epsilon$')
ylabel('p*($\\epsilon$)')
show()