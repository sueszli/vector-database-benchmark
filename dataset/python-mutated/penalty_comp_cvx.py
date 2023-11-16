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
import cvxopt
import numpy as np
from pylab import hist, plot, show, subplot, title, ylabel
from cvxpy import Minimize, Problem, Variable, norm, pos
(m, n) = (100, 30)
A = cvxopt.normal(m, n)
b = cvxopt.normal(m, 1)
x1 = Variable(n)
objective1 = Minimize(norm(A * x1 - b, 1))
p1 = Problem(objective1, [])
x2 = Variable(n)
objective2 = Minimize(norm(A * x2 - b, 2))
p2 = Problem(objective2, [])

def deadzone(y, z):
    if False:
        print('Hello World!')
    return pos(abs(y) - z)
dz = 0.5
xdz = Variable(n)
objective3 = Minimize(sum(deadzone(A * xdz + b, dz)))
p3 = Problem(objective3, [])
p1.solve()
p2.solve()
p3.solve()
range_max = 2.0
rr = np.linspace(-2, 3, 20)
subplot(3, 1, 1)
(n, bins, patches) = hist(A * x1.value - b, 50, range=[-2, 2])
plot(bins, np.abs(bins) * 35 / 3, '-')
ylabel('l-1 norm')
title('Penalty function approximation')
subplot(3, 1, 2)
(n, bins, patches) = hist(A * x2.value - b, 50, range=[-2, 2])
plot(bins, np.power(bins, 2) * 2, '-')
ylabel('l-2 norm')
subplot(3, 1, 3)
(n, bins, patches) = hist(A * xdz.value + b, 50, range=[-2, 2])
zeros = np.array([0 for x in bins])
plot(bins, np.maximum((np.abs(bins) - dz) * 35 / 3, zeros), '-')
ylabel('deadzone')
show()