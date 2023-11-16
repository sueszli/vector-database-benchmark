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
import numpy as np
from pylab import axis, plot, show, title, xlabel, ylabel
from cvxpy import Minimize, Parameter, Problem, Variable, quad_form
u = Parameter()
x = Variable()
objective = Minimize(quad_form(x, 1) + 1)
constraint = [quad_form(x, 1) - 6 * x + 8 <= u]
p = Problem(objective, constraint)

def get_x(u_value):
    if False:
        for i in range(10):
            print('nop')
    u.value = u_value
    result = p.solve()
    return x.value
u_values = np.linspace(-0.9, 10, num=50)
x_values = [get_x(value) for value in u_values]
pool = Pool(processes=4)
x_values = pool.map(get_x, u_values)
plot(u_values, x_values)
title('Sensitivity Analysis: p*(u) vs u')
xlabel('u')
ylabel('p*(u)')
axis([-2, 10, -1, 3])
show()