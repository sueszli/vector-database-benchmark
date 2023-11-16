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
"\nInput parameters\n  n: Number of communication channels or 'buckets'\n  a: Floor above the baseline for each channel at which power can be added\n  sum_x: Total power to be allocated to the n channels\n"

def water_filling(n, a, sum_x: float=1):
    if False:
        while True:
            i = 10
    '\nBoyd and Vandenberghe, Convex Optimization, example 5.2 page 145\nWater-filling.\n\nThis problem arises in information theory, in allocating power to a set of\nn communication channels in order to maximise the total channel capacity.\nThe variable x_i represents the transmitter power allocated to the ith channel,\nand log(α_i+x_i) gives the capacity or maximum communication rate of the channel.\nThe objective is to minimize  -∑log(α_i+x_i) subject to the constraint ∑x_i = 1\n  '
    x = cvx.Variable(n)
    alpha = cvx.Parameter(n, nonneg=True)
    alpha.value = a
    obj = cvx.Maximize(cvx.sum(cvx.log(alpha + x)))
    constraints = [x >= 0, cvx.sum(x) - sum_x == 0]
    prob = cvx.Problem(obj, constraints)
    prob.solve()
    if prob.status == 'optimal':
        return (prob.status, prob.value, x.value)
    else:
        return (prob.status, np.nan, np.nan)
if __name__ == '__main__':
    print(water_filling.__doc__)
    np.set_printoptions(precision=3)
    buckets = 3
    alpha = np.array([0.8, 1.0, 1.2])
    print('Number of buckets = %s' % buckets)
    (stat, prob, x) = water_filling(buckets, alpha)
    print('Problem status: ', stat)
    print('Optimal communication rate = %.4g ' % prob)
    print('Transmitter powers:\n', x)