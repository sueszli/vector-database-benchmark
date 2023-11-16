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
'\nInput parameters\n  P: channel transition matrix P_ij(t) = P(output|input) at time t\n  n: size of input\n  m: size of output\n'

def channel_capacity(n, m, sum_x: float=1.0):
    if False:
        return 10
    '\nBoyd and Vandenberghe, Convex Optimization, exercise 4.57 page 207\nCapacity of a communication channel.\n\nWe consider a communication channel, with input x(t)∈{1,..,n} and\noutput Y(t)∈{1,...,m}, for t=1,2,... .The relation between the\ninput and output is given statistically:\np_(i,j) = ℙ(Y(t)=i|X(t)=j), i=1,..,m  j=1,...,m\nThe matrix P ∈ ℝ^(m*n) is called the channel transition matrix, and\nthe channel is called a discrete memoryless channel. Assuming X has a\nprobability distribution denoted x ∈ ℝ^n, i.e.,\nx_j = ℙ(X=j), j=1,...,n\nThe mutual information between X and Y is given by\n∑(∑(x_j p_(i,j)log_2(p_(i,j)/∑(x_k p_(i,k)))))\nThen channel capacity C is given by\nC = sup I(X;Y).\nWith a variable change of y = Px this becomes\nI(X;Y)=  c^T x - ∑(y_i log_2 y_i)\nwhere c_j = ∑(p_(i,j)log_2(p_(i,j)))\n  '
    if n * m == 0:
        print('The range of both input and output values must be greater than zero')
        return ('failed', np.nan, np.nan)
    P = np.ones((m, n))
    x = cvx.Variable(rows=n, cols=1)
    y = P * x
    c = np.sum(P * np.log2(P), axis=0)
    I = c * x + cvx.sum(cvx.entr(y))
    obj = cvx.Minimize(-I)
    constraints = [cvx.sum(x) == sum_x, x >= 0]
    prob = cvx.Problem(obj, constraints)
    prob.solve()
    if prob.status == 'optimal':
        return (prob.status, prob.value, x.value)
    else:
        return (prob.status, np.nan, np.nan)
if __name__ == '__main__':
    print(channel_capacity.__doc__)
    np.set_printoptions(precision=3)
    n = 2
    m = 2
    print('Number of input values=%s' % n)
    print('Number of outputs=%s' % m)
    (stat, C, x) = channel_capacity(n, m)
    print('Problem status ', stat)
    print('Optimal value of C = %.4g' % C)
    print('Optimal variable x = \n', x)