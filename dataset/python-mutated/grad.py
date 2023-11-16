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
import scipy.sparse as sp

def constant_grad(expr):
    if False:
        print('Hello World!')
    'Returns the gradient of constant terms in an expression.\n\n    Matrix expressions are vectorized, so the gradient is a matrix.\n\n    Args:\n        expr: An expression.\n\n    Returns:\n        A map of variable value to empty SciPy CSC sparse matrices.\n    '
    grad = {}
    for var in expr.variables():
        rows = var.size
        cols = expr.size
        if (rows, cols) == (1, 1):
            grad[var] = 0.0
        else:
            grad[var] = sp.csc_matrix((rows, cols), dtype='float64')
    return grad

def error_grad(expr):
    if False:
        while True:
            i = 10
    'Returns a gradient of all None.\n\n    Args:\n        expr: An expression.\n\n    Returns:\n        A map of variable value to None.\n    '
    return {var: None for var in expr.variables()}