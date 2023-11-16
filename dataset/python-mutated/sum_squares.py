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
from cvxpy.atoms.quad_over_lin import quad_over_lin

def sum_squares(expr):
    if False:
        return 10
    'The sum of the squares of the entries.\n\n    Parameters\n    ----------\n    expr: Expression\n        The expression to take the sum of squares of.\n\n    Returns\n    -------\n    Expression\n        An expression representing the sum of squares.\n    '
    return quad_over_lin(expr, 1)