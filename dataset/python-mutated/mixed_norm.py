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
from typing import Union
from cvxpy.atoms.norm import norm
from cvxpy.expressions.expression import Expression

def mixed_norm(X, p: Union[int, str]=2, q: Union[int, str]=1):
    if False:
        i = 10
        return i + 15
    'Lp,q norm; :math:`(\\sum_k (\\sum_l \\lvert x_{k,l} \\rvert^p)^{q/p})^{1/q}`.\n\n    Parameters\n    ----------\n    X : Expression or numeric constant\n        The matrix to take the l_{p,q} norm of.\n    p : int or str, optional\n        The type of inner norm.\n    q : int or str, optional\n        The type of outer norm.\n\n    Returns\n    -------\n    Expression\n        An Expression representing the mixed norm.\n    '
    X = Expression.cast_to_const(X)
    vecnorms = norm(X, p, axis=1)
    return norm(vecnorms, q)