"""
Copyright, the CVXPY authors

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
from cvxpy.atoms.elementwise.inv_pos import inv_pos
from cvxpy.atoms.elementwise.power import power
from cvxpy.atoms.geo_mean import geo_mean

def inv_prod(value):
    if False:
        while True:
            i = 10
    'The reciprocal of a product of the entries of a vector ``x``.\n\n    Parameters\n    ----------\n    x : Expression or numeric\n        The expression whose reciprocal product is to be computed. Must have\n        positive entries.\n\n    Returns\n    -------\n    Expression\n        .. math::\n            \\left(\\prod_{i=1}^n x_i\\right)^{-1},\n\n        where :math:`n` is the length of :math:`x`.\n    '
    return power(inv_pos(geo_mean(value)), int(sum(value.shape)))