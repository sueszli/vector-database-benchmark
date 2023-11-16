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
from cvxpy.atoms import promote, reshape
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variable import Variable

def max_canon(expr, args):
    if False:
        while True:
            i = 10
    x = args[0]
    shape = expr.shape
    axis = expr.axis
    t = Variable(shape)
    if axis is None:
        promoted_t = promote(t, x.shape)
    elif axis == 0:
        promoted_t = Constant(np.ones((x.shape[0], 1))) @ reshape(t, (1, x.shape[1]))
    else:
        promoted_t = reshape(t, (x.shape[0], 1)) @ Constant(np.ones((1, x.shape[1])))
    constraints = [x <= promoted_t]
    return (t, constraints)