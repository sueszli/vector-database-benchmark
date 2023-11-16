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
from cvxpy.constraints.exponential import ExpCone
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variable import Variable

def log_canon(expr, args):
    if False:
        while True:
            i = 10
    x = args[0]
    shape = expr.shape
    t = Variable(shape)
    ones = Constant(np.ones(shape))
    constraints = [ExpCone(t, ones, x)]
    return (t, constraints)