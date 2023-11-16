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

def version():
    if False:
        for i in range(10):
            print('nop')
    from cvxpy import __version__
    return __version__

def expression():
    if False:
        while True:
            i = 10
    from cvxpy.expressions import expression
    return expression.Expression

def add_expr():
    if False:
        print('Hello World!')
    from cvxpy.atoms.affine import add_expr
    return add_expr.AddExpression

def conj():
    if False:
        for i in range(10):
            print('nop')
    from cvxpy.atoms.affine import conj
    return conj.conj

def constant():
    if False:
        for i in range(10):
            print('nop')
    from cvxpy.expressions import constants
    return constants.Constant

def parameter():
    if False:
        for i in range(10):
            print('nop')
    from cvxpy.expressions import constants
    return constants.Parameter

def variable():
    if False:
        i = 10
        return i + 15
    from cvxpy.expressions import variable
    return variable.Variable

def index():
    if False:
        print('Hello World!')
    from cvxpy.atoms.affine import index
    return index.index

def special_index():
    if False:
        i = 10
        return i + 15
    from cvxpy.atoms.affine import index
    return index.special_index

def indicator():
    if False:
        return 10
    from cvxpy.transforms.indicator import indicator
    return indicator

def minimize():
    if False:
        i = 10
        return i + 15
    from cvxpy.problems import objective
    return objective.Minimize

def matmul_expr():
    if False:
        while True:
            i = 10
    from cvxpy.atoms.affine import binary_operators
    return binary_operators.MulExpression

def elmul_expr():
    if False:
        print('Hello World!')
    from cvxpy.atoms.affine import binary_operators
    return binary_operators.multiply

def div_expr():
    if False:
        return 10
    from cvxpy.atoms.affine import binary_operators
    return binary_operators.DivExpression

def neg_expr():
    if False:
        i = 10
        return i + 15
    from cvxpy.atoms.affine import unary_operators
    return unary_operators.NegExpression

def abs():
    if False:
        return 10
    from cvxpy.atoms.elementwise import abs
    return abs.abs

def lambda_min():
    if False:
        while True:
            i = 10
    from cvxpy.atoms import lambda_min
    return lambda_min

def pos():
    if False:
        return 10
    from cvxpy.atoms.elementwise import pos
    return pos.pos

def promote():
    if False:
        for i in range(10):
            print('nop')
    from cvxpy.atoms.affine.promote import promote
    return promote

def maximum():
    if False:
        return 10
    from cvxpy.atoms.elementwise import maximum
    return maximum.maximum

def minimum():
    if False:
        print('Hello World!')
    from cvxpy.atoms.elementwise import minimum
    return minimum.minimum

def neg():
    if False:
        print('Hello World!')
    from cvxpy.atoms.elementwise import neg
    return neg.neg

def partial_optimize():
    if False:
        for i in range(10):
            print('nop')
    from cvxpy.transforms import partial_optimize
    return partial_optimize

def partial_problem():
    if False:
        i = 10
        return i + 15
    from cvxpy.transforms.partial_optimize import PartialProblem
    return PartialProblem

def power():
    if False:
        i = 10
        return i + 15
    from cvxpy.atoms.elementwise import power
    return power.power

def problem():
    if False:
        for i in range(10):
            print('nop')
    from cvxpy.problems import problem
    return problem.Problem

def reshape():
    if False:
        while True:
            i = 10
    from cvxpy.atoms.affine import reshape
    return reshape.reshape

def transpose():
    if False:
        i = 10
        return i + 15
    from cvxpy.atoms.affine import transpose
    return transpose.transpose

def vec():
    if False:
        print('Hello World!')
    from cvxpy.atoms.affine import vec
    return vec.vec

def vstack():
    if False:
        for i in range(10):
            print('nop')
    from cvxpy.atoms.affine import vstack
    return vstack.vstack

def quad_form():
    if False:
        while True:
            i = 10
    from cvxpy.atoms.quad_form import QuadForm
    return QuadForm