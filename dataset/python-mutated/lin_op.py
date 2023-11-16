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

THIS FILE IS DEPRECATED AND MAY BE REMOVED WITHOUT WARNING!
DO NOT CALL THESE FUNCTIONS IN YOUR CODE!
"""
from typing import Tuple

class LinOp:

    def __init__(self, type, shape: Tuple[int, ...], args, data) -> None:
        if False:
            print('Hello World!')
        self.type = type
        self.shape = shape
        self.args = args
        self.data = data

    def __repr__(self) -> str:
        if False:
            return 10
        return f'LinOp({self.type}, {self.shape})'
VARIABLE = 'variable'
PROMOTE = 'promote'
MUL = 'mul'
RMUL = 'rmul'
MUL_ELEM = 'mul_elem'
DIV = 'div'
SUM = 'sum'
NEG = 'neg'
INDEX = 'index'
TRANSPOSE = 'transpose'
SUM_ENTRIES = 'sum_entries'
TRACE = 'trace'
RESHAPE = 'reshape'
DIAG_VEC = 'diag_vec'
DIAG_MAT = 'diag_mat'
UPPER_TRI = 'upper_tri'
CONV = 'conv'
KRON_R = 'kron_r'
KRON_L = 'kron_l'
HSTACK = 'hstack'
VSTACK = 'vstack'
SCALAR_CONST = 'scalar_const'
DENSE_CONST = 'dense_const'
SPARSE_CONST = 'sparse_const'
PARAM = 'param'
NO_OP = 'no_op'
CONSTANT_ID = -1