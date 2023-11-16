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
import cvxopt
import numpy as np
from cvxpy.expressions.constants import Parameter
from cvxpy.expressions.variable import Variable

class Boolean(Variable):

    def __init__(self, rows: int=1, cols: int=1, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        self._LB = Parameter(rows, cols)
        self._LB.value = cvxopt.matrix(0, (rows, cols), tc='d')
        self._UB = Parameter(rows, cols)
        self._UB.value = cvxopt.matrix(1, (rows, cols), tc='d')
        self._fix_values = cvxopt.matrix(False, (rows, cols))
        super(Boolean, self).__init__(rows, cols, *args, **kwargs)

    def round(self):
        if False:
            print('Hello World!')
        self.LB = cvxopt.matrix(self._rounded, self.size)
        self.UB = cvxopt.matrix(self._rounded, self.size)

    def relax(self):
        if False:
            print('Hello World!')
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                if not self.fix_values[i, j]:
                    self.LB[i, j] = 0
                    self.UB[i, j] = 1

    def set(self, value):
        if False:
            print('Hello World!')
        if not isinstance(value, bool):
            raise 'Must set to boolean value'
        self.LB = cvxopt.matrix(value, self.size)
        self.UB = cvxopt.matrix(value, self.size)
        self.fix_values = cvxopt.matrix(True, self.size)

    def unset(self):
        if False:
            while True:
                i = 10
        self.fix_values = cvxopt.matrix(False, self.size)

    @property
    def _rounded(self):
        if False:
            while True:
                i = 10
        if self.size == (1, 1):
            return round(self.value)
        else:
            return np.around(self.value)

    @property
    def LB(self):
        if False:
            for i in range(10):
                print('nop')
        return self._LB.value

    @LB.setter
    def LB(self, value):
        if False:
            i = 10
            return i + 15
        self._LB.value = value

    @property
    def UB(self):
        if False:
            i = 10
            return i + 15
        return self._UB.value

    @UB.setter
    def UB(self, value):
        if False:
            i = 10
            return i + 15
        self._UB.value = value

    @property
    def fix_values(self):
        if False:
            return 10
        return self._fix_values

    @fix_values.setter
    def fix_values(self, value):
        if False:
            while True:
                i = 10
        self._fix_values = value