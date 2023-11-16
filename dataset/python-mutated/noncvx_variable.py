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
import abc
import cvxopt
import cvxpy
import cvxpy.interface as intf

class NonCvxVariable(cvxpy.Variable):
    __metaclass__ = abc.ABCMeta

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super(NonCvxVariable, self).__init__(*args, **kwargs)
        self.noncvx = True
        self.z = cvxpy.Parameter(*self.size)
        self.init_z()
        self.u = cvxpy.Parameter(*self.size)
        self.u.value = cvxopt.matrix(0, self.size, tc='d')

    def init_z(self):
        if False:
            print('Hello World!')
        self.z.value = cvxopt.matrix(0, self.size, tc='d')

    def validate_matrix(self, matrix):
        if False:
            i = 10
            return i + 15
        if self.size != intf.shape(matrix):
            raise Exception("The argument's dimensions must match the variable's dimensions.")

    def round(self, matrix):
        if False:
            return 10
        self.validate_matrix(matrix)
        return self._round(matrix)

    @abc.abstractmethod
    def _round(self, matrix):
        if False:
            return 10
        raise NotImplementedError()

    def fix(self, matrix):
        if False:
            print('Hello World!')
        matrix = self.round(matrix)
        return self._fix(matrix)

    @abc.abstractmethod
    def _fix(self, matrix):
        if False:
            print('Hello World!')
        raise NotImplementedError()