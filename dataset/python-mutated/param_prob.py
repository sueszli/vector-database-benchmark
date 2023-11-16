"""
Copyright 2013 Steven Diamond, 2017 Akshay Agrawal

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

class ParamProb:
    """An abstract base class for parameterized problems.

    Parameterized problems are produced during the first canonicalization
    and allow canonicalization to be short-circuited for future solves.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def is_mixed_integer(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Is the problem mixed-integer?'
        raise NotImplementedError()

    @abc.abstractproperty
    def apply_parameters(self, id_to_param_value=None, zero_offset: bool=False, keep_zeros: bool=False):
        if False:
            while True:
                i = 10
        'Returns A, b after applying parameters (and reshaping).\n\n        Args:\n          id_to_param_value: (optional) dict mapping parameter ids to values\n          zero_offset: (optional) if True, zero out the constant offset in the\n                       parameter vector\n          keep_zeros: (optional) if True, store explicit zeros in A where\n                        parameters are affected\n        '
        raise NotImplementedError()