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
from abc import ABCMeta, abstractmethod

class Reduction:
    """Abstract base class for reductions.

    A reduction is an actor that transforms a problem into an
    equivalent problem. By equivalent we mean that there exists
    a mapping between solutions of either problem: if we reduce a problem
    :math:`A` to another problem :math:`B` and then proceed to find a solution
    to :math:`B`, we can convert it to a solution of :math:`A` with at most a
    moderate amount of effort.

    A reduction that is instantiated with a non-None problem offers
    two key methods: `reduce` and `retrieve`. The `reduce()` method converts
    the problem the reduction was instantiated with to an equivalent
    problem. The `retrieve()` method takes as an argument a Solution
    for the equivalent problem and returns a Solution for the problem
    owned by the reduction.

    Every reduction offers three low-level methods: accepts, apply, and invert.
    The accepts method of a particular reduction specifies the types of problems
    that it is applicable to; the apply method takes a problem and reduces
    it to an equivalent form, and the invert method maps solutions
    from reduced-to problems to their problems of provenance.

    Parameters
    ----------
    problem : Problem
        A problem owned by this reduction; possibly None.
    """
    __metaclass__ = ABCMeta

    def __init__(self, problem=None) -> None:
        if False:
            while True:
                i = 10
        'Construct a reduction for reducing `problem`.\n\n        If `problem` is not None, then a subsequent invocation of `reduce()`\n        will reduce `problem` and return an equivalent one.\n        '
        self.problem = problem

    def accepts(self, problem):
        if False:
            return 10
        'States whether the reduction accepts a problem.\n\n        Parameters\n        ----------\n        problem : Problem\n            The problem to check.\n\n        Returns\n        -------\n        bool\n            True if the reduction can be applied, False otherwise.\n        '
        raise NotImplementedError()

    def reduce(self):
        if False:
            i = 10
            return i + 15
        'Reduces the owned problem to an equivalent problem.\n\n        Returns\n        -------\n        Problem or dict\n            An equivalent problem, encoded either as a Problem or a dict.\n\n        Raises\n        ------\n        ValueError\n            If this Reduction was constructed without a Problem.\n        '
        if hasattr(self, '_emitted_problem'):
            return self._emitted_problem
        if self.problem is None:
            raise ValueError('The reduction was constructed without a Problem.')
        (problem, retrieval_data) = self.apply(self.problem)
        self._emitted_problem = problem
        self._retrieval_data = retrieval_data
        return problem

    def retrieve(self, solution):
        if False:
            i = 10
            return i + 15
        'Retrieves a solution to the owned problem.\n\n        Parameters\n        ----------\n        solution : Solution\n            A solution to the problem emitted by `reduce()`.\n\n        Returns\n        -------\n        Solution\n            A solution to the owned problem.\n\n        Raises\n        ------\n        ValueError\n            If `self.problem` is None, or if `reduce()` was not previously\n            called.\n        '
        if not hasattr(self, '_retrieval_data'):
            raise ValueError('`reduce()` must be called before `retrieve()`.')
        return self.invert(solution, self._retrieval_data)

    @abstractmethod
    def apply(self, problem):
        if False:
            return 10
        'Applies the reduction to a problem and returns an equivalent problem.\n\n        Parameters\n        ----------\n        problem : Problem\n            The problem to which the reduction will be applied.\n\n        Returns\n        -------\n        Problem or dict\n            An equivalent problem, encoded either as a Problem or a dict.\n\n        InverseData, list or dict\n            Data needed by the reduction in order to invert this particular\n            application.\n        '
        raise NotImplementedError()

    @abstractmethod
    def invert(self, solution, inverse_data):
        if False:
            return 10
        'Returns a solution to the original problem given the inverse_data.\n\n        Parameters\n        ----------\n        solution : Solution\n            A solution to a problem that generated the inverse_data.\n        inverse_data\n            The data encoding the original problem.\n\n        Returns\n        -------\n        Solution\n            A solution to the original problem.\n        '
        raise NotImplementedError()