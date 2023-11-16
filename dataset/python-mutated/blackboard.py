"""
@author: Eugene Duboviy <eugene.dubovoy@gmail.com> | github.com/duboviy

In Blackboard pattern several specialised sub-systems (knowledge sources)
assemble their knowledge to build a possibly partial or approximate solution.
In this way, the sub-systems work together to solve the problem,
where the solution is the sum of its parts.

https://en.wikipedia.org/wiki/Blackboard_system
"""
from __future__ import annotations
import abc
import random

class Blackboard:

    def __init__(self) -> None:
        if False:
            return 10
        self.experts = []
        self.common_state = {'problems': 0, 'suggestions': 0, 'contributions': [], 'progress': 0}

    def add_expert(self, expert: AbstractExpert) -> None:
        if False:
            return 10
        self.experts.append(expert)

class Controller:

    def __init__(self, blackboard: Blackboard) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.blackboard = blackboard

    def run_loop(self):
        if False:
            print('Hello World!')
        '\n        This function is a loop that runs until the progress reaches 100.\n        It checks if an expert is eager to contribute and then calls its contribute method.\n        '
        while self.blackboard.common_state['progress'] < 100:
            for expert in self.blackboard.experts:
                if expert.is_eager_to_contribute:
                    expert.contribute()
        return self.blackboard.common_state['contributions']

class AbstractExpert(metaclass=abc.ABCMeta):

    def __init__(self, blackboard: Blackboard) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.blackboard = blackboard

    @property
    @abc.abstractmethod
    def is_eager_to_contribute(self):
        if False:
            print('Hello World!')
        raise NotImplementedError('Must provide implementation in subclass.')

    @abc.abstractmethod
    def contribute(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('Must provide implementation in subclass.')

class Student(AbstractExpert):

    @property
    def is_eager_to_contribute(self) -> bool:
        if False:
            i = 10
            return i + 15
        return True

    def contribute(self) -> None:
        if False:
            i = 10
            return i + 15
        self.blackboard.common_state['problems'] += random.randint(1, 10)
        self.blackboard.common_state['suggestions'] += random.randint(1, 10)
        self.blackboard.common_state['contributions'] += [self.__class__.__name__]
        self.blackboard.common_state['progress'] += random.randint(1, 2)

class Scientist(AbstractExpert):

    @property
    def is_eager_to_contribute(self) -> int:
        if False:
            print('Hello World!')
        return random.randint(0, 1)

    def contribute(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.blackboard.common_state['problems'] += random.randint(10, 20)
        self.blackboard.common_state['suggestions'] += random.randint(10, 20)
        self.blackboard.common_state['contributions'] += [self.__class__.__name__]
        self.blackboard.common_state['progress'] += random.randint(10, 30)

class Professor(AbstractExpert):

    @property
    def is_eager_to_contribute(self) -> bool:
        if False:
            return 10
        return True if self.blackboard.common_state['problems'] > 100 else False

    def contribute(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.blackboard.common_state['problems'] += random.randint(1, 2)
        self.blackboard.common_state['suggestions'] += random.randint(10, 20)
        self.blackboard.common_state['contributions'] += [self.__class__.__name__]
        self.blackboard.common_state['progress'] += random.randint(10, 100)

def main():
    if False:
        print('Hello World!')
    "\n    >>> blackboard = Blackboard()\n    >>> blackboard.add_expert(Student(blackboard))\n    >>> blackboard.add_expert(Scientist(blackboard))\n    >>> blackboard.add_expert(Professor(blackboard))\n\n    >>> c = Controller(blackboard)\n    >>> contributions = c.run_loop()\n\n    >>> from pprint import pprint\n    >>> pprint(contributions)\n    ['Student',\n     'Student',\n     'Student',\n     'Student',\n     'Scientist',\n     'Student',\n     'Student',\n     'Student',\n     'Scientist',\n     'Student',\n     'Scientist',\n     'Student',\n     'Student',\n     'Scientist',\n     'Professor']\n    "
if __name__ == '__main__':
    random.seed(1234)
    import doctest
    doctest.testmod()