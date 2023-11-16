"""
http://code.activestate.com/recipes/413838-memento-closure/

*TL;DR
Provides the ability to restore an object to its previous state.
"""
from typing import Callable, List
from copy import copy, deepcopy

def memento(obj, deep=False):
    if False:
        return 10
    state = deepcopy(obj.__dict__) if deep else copy(obj.__dict__)

    def restore():
        if False:
            return 10
        obj.__dict__.clear()
        obj.__dict__.update(state)
    return restore

class Transaction:
    """A transaction guard.

    This is, in fact, just syntactic sugar around a memento closure.
    """
    deep = False
    states: List[Callable[[], None]] = []

    def __init__(self, deep, *targets):
        if False:
            print('Hello World!')
        self.deep = deep
        self.targets = targets
        self.commit()

    def commit(self):
        if False:
            return 10
        self.states = [memento(target, self.deep) for target in self.targets]

    def rollback(self):
        if False:
            for i in range(10):
                print('nop')
        for a_state in self.states:
            a_state()

class Transactional:
    """Adds transactional semantics to methods. Methods decorated  with

    @Transactional will rollback to entry-state upon exceptions.
    """

    def __init__(self, method):
        if False:
            i = 10
            return i + 15
        self.method = method

    def __get__(self, obj, T):
        if False:
            for i in range(10):
                print('nop')
        '\n        A decorator that makes a function transactional.\n\n        :param method: The function to be decorated.\n        '

        def transaction(*args, **kwargs):
            if False:
                while True:
                    i = 10
            state = memento(obj)
            try:
                return self.method(obj, *args, **kwargs)
            except Exception as e:
                state()
                raise e
        return transaction

class NumObj:

    def __init__(self, value):
        if False:
            print('Hello World!')
        self.value = value

    def __repr__(self):
        if False:
            return 10
        return f'<{self.__class__.__name__}: {self.value!r}>'

    def increment(self):
        if False:
            while True:
                i = 10
        self.value += 1

    @Transactional
    def do_stuff(self):
        if False:
            for i in range(10):
                print('nop')
        self.value = '1111'
        self.increment()

def main():
    if False:
        return 10
    "\n    >>> num_obj = NumObj(-1)\n    >>> print(num_obj)\n    <NumObj: -1>\n\n    >>> a_transaction = Transaction(True, num_obj)\n\n    >>> try:\n    ...    for i in range(3):\n    ...        num_obj.increment()\n    ...        print(num_obj)\n    ...    a_transaction.commit()\n    ...    print('-- committed')\n    ...    for i in range(3):\n    ...        num_obj.increment()\n    ...        print(num_obj)\n    ...    num_obj.value += 'x'  # will fail\n    ...    print(num_obj)\n    ... except Exception:\n    ...    a_transaction.rollback()\n    ...    print('-- rolled back')\n    <NumObj: 0>\n    <NumObj: 1>\n    <NumObj: 2>\n    -- committed\n    <NumObj: 3>\n    <NumObj: 4>\n    <NumObj: 5>\n    -- rolled back\n\n    >>> print(num_obj)\n    <NumObj: 2>\n\n    >>> print('-- now doing stuff ...')\n    -- now doing stuff ...\n\n    >>> try:\n    ...    num_obj.do_stuff()\n    ... except Exception:\n    ...    print('-> doing stuff failed!')\n    ...    import sys\n    ...    import traceback\n    ...    traceback.print_exc(file=sys.stdout)\n    -> doing stuff failed!\n    Traceback (most recent call last):\n    ...\n    TypeError: ...str...int...\n\n    >>> print(num_obj)\n    <NumObj: 2>\n    "
if __name__ == '__main__':
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)