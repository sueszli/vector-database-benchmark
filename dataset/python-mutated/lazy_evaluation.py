"""
Lazily-evaluated property pattern in Python.

https://en.wikipedia.org/wiki/Lazy_evaluation

*References:
bottle
https://github.com/bottlepy/bottle/blob/cafc15419cbb4a6cb748e6ecdccf92893bb25ce5/bottle.py#L270
django
https://github.com/django/django/blob/ffd18732f3ee9e6f0374aff9ccf350d85187fac2/django/utils/functional.py#L19
pip
https://github.com/pypa/pip/blob/cb75cca785629e15efb46c35903827b3eae13481/pip/utils/__init__.py#L821
pyramid
https://github.com/Pylons/pyramid/blob/7909e9503cdfc6f6e84d2c7ace1d3c03ca1d8b73/pyramid/decorator.py#L4
werkzeug
https://github.com/pallets/werkzeug/blob/5a2bf35441006d832ab1ed5a31963cbc366c99ac/werkzeug/utils.py#L35

*TL;DR
Delays the eval of an expr until its value is needed and avoids repeated evals.
"""
import functools

class lazy_property:

    def __init__(self, function):
        if False:
            for i in range(10):
                print('nop')
        self.function = function
        functools.update_wrapper(self, function)

    def __get__(self, obj, type_):
        if False:
            for i in range(10):
                print('nop')
        if obj is None:
            return self
        val = self.function(obj)
        obj.__dict__[self.function.__name__] = val
        return val

def lazy_property2(fn):
    if False:
        i = 10
        return i + 15
    '\n    A lazy property decorator.\n\n    The function decorated is called the first time to retrieve the result and\n    then that calculated result is used the next time you access the value.\n    '
    attr = '_lazy__' + fn.__name__

    @property
    def _lazy_property(self):
        if False:
            i = 10
            return i + 15
        if not hasattr(self, attr):
            setattr(self, attr, fn(self))
        return getattr(self, attr)
    return _lazy_property

class Person:

    def __init__(self, name, occupation):
        if False:
            print('Hello World!')
        self.name = name
        self.occupation = occupation
        self.call_count2 = 0

    @lazy_property
    def relatives(self):
        if False:
            return 10
        relatives = 'Many relatives.'
        return relatives

    @lazy_property2
    def parents(self):
        if False:
            i = 10
            return i + 15
        self.call_count2 += 1
        return 'Father and mother'

def main():
    if False:
        return 10
    "\n    >>> Jhon = Person('Jhon', 'Coder')\n\n    >>> Jhon.name\n    'Jhon'\n    >>> Jhon.occupation\n    'Coder'\n\n    # Before we access `relatives`\n    >>> sorted(Jhon.__dict__.items())\n    [('call_count2', 0), ('name', 'Jhon'), ('occupation', 'Coder')]\n\n    >>> Jhon.relatives\n    'Many relatives.'\n\n    # After we've accessed `relatives`\n    >>> sorted(Jhon.__dict__.items())\n    [('call_count2', 0), ..., ('relatives', 'Many relatives.')]\n\n    >>> Jhon.parents\n    'Father and mother'\n\n    >>> sorted(Jhon.__dict__.items())\n    [('_lazy__parents', 'Father and mother'), ('call_count2', 1), ..., ('relatives', 'Many relatives.')]\n\n    >>> Jhon.parents\n    'Father and mother'\n\n    >>> Jhon.call_count2\n    1\n    "
if __name__ == '__main__':
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)