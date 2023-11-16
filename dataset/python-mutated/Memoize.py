from __future__ import print_function
__revision__ = 'src/engine/SCons/Memoize.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
__doc__ = 'Memoizer\n\nA decorator-based implementation to count hits and misses of the computed\nvalues that various methods cache in memory.\n\nUse of this modules assumes that wrapped methods be coded to cache their\nvalues in a consistent way. In particular, it requires that the class uses a\ndictionary named "_memo" to store the cached values.\n\nHere is an example of wrapping a method that returns a computed value,\nwith no input parameters::\n\n    @SCons.Memoize.CountMethodCall\n    def foo(self):\n\n        try:                                                    # Memoization\n            return self._memo[\'foo\']                            # Memoization\n        except KeyError:                                        # Memoization\n            pass                                                # Memoization\n\n        result = self.compute_foo_value()\n\n        self._memo[\'foo\'] = result                              # Memoization\n\n        return result\n\nHere is an example of wrapping a method that will return different values\nbased on one or more input arguments::\n\n    def _bar_key(self, argument):                               # Memoization\n        return argument                                         # Memoization\n\n    @SCons.Memoize.CountDictCall(_bar_key)\n    def bar(self, argument):\n\n        memo_key = argument                                     # Memoization\n        try:                                                    # Memoization\n            memo_dict = self._memo[\'bar\']                       # Memoization\n        except KeyError:                                        # Memoization\n            memo_dict = {}                                      # Memoization\n            self._memo[\'dict\'] = memo_dict                      # Memoization\n        else:                                                   # Memoization\n            try:                                                # Memoization\n                return memo_dict[memo_key]                      # Memoization\n            except KeyError:                                    # Memoization\n                pass                                            # Memoization\n\n        result = self.compute_bar_value(argument)\n\n        memo_dict[memo_key] = result                            # Memoization\n\n        return result\n\nDeciding what to cache is tricky, because different configurations\ncan have radically different performance tradeoffs, and because the\ntradeoffs involved are often so non-obvious.  Consequently, deciding\nwhether or not to cache a given method will likely be more of an art than\na science, but should still be based on available data from this module.\nHere are some VERY GENERAL guidelines about deciding whether or not to\ncache return values from a method that\'s being called a lot:\n\n    --  The first question to ask is, "Can we change the calling code\n        so this method isn\'t called so often?"  Sometimes this can be\n        done by changing the algorithm.  Sometimes the *caller* should\n        be memoized, not the method you\'re looking at.\n\n    --  The memoized function should be timed with multiple configurations\n        to make sure it doesn\'t inadvertently slow down some other\n        configuration.\n\n    --  When memoizing values based on a dictionary key composed of\n        input arguments, you don\'t need to use all of the arguments\n        if some of them don\'t affect the return values.\n\n'
use_memoizer = None
CounterList = {}

class Counter(object):
    """
    Base class for counting memoization hits and misses.

    We expect that the initialization in a matching decorator will
    fill in the correct class name and method name that represents
    the name of the function being counted.
    """

    def __init__(self, cls_name, method_name):
        if False:
            print('Hello World!')
        '\n        '
        self.cls_name = cls_name
        self.method_name = method_name
        self.hit = 0
        self.miss = 0

    def key(self):
        if False:
            for i in range(10):
                print('nop')
        return self.cls_name + '.' + self.method_name

    def display(self):
        if False:
            while True:
                i = 10
        print('    {:7d} hits {:7d} misses    {}()'.format(self.hit, self.miss, self.key()))

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        try:
            return self.key() == other.key()
        except AttributeError:
            return True

class CountValue(Counter):
    """
    A counter class for simple, atomic memoized values.

    A CountValue object should be instantiated in a decorator for each of
    the class's methods that memoizes its return value by simply storing
    the return value in its _memo dictionary.
    """

    def count(self, *args, **kw):
        if False:
            i = 10
            return i + 15
        ' Counts whether the memoized value has already been\n            set (a hit) or not (a miss).\n        '
        obj = args[0]
        if self.method_name in obj._memo:
            self.hit = self.hit + 1
        else:
            self.miss = self.miss + 1

class CountDict(Counter):
    """
    A counter class for memoized values stored in a dictionary, with
    keys based on the method's input arguments.

    A CountDict object is instantiated in a decorator for each of the
    class's methods that memoizes its return value in a dictionary,
    indexed by some key that can be computed from one or more of
    its input arguments.
    """

    def __init__(self, cls_name, method_name, keymaker):
        if False:
            return 10
        '\n        '
        Counter.__init__(self, cls_name, method_name)
        self.keymaker = keymaker

    def count(self, *args, **kw):
        if False:
            i = 10
            return i + 15
        ' Counts whether the computed key value is already present\n           in the memoization dictionary (a hit) or not (a miss).\n        '
        obj = args[0]
        try:
            memo_dict = obj._memo[self.method_name]
        except KeyError:
            self.miss = self.miss + 1
        else:
            key = self.keymaker(*args, **kw)
            if key in memo_dict:
                self.hit = self.hit + 1
            else:
                self.miss = self.miss + 1

def Dump(title=None):
    if False:
        while True:
            i = 10
    ' Dump the hit/miss count for all the counters\n        collected so far.\n    '
    if title:
        print(title)
    for counter in sorted(CounterList):
        CounterList[counter].display()

def EnableMemoization():
    if False:
        return 10
    global use_memoizer
    use_memoizer = 1

def CountMethodCall(fn):
    if False:
        print('Hello World!')
    ' Decorator for counting memoizer hits/misses while retrieving\n        a simple value in a class method. It wraps the given method\n        fn and uses a CountValue object to keep track of the\n        caching statistics.\n        Wrapping gets enabled by calling EnableMemoization().\n    '
    if use_memoizer:

        def wrapper(self, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            global CounterList
            key = self.__class__.__name__ + '.' + fn.__name__
            if key not in CounterList:
                CounterList[key] = CountValue(self.__class__.__name__, fn.__name__)
            CounterList[key].count(self, *args, **kwargs)
            return fn(self, *args, **kwargs)
        wrapper.__name__ = fn.__name__
        return wrapper
    else:
        return fn

def CountDictCall(keyfunc):
    if False:
        for i in range(10):
            print('nop')
    ' Decorator for counting memoizer hits/misses while accessing\n        dictionary values with a key-generating function. Like\n        CountMethodCall above, it wraps the given method\n        fn and uses a CountDict object to keep track of the\n        caching statistics. The dict-key function keyfunc has to\n        get passed in the decorator call and gets stored in the\n        CountDict instance.\n        Wrapping gets enabled by calling EnableMemoization().\n    '

    def decorator(fn):
        if False:
            print('Hello World!')
        if use_memoizer:

            def wrapper(self, *args, **kwargs):
                if False:
                    i = 10
                    return i + 15
                global CounterList
                key = self.__class__.__name__ + '.' + fn.__name__
                if key not in CounterList:
                    CounterList[key] = CountDict(self.__class__.__name__, fn.__name__, keyfunc)
                CounterList[key].count(self, *args, **kwargs)
                return fn(self, *args, **kwargs)
            wrapper.__name__ = fn.__name__
            return wrapper
        else:
            return fn
    return decorator