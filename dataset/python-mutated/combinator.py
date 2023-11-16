"""
CCG Combinators
"""
from abc import ABCMeta, abstractmethod
from nltk.ccg.api import FunctionalCategory

class UndirectedBinaryCombinator(metaclass=ABCMeta):
    """
    Abstract class for representing a binary combinator.
    Merely defines functions for checking if the function and argument
    are able to be combined, and what the resulting category is.

    Note that as no assumptions are made as to direction, the unrestricted
    combinators can perform all backward, forward and crossed variations
    of the combinators; these restrictions must be added in the rule
    class.
    """

    @abstractmethod
    def can_combine(self, function, argument):
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def combine(self, function, argument):
        if False:
            for i in range(10):
                print('nop')
        pass

class DirectedBinaryCombinator(metaclass=ABCMeta):
    """
    Wrapper for the undirected binary combinator.
    It takes left and right categories, and decides which is to be
    the function, and which the argument.
    It then decides whether or not they can be combined.
    """

    @abstractmethod
    def can_combine(self, left, right):
        if False:
            while True:
                i = 10
        pass

    @abstractmethod
    def combine(self, left, right):
        if False:
            for i in range(10):
                print('nop')
        pass

class ForwardCombinator(DirectedBinaryCombinator):
    """
    Class representing combinators where the primary functor is on the left.

    Takes an undirected combinator, and a predicate which adds constraints
    restricting the cases in which it may apply.
    """

    def __init__(self, combinator, predicate, suffix=''):
        if False:
            return 10
        self._combinator = combinator
        self._predicate = predicate
        self._suffix = suffix

    def can_combine(self, left, right):
        if False:
            for i in range(10):
                print('nop')
        return self._combinator.can_combine(left, right) and self._predicate(left, right)

    def combine(self, left, right):
        if False:
            for i in range(10):
                print('nop')
        yield from self._combinator.combine(left, right)

    def __str__(self):
        if False:
            print('Hello World!')
        return f'>{self._combinator}{self._suffix}'

class BackwardCombinator(DirectedBinaryCombinator):
    """
    The backward equivalent of the ForwardCombinator class.
    """

    def __init__(self, combinator, predicate, suffix=''):
        if False:
            while True:
                i = 10
        self._combinator = combinator
        self._predicate = predicate
        self._suffix = suffix

    def can_combine(self, left, right):
        if False:
            while True:
                i = 10
        return self._combinator.can_combine(right, left) and self._predicate(left, right)

    def combine(self, left, right):
        if False:
            print('Hello World!')
        yield from self._combinator.combine(right, left)

    def __str__(self):
        if False:
            return 10
        return f'<{self._combinator}{self._suffix}'

class UndirectedFunctionApplication(UndirectedBinaryCombinator):
    """
    Class representing function application.
    Implements rules of the form:
    X/Y Y -> X (>)
    And the corresponding backwards application rule
    """

    def can_combine(self, function, argument):
        if False:
            return 10
        if not function.is_function():
            return False
        return not function.arg().can_unify(argument) is None

    def combine(self, function, argument):
        if False:
            while True:
                i = 10
        if not function.is_function():
            return
        subs = function.arg().can_unify(argument)
        if subs is None:
            return
        yield function.res().substitute(subs)

    def __str__(self):
        if False:
            while True:
                i = 10
        return ''

def forwardOnly(left, right):
    if False:
        i = 10
        return i + 15
    return left.dir().is_forward()

def backwardOnly(left, right):
    if False:
        print('Hello World!')
    return right.dir().is_backward()
ForwardApplication = ForwardCombinator(UndirectedFunctionApplication(), forwardOnly)
BackwardApplication = BackwardCombinator(UndirectedFunctionApplication(), backwardOnly)

class UndirectedComposition(UndirectedBinaryCombinator):
    """
    Functional composition (harmonic) combinator.
    Implements rules of the form
    X/Y Y/Z -> X/Z (B>)
    And the corresponding backwards and crossed variations.
    """

    def can_combine(self, function, argument):
        if False:
            return 10
        if not (function.is_function() and argument.is_function()):
            return False
        if function.dir().can_compose() and argument.dir().can_compose():
            return not function.arg().can_unify(argument.res()) is None
        return False

    def combine(self, function, argument):
        if False:
            print('Hello World!')
        if not (function.is_function() and argument.is_function()):
            return
        if function.dir().can_compose() and argument.dir().can_compose():
            subs = function.arg().can_unify(argument.res())
            if subs is not None:
                yield FunctionalCategory(function.res().substitute(subs), argument.arg().substitute(subs), argument.dir())

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'B'

def bothForward(left, right):
    if False:
        return 10
    return left.dir().is_forward() and right.dir().is_forward()

def bothBackward(left, right):
    if False:
        i = 10
        return i + 15
    return left.dir().is_backward() and right.dir().is_backward()

def crossedDirs(left, right):
    if False:
        return 10
    return left.dir().is_forward() and right.dir().is_backward()

def backwardBxConstraint(left, right):
    if False:
        while True:
            i = 10
    if not crossedDirs(left, right):
        return False
    if not left.dir().can_cross() and right.dir().can_cross():
        return False
    return left.arg().is_primitive()
ForwardComposition = ForwardCombinator(UndirectedComposition(), forwardOnly)
BackwardComposition = BackwardCombinator(UndirectedComposition(), backwardOnly)
BackwardBx = BackwardCombinator(UndirectedComposition(), backwardBxConstraint, suffix='x')

class UndirectedSubstitution(UndirectedBinaryCombinator):
    """
    Substitution (permutation) combinator.
    Implements rules of the form
    Y/Z (X\\Y)/Z -> X/Z (<Sx)
    And other variations.
    """

    def can_combine(self, function, argument):
        if False:
            print('Hello World!')
        if function.is_primitive() or argument.is_primitive():
            return False
        if function.res().is_primitive():
            return False
        if not function.arg().is_primitive():
            return False
        if not (function.dir().can_compose() and argument.dir().can_compose()):
            return False
        return function.res().arg() == argument.res() and function.arg() == argument.arg()

    def combine(self, function, argument):
        if False:
            while True:
                i = 10
        if self.can_combine(function, argument):
            yield FunctionalCategory(function.res().res(), argument.arg(), argument.dir())

    def __str__(self):
        if False:
            print('Hello World!')
        return 'S'

def forwardSConstraint(left, right):
    if False:
        return 10
    if not bothForward(left, right):
        return False
    return left.res().dir().is_forward() and left.arg().is_primitive()

def backwardSxConstraint(left, right):
    if False:
        return 10
    if not left.dir().can_cross() and right.dir().can_cross():
        return False
    if not bothForward(left, right):
        return False
    return right.res().dir().is_backward() and right.arg().is_primitive()
ForwardSubstitution = ForwardCombinator(UndirectedSubstitution(), forwardSConstraint)
BackwardSx = BackwardCombinator(UndirectedSubstitution(), backwardSxConstraint, 'x')

def innermostFunction(categ):
    if False:
        i = 10
        return i + 15
    while categ.res().is_function():
        categ = categ.res()
    return categ

class UndirectedTypeRaise(UndirectedBinaryCombinator):
    """
    Undirected combinator for type raising.
    """

    def can_combine(self, function, arg):
        if False:
            while True:
                i = 10
        if not (arg.is_function() and arg.res().is_function()):
            return False
        arg = innermostFunction(arg)
        subs = left.can_unify(arg_categ.arg())
        if subs is not None:
            return True
        return False

    def combine(self, function, arg):
        if False:
            for i in range(10):
                print('nop')
        if not (function.is_primitive() and arg.is_function() and arg.res().is_function()):
            return
        arg = innermostFunction(arg)
        subs = function.can_unify(arg.arg())
        if subs is not None:
            xcat = arg.res().substitute(subs)
            yield FunctionalCategory(xcat, FunctionalCategory(xcat, function, arg.dir()), -arg.dir())

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'T'

def forwardTConstraint(left, right):
    if False:
        for i in range(10):
            print('nop')
    arg = innermostFunction(right)
    return arg.dir().is_backward() and arg.res().is_primitive()

def backwardTConstraint(left, right):
    if False:
        while True:
            i = 10
    arg = innermostFunction(left)
    return arg.dir().is_forward() and arg.res().is_primitive()
ForwardT = ForwardCombinator(UndirectedTypeRaise(), forwardTConstraint)
BackwardT = BackwardCombinator(UndirectedTypeRaise(), backwardTConstraint)