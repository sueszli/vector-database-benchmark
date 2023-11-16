from collections import namedtuple
from itertools import chain
from six.moves import map, zip_longest
from zipline.errors import ZiplineError
from zipline.utils.compat import getargspec
Argspec = namedtuple('Argspec', ['args', 'starargs', 'kwargs'])

def singleton(cls):
    if False:
        i = 10
        return i + 15
    instances = {}

    def getinstance():
        if False:
            print('Hello World!')
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]
    return getinstance

@singleton
class Ignore(object):

    def __str__(self):
        if False:
            return 10
        return 'Argument.ignore'
    __repr__ = __str__

@singleton
class NoDefault(object):

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'Argument.no_default'
    __repr__ = __str__

@singleton
class AnyDefault(object):

    def __str__(self):
        if False:
            while True:
                i = 10
        return 'Argument.any_default'
    __repr__ = __str__

class Argument(namedtuple('Argument', ['name', 'default'])):
    """
    An argument to a function.
    Argument.no_default is a value representing no default to the argument.
    Argument.ignore is a value that says you should ignore the default value.
    """
    no_default = NoDefault()
    any_default = AnyDefault()
    ignore = Ignore()

    def __new__(cls, name=ignore, default=ignore):
        if False:
            return 10
        return super(Argument, cls).__new__(cls, name, default)

    def __str__(self):
        if False:
            while True:
                i = 10
        if self.has_no_default(self) or self.ignore_default(self):
            return str(self.name)
        else:
            return '='.join([str(self.name), str(self.default)])

    def __repr__(self):
        if False:
            while True:
                i = 10
        return 'Argument(%s, %s)' % (repr(self.name), repr(self.default))

    def _defaults_match(self, arg):
        if False:
            i = 10
            return i + 15
        return any(map(Argument.ignore_default, [self, arg])) or (self.default is Argument.any_default and arg.default is not Argument.no_default) or (arg.default is Argument.any_default and self.default is not Argument.no_default) or (self.default == arg.default)

    def _names_match(self, arg):
        if False:
            for i in range(10):
                print('nop')
        return self.name == arg.name or self.name is Argument.ignore or arg.name is Argument.ignore

    def matches(self, arg):
        if False:
            i = 10
            return i + 15
        return self._names_match(arg) and self._defaults_match(arg)
    __eq__ = matches

    @staticmethod
    def parse_argspec(callable_):
        if False:
            return 10
        '\n        Takes a callable and returns a tuple with the list of Argument objects,\n        the name of *args, and the name of **kwargs.\n        If *args or **kwargs is not present, it will be None.\n        This returns a namedtuple called Argspec that has three fields named:\n        args, starargs, and kwargs.\n        '
        (args, varargs, keywords, defaults) = getargspec(callable_)
        defaults = list(defaults or [])
        if getattr(callable_, '__self__', None) is not None:
            args = args[1:]
        first_default = len(args) - len(defaults)
        return Argspec([Argument(arg, Argument.no_default if n < first_default else defaults[n - first_default]) for (n, arg) in enumerate(args)], varargs, keywords)

    @staticmethod
    def has_no_default(arg):
        if False:
            print('Hello World!')
        return arg.default is Argument.no_default

    @staticmethod
    def ignore_default(arg):
        if False:
            return 10
        return arg.default is Argument.ignore

def _expect_extra(expected, present, exc_unexpected, exc_missing, exc_args):
    if False:
        for i in range(10):
            print('nop')
    '\n    Checks for the presence of an extra to the argument list. Raises expections\n    if this is unexpected or if it is missing and expected.\n    '
    if present:
        if not expected:
            raise exc_unexpected(*exc_args)
    elif expected and expected is not Argument.ignore:
        raise exc_missing(*exc_args)

def verify_callable_argspec(callable_, expected_args=Argument.ignore, expect_starargs=Argument.ignore, expect_kwargs=Argument.ignore):
    if False:
        return 10
    "\n    Checks the callable_ to make sure that it satisfies the given\n    expectations.\n    expected_args should be an iterable of Arguments in the order you expect to\n    receive them.\n    expect_starargs means that the function should or should not take a *args\n    param. expect_kwargs says the callable should or should not take  **kwargs\n    param.\n    If expected_args, expect_starargs, or expect_kwargs is Argument.ignore,\n    then the checks related to that argument will not occur.\n\n    Example usage:\n\n    callable_check(\n        f,\n        [Argument('a'), Argument('b', 1)],\n        expect_starargs=True,\n        expect_kwargs=Argument.ignore\n    )\n    "
    if not callable(callable_):
        raise NotCallable(callable_)
    expected_arg_list = list(expected_args if expected_args is not Argument.ignore else [])
    (args, starargs, kwargs) = Argument.parse_argspec(callable_)
    exc_args = (callable_, args, starargs, kwargs)
    _expect_extra(expect_starargs, starargs, UnexpectedStarargs, NoStarargs, exc_args)
    _expect_extra(expect_kwargs, kwargs, UnexpectedKwargs, NoKwargs, exc_args)
    if expected_args is Argument.ignore:
        return
    if len(args) < len(expected_arg_list):
        raise NotEnoughArguments(callable_, args, starargs, kwargs, [arg for arg in expected_arg_list if arg not in args])
    elif len(args) > len(expected_arg_list):
        raise TooManyArguments(callable_, args, starargs, kwargs)
    missing_arg = Argument(object(), object())
    for (expected, provided) in zip_longest(expected_arg_list, args, fillvalue=missing_arg):
        if not expected.matches(provided):
            raise MismatchedArguments(callable_, args, starargs, kwargs)

class BadCallable(TypeError, AssertionError, ZiplineError):
    """
    The given callable is not structured in the expected way.
    """
    _lambda_name = (lambda : None).__name__

    def __init__(self, callable_, args, starargs, kwargs):
        if False:
            return 10
        self.callable_ = callable_
        self.args = args
        self.starargs = starargs
        self.kwargsname = kwargs
        self.kwargs = {}

    def format_callable(self):
        if False:
            for i in range(10):
                print('nop')
        if self.callable_.__name__ == self._lambda_name:
            fmt = '%s %s'
            name = 'lambda'
        else:
            fmt = '%s(%s)'
            name = self.callable_.__name__
        return fmt % (name, ', '.join(chain((str(arg) for arg in self.args), ('*' + sa for sa in (self.starargs,) if sa is not None), ('**' + ka for ka in (self.kwargsname,) if ka is not None))))

    @property
    def msg(self):
        if False:
            i = 10
            return i + 15
        return str(self)

class NoStarargs(BadCallable):

    def __str__(self):
        if False:
            while True:
                i = 10
        return '%s does not allow for *args' % self.format_callable()

class UnexpectedStarargs(BadCallable):

    def __str__(self):
        if False:
            while True:
                i = 10
        return '%s should not allow for *args' % self.format_callable()

class NoKwargs(BadCallable):

    def __str__(self):
        if False:
            print('Hello World!')
        return '%s does not allow for **kwargs' % self.format_callable()

class UnexpectedKwargs(BadCallable):

    def __str__(self):
        if False:
            return 10
        return '%s should not allow for **kwargs' % self.format_callable()

class NotCallable(BadCallable):
    """
    The provided 'callable' is not actually a callable.
    """

    def __init__(self, callable_):
        if False:
            i = 10
            return i + 15
        self.callable_ = callable_

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return '%s is not callable' % self.format_callable()

    def format_callable(self):
        if False:
            i = 10
            return i + 15
        try:
            return self.callable_.__name__
        except AttributeError:
            return str(self.callable_)

class NotEnoughArguments(BadCallable):
    """
    The callback does not accept enough arguments.
    """

    def __init__(self, callable_, args, starargs, kwargs, missing_args):
        if False:
            i = 10
            return i + 15
        super(NotEnoughArguments, self).__init__(callable_, args, starargs, kwargs)
        self.missing_args = missing_args

    def __str__(self):
        if False:
            i = 10
            return i + 15
        missing_args = list(map(str, self.missing_args))
        return '%s is missing argument%s: %s' % (self.format_callable(), 's' if len(missing_args) > 1 else '', ', '.join(missing_args))

class TooManyArguments(BadCallable):
    """
    The callback cannot be called by passing the expected number of arguments.
    """

    def __str__(self):
        if False:
            return 10
        return '%s accepts too many arguments' % self.format_callable()

class MismatchedArguments(BadCallable):
    """
    The argument lists are of the same lengths, but not in the correct order.
    """

    def __str__(self):
        if False:
            while True:
                i = 10
        return '%s accepts mismatched parameters' % self.format_callable()