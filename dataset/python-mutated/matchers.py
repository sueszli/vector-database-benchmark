"""
This module provides the step matchers functionality that matches a
step definition (as text) with step-functions that implement this step.
"""
from __future__ import absolute_import, print_function, with_statement
import copy
import inspect
import re
import warnings
import six
import parse
from parse_type import cfparse
from behave._types import ChainedExceptionUtil, ExceptionUtil
from behave.exception import NotSupportedWarning, ResourceExistsError
from behave.model_core import Argument, FileLocation, Replayable

class StepParseError(ValueError):
    """Exception class, used when step matching fails before a step is run.
    This is normally the case when an error occurs during the type conversion
    of step parameters.
    """

    def __init__(self, text=None, exc_cause=None):
        if False:
            for i in range(10):
                print('nop')
        if not text and exc_cause:
            text = six.text_type(exc_cause)
        if exc_cause and six.PY2:
            cause_text = ExceptionUtil.describe(exc_cause, use_traceback=True, prefix='CAUSED-BY: ')
            text += u'\n' + cause_text
        ValueError.__init__(self, text)
        if exc_cause:
            ChainedExceptionUtil.set_cause(self, exc_cause)

class Match(Replayable):
    """An parameter-matched step name extracted from a *feature file*.

    .. attribute:: func

       The step function that this match will be applied to.

    .. attribute:: arguments

       A list of :class:`~behave.model_core.Argument` instances containing the
       matched parameters from the step name.
    """
    type = 'match'

    def __init__(self, func, arguments=None):
        if False:
            for i in range(10):
                print('nop')
        super(Match, self).__init__()
        self.func = func
        self.arguments = arguments
        self.location = None
        if func:
            self.location = self.make_location(func)

    def __repr__(self):
        if False:
            print('Hello World!')
        if self.func:
            func_name = self.func.__name__
        else:
            func_name = '<no function>'
        return '<Match %s, %s>' % (func_name, self.location)

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        if not isinstance(other, Match):
            return False
        return (self.func, self.location) == (other.func, other.location)

    def with_arguments(self, arguments):
        if False:
            print('Hello World!')
        match = copy.copy(self)
        match.arguments = arguments
        return match

    def run(self, context):
        if False:
            return 10
        args = []
        kwargs = {}
        for arg in self.arguments:
            if arg.name is not None:
                kwargs[arg.name] = arg.value
            else:
                args.append(arg.value)
        with context.use_with_user_mode():
            self.func(context, *args, **kwargs)

    @staticmethod
    def make_location(step_function):
        if False:
            for i in range(10):
                print('nop')
        'Extracts the location information from the step function and\n        builds a FileLocation object with (filename, line_number) info.\n\n        :param step_function: Function whose location should be determined.\n        :return: FileLocation object for step function.\n        '
        return FileLocation.for_function(step_function)

class NoMatch(Match):
    """Used for an "undefined step" when it can not be matched with a
    step definition.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        Match.__init__(self, func=None)
        self.func = None
        self.arguments = []
        self.location = None

class MatchWithError(Match):
    """Match class when error occur during step-matching

    REASON:
      * Type conversion error occured.
      * ...
    """

    def __init__(self, func, error):
        if False:
            return 10
        if not ExceptionUtil.has_traceback(error):
            ExceptionUtil.set_traceback(error)
        Match.__init__(self, func=func)
        self.stored_error = error

    def run(self, context):
        if False:
            return 10
        'Raises stored error from step matching phase (type conversion).'
        raise StepParseError(exc_cause=self.stored_error)

class Matcher(object):
    """Pull parameters out of step names.

    .. attribute:: pattern

       The match pattern attached to the step function.

    .. attribute:: func

       The step function the pattern is being attached to.
    """
    schema = u"@%s('%s')"

    @classmethod
    def register_type(cls, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Register one (or more) user-defined types used for matching types\n        in step patterns of this matcher.\n        '
        raise NotImplementedError()

    @classmethod
    def clear_registered_types(cls):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    def __init__(self, func, pattern, step_type=None):
        if False:
            return 10
        self.func = func
        self.pattern = pattern
        self.step_type = step_type
        self._location = None

    @property
    def string(self):
        if False:
            print('Hello World!')
        warnings.warn("deprecated: Use 'pattern' instead", DeprecationWarning)
        return self.pattern

    @property
    def location(self):
        if False:
            for i in range(10):
                print('nop')
        if self._location is None:
            self._location = Match.make_location(self.func)
        return self._location

    @property
    def regex_pattern(self):
        if False:
            i = 10
            return i + 15
        'Return the used textual regex pattern.'
        return self.pattern

    def describe(self, schema=None):
        if False:
            while True:
                i = 10
        'Provide a textual description of the step function/matcher object.\n\n        :param schema:  Text schema to use.\n        :return: Textual description of this step definition (matcher).\n        '
        step_type = self.step_type or 'step'
        if not schema:
            schema = self.schema
        return schema % (step_type, self.pattern)

    def check_match(self, step):
        if False:
            for i in range(10):
                print('nop')
        'Match me against the "step" name supplied.\n\n        Return None, if I don\'t match otherwise return a list of matches as\n        :class:`~behave.model_core.Argument` instances.\n\n        The return value from this function will be converted into a\n        :class:`~behave.matchers.Match` instance by *behave*.\n        '
        raise NotImplementedError

    def match(self, step):
        if False:
            print('Hello World!')
        try:
            result = self.check_match(step)
        except Exception as e:
            return MatchWithError(self.func, e)
        if result is None:
            return None
        return Match(self.func, result)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return u'<%s: %r>' % (self.__class__.__name__, self.pattern)

class ParseMatcher(Matcher):
    """Uses :class:`~parse.Parser` class to be able to use simpler
    parse expressions compared to normal regular expressions.
    """
    custom_types = {}
    parser_class = parse.Parser

    @classmethod
    def register_type(cls, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Register one (or more) user-defined types used for matching types\n        in step patterns of this matcher.\n\n        A type converter should follow :pypi:`parse` module rules.\n        In general, a type converter is a function that converts text (as string)\n        into a value-type (type converted value).\n\n        EXAMPLE:\n\n        .. code-block:: python\n\n            from behave import register_type, given\n            import parse\n\n\n            # -- TYPE CONVERTER: For a simple, positive integer number.\n            @parse.with_pattern(r"\\d+")\n            def parse_number(text):\n                return int(text)\n\n            # -- REGISTER TYPE-CONVERTER: With behave\n            register_type(Number=parse_number)\n            # ALTERNATIVE:\n            current_step_matcher = use_step_matcher("parse")\n            current_step_matcher.register_type(Number=parse_number)\n\n            # -- STEP DEFINITIONS: Use type converter.\n            @given(\'{amount:Number} vehicles\')\n            def step_impl(context, amount):\n                assert isinstance(amount, int)\n        '
        cls.custom_types.update(**kwargs)

    @classmethod
    def clear_registered_types(cls):
        if False:
            print('Hello World!')
        cls.custom_types.clear()

    def __init__(self, func, pattern, step_type=None):
        if False:
            print('Hello World!')
        super(ParseMatcher, self).__init__(func, pattern, step_type)
        self.parser = self.parser_class(pattern, self.custom_types)

    @property
    def regex_pattern(self):
        if False:
            return 10
        return self.parser._expression

    def check_match(self, step):
        if False:
            i = 10
            return i + 15
        result = self.parser.parse(step)
        if not result:
            return None
        args = []
        for (index, value) in enumerate(result.fixed):
            (start, end) = result.spans[index]
            args.append(Argument(start, end, step[start:end], value))
        for (name, value) in result.named.items():
            (start, end) = result.spans[name]
            args.append(Argument(start, end, step[start:end], value, name))
        args.sort(key=lambda x: x.start)
        return args

class CFParseMatcher(ParseMatcher):
    """Uses :class:`~parse_type.cfparse.Parser` instead of "parse.Parser".
    Provides support for automatic generation of type variants
    for fields with CardinalityField part.
    """
    parser_class = cfparse.Parser

class RegexMatcher(Matcher):

    @classmethod
    def register_type(cls, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Register one (or more) user-defined types used for matching types\n        in step patterns of this matcher.\n\n        NOTE:\n        This functionality is not supported for :class:`RegexMatcher` classes.\n        '
        raise NotSupportedWarning('%s.register_type' % cls.__name__)

    @classmethod
    def clear_registered_types(cls):
        if False:
            while True:
                i = 10
        pass

    def __init__(self, func, pattern, step_type=None):
        if False:
            print('Hello World!')
        super(RegexMatcher, self).__init__(func, pattern, step_type)
        self.regex = re.compile(self.pattern)

    def check_match(self, step):
        if False:
            print('Hello World!')
        m = self.regex.match(step)
        if not m:
            return None
        groupindex = dict(((y, x) for (x, y) in self.regex.groupindex.items()))
        args = []
        for (index, group) in enumerate(m.groups()):
            index += 1
            name = groupindex.get(index, None)
            args.append(Argument(m.start(index), m.end(index), group, group, name))
        return args

class SimplifiedRegexMatcher(RegexMatcher):
    """
    Simplified regular expression step-matcher that automatically adds
    start-of-line/end-of-line matcher symbols to string:

    .. code-block:: python

        @when(u'a step passes')     # re.pattern = "^a step passes$"
        def step_impl(context): pass
    """

    def __init__(self, func, pattern, step_type=None):
        if False:
            while True:
                i = 10
        assert not (pattern.startswith('^') or pattern.endswith('$')), 'Regular expression should not use begin/end-markers: ' + pattern
        expression = '^%s$' % pattern
        super(SimplifiedRegexMatcher, self).__init__(func, expression, step_type)
        self.pattern = pattern

class CucumberRegexMatcher(RegexMatcher):
    """
    Compatible to (old) Cucumber style regular expressions.
    Text must contain start-of-line/end-of-line matcher symbols to string:

    .. code-block:: python

        @when(u'^a step passes$')   # re.pattern = "^a step passes$"
        def step_impl(context): pass
    """

class StepMatcherFactory(object):
    """
    This class provides functionality for the public API of step-matchers.

    It allows to  change the step-matcher class in use
    while parsing step definitions.
    This allows to use multiple step-matcher classes:

    * in the same steps module
    * in different step modules

    There are several step-matcher classes available in **behave**:

    * **parse** (the default, based on: :pypi:`parse`):
    * **cfparse** (extends: :pypi:`parse`, requires: :pypi:`parse_type`)
    * **re** (using regular expressions)

    You may `define your own step-matcher class`_.

    .. _`define your own step-matcher class`: api.html#step-parameters

    parse
    ------

    Provides a simple parser that replaces regular expressions for
    step parameters with a readable syntax like ``{param:Type}``.
    The syntax is inspired by the Python builtin ``string.format()`` function.
    Step parameters must use the named fields syntax of :pypi:`parse`
    in step definitions. The named fields are extracted,
    optionally type converted and then used as step function arguments.

    Supports type conversions by using type converters
    (see :func:`~behave.register_type()`).

    cfparse
    -------

    Provides an extended parser with "Cardinality Field" (CF) support.
    Automatically creates missing type converters for related cardinality
    as long as a type converter for cardinality=1 is provided.
    Supports parse expressions like:

    * ``{values:Type+}`` (cardinality=1..N, many)
    * ``{values:Type*}`` (cardinality=0..N, many0)
    * ``{value:Type?}``  (cardinality=0..1, optional)

    Supports type conversions (as above).

    re (regex based parser)
    -----------------------

    This uses full regular expressions to parse the clause text. You will
    need to use named groups "(?P<name>...)" to define the variables pulled
    from the text and passed to your ``step()`` function.

    Type conversion is **not supported**.
    A step function writer may implement type conversion
    inside the step function (implementation).
    """
    MATCHER_MAPPING = {'parse': ParseMatcher, 'cfparse': CFParseMatcher, 're': SimplifiedRegexMatcher, 're0': CucumberRegexMatcher}
    DEFAULT_MATCHER_NAME = 'parse'

    def __init__(self, matcher_mapping=None, default_matcher_name=None):
        if False:
            i = 10
            return i + 15
        if matcher_mapping is None:
            matcher_mapping = self.MATCHER_MAPPING.copy()
        if default_matcher_name is None:
            default_matcher_name = self.DEFAULT_MATCHER_NAME
        self.matcher_mapping = matcher_mapping
        self.initial_matcher_name = default_matcher_name
        self.default_matcher_name = default_matcher_name
        self.default_matcher = matcher_mapping[default_matcher_name]
        self._current_matcher = self.default_matcher
        assert self.default_matcher in self.matcher_mapping.values()

    def reset(self):
        if False:
            while True:
                i = 10
        self.use_default_step_matcher(self.initial_matcher_name)
        self.clear_registered_types()

    @property
    def current_matcher(self):
        if False:
            print('Hello World!')
        return self._current_matcher

    def register_type(self, **kwargs):
        if False:
            return 10
        '\n        Registers one (or more) custom type that will be available\n        by some matcher classes, like the :class:`ParseMatcher` and its\n        derived classes, for type conversion during step matching.\n\n        Converters should be supplied as ``name=callable`` arguments (or as dict).\n        A type converter should follow the rules of its :class:`Matcher` class.\n        '
        self.current_matcher.register_type(**kwargs)

    def clear_registered_types(self):
        if False:
            while True:
                i = 10
        for step_matcher_class in self.matcher_mapping.values():
            step_matcher_class.clear_registered_types()

    def register_step_matcher_class(self, name, step_matcher_class, override=False):
        if False:
            i = 10
            return i + 15
        'Register a new step-matcher class to use.\n\n        :param name:  Name of the step-matcher to use.\n        :param step_matcher_class:  Step-matcher class.\n        :param override:  Use ``True`` to override any existing step-matcher class.\n        '
        assert inspect.isclass(step_matcher_class)
        assert issubclass(step_matcher_class, Matcher), 'OOPS: %r' % step_matcher_class
        known_class = self.matcher_mapping.get(name, None)
        if not override and known_class is not None and (known_class is not step_matcher_class):
            message = 'ALREADY REGISTERED: {name}={class_name}'.format(name=name, class_name=known_class.__name__)
            raise ResourceExistsError(message)
        self.matcher_mapping[name] = step_matcher_class

    def use_step_matcher(self, name):
        if False:
            print('Hello World!')
        '\n        Changes the step-matcher class to use while parsing step definitions.\n        This allows to use multiple step-matcher classes:\n\n        * in the same steps module\n        * in different step modules\n\n        There are several step-matcher classes available in **behave**:\n\n        * **parse** (the default, based on: :pypi:`parse`):\n        * **cfparse** (extends: :pypi:`parse`, requires: :pypi:`parse_type`)\n        * **re** (using regular expressions)\n\n        :param name:  Name of the step-matcher class.\n        :return: Current step-matcher class that is now in use.\n        '
        self._current_matcher = self.matcher_mapping[name]
        return self._current_matcher

    def use_default_step_matcher(self, name=None):
        if False:
            i = 10
            return i + 15
        'Use the default step-matcher.\n        If a :param:`name` is provided, the default step-matcher is defined.\n\n        :param name:    Optional, use it to specify the default step-matcher.\n        :return: Current step-matcher class (or object).\n        '
        if name:
            self.default_matcher = self.matcher_mapping[name]
            self.default_matcher_name = name
        self._current_matcher = self.default_matcher
        return self._current_matcher

    def use_current_step_matcher_as_default(self):
        if False:
            i = 10
            return i + 15
        self.default_matcher = self._current_matcher

    def make_matcher(self, func, step_text, step_type=None):
        if False:
            print('Hello World!')
        return self.current_matcher(func, step_text, step_type=step_type)
_the_matcher_factory = StepMatcherFactory()

def get_matcher_factory():
    if False:
        return 10
    return _the_matcher_factory

def make_matcher(func, step_text, step_type=None):
    if False:
        return 10
    return _the_matcher_factory.make_matcher(func, step_text, step_type=step_type)

def use_current_step_matcher_as_default():
    if False:
        while True:
            i = 10
    return _the_matcher_factory.use_current_step_matcher_as_default()

def use_step_matcher(name):
    if False:
        i = 10
        return i + 15
    return _the_matcher_factory.use_step_matcher(name)

def use_default_step_matcher(name=None):
    if False:
        print('Hello World!')
    return _the_matcher_factory.use_default_step_matcher(name=name)

def register_type(**kwargs):
    if False:
        while True:
            i = 10
    _the_matcher_factory.register_type(**kwargs)
register_type.__doc__ = StepMatcherFactory.register_type.__doc__
use_step_matcher.__doc__ = StepMatcherFactory.use_step_matcher.__doc__
use_default_step_matcher.__doc__ = StepMatcherFactory.use_default_step_matcher.__doc__

def register_step_matcher_class(name, step_matcher_class, override=False):
    if False:
        print('Hello World!')
    _the_matcher_factory.register_step_matcher_class(name, step_matcher_class, override=override)
register_step_matcher_class.__doc__ = StepMatcherFactory.register_step_matcher_class.__doc__