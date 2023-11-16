"""
This module provides the abstract base classes and core concepts
for the model elements in behave.
"""
import os.path
import sys
import six
from behave.capture import Captured
from behave.textutil import text as _text
from enum import Enum
PLATFORM_WIN = sys.platform.startswith('win')

def posixpath_normalize(path):
    if False:
        while True:
            i = 10
    return path.replace('\\', '/')

class Status(Enum):
    """Provides the (test-run) status of a model element.
    Features and Scenarios use: untested, skipped, passed, failed.
    Steps may use all enum-values.

    Enum values:
    * untested (initial state):

        Defines the initial state before a test-run.
        Sometimes used to indicate that the model element was not executed
        during a test run.

    * skipped:

        A model element is skipped because it should not run.
        This is caused by filtering mechanisms, like tags, active-tags,
        file-location arg, select-by-name, etc.

    * passed: A model element was executed and passed (without failures).
    * failed: Failures occurred while executing it.
    * undefined: Used for undefined-steps (no step implementation was found).
    * executing: Marks the steps during execution (used in a formatter)

    .. versionadded:: 1.2.6
        Superceeds string-based status values.
    """
    untested = 0
    skipped = 1
    passed = 2
    failed = 3
    undefined = 4
    executing = 5

    def __eq__(self, other):
        if False:
            print('Hello World!')
        'Comparison operator equals-to other value.\n        Supports other enum-values and string (for backward compatibility).\n\n        EXAMPLES::\n\n            status = Status.passed\n            assert status == Status.passed\n            assert status == "passed"\n            assert status != "failed"\n\n        :param other:   Other value to compare (enum-value, string).\n        :return: True, if both values are equal. False, otherwise.\n        '
        if isinstance(other, six.string_types):
            return self.name == other
        return super(Status, self).__eq__(other)

    @classmethod
    def from_name(cls, name):
        if False:
            i = 10
            return i + 15
        'Select enumeration value by using its name.\n\n        :param name:    Name as key to the enum value (as string).\n        :return: Enum value (instance)\n        :raises: LookupError, if status name is unknown.\n        '
        enum_value = cls.__members__.get(name, None)
        if enum_value is None:
            known_names = ', '.join(cls.__members__.keys())
            raise LookupError('%s (expected: %s)' % (name, known_names))
        return enum_value

class Argument(object):
    """An argument found in a *feature file* step name.

    The attributes are:

    .. attribute:: original

       The actual text matched in the step name.

    .. attribute:: value

       The potentially type-converted value of the argument.

    .. attribute:: name

       The name of the argument.
       This will be None if the parameter is anonymous.

    .. attribute:: start

       The start index in the step name of the argument. Used for display.

    .. attribute:: end

       The end index in the step name of the argument. Used for display.
    """

    def __init__(self, start, end, original, value, name=None):
        if False:
            print('Hello World!')
        self.start = start
        self.end = end
        self.original = original
        self.value = value
        self.name = name

class FileLocation(object):
    """
    Provides a value object for file location objects.
    A file location consists of:

      * filename
      * line (number), optional

    LOCATION SCHEMA:
      * "{filename}:{line}" or
      * "{filename}" (if line number is not present)
    """
    __pychecker__ = 'missingattrs=line'

    def __init__(self, filename, line=None):
        if False:
            i = 10
            return i + 15
        if PLATFORM_WIN:
            filename = posixpath_normalize(filename)
        self.filename = filename
        self.line = line

    def get(self):
        if False:
            while True:
                i = 10
        return self.filename

    def abspath(self):
        if False:
            while True:
                i = 10
        return os.path.abspath(self.filename)

    def basename(self):
        if False:
            return 10
        return os.path.basename(self.filename)

    def dirname(self):
        if False:
            print('Hello World!')
        return os.path.dirname(self.filename)

    def relpath(self, start=os.curdir):
        if False:
            print('Hello World!')
        'Compute relative path for start to filename.\n\n        :param start: Base path or start directory (default=current dir).\n        :return: Relative path from start to filename\n        '
        return os.path.relpath(self.filename, start)

    def exists(self):
        if False:
            i = 10
            return i + 15
        return os.path.exists(self.filename)

    def _line_lessthan(self, other_line):
        if False:
            for i in range(10):
                print('nop')
        if self.line is None:
            return other_line is not None
        elif other_line is None:
            return False
        else:
            return self.line < other_line

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        if isinstance(other, FileLocation):
            return self.filename == other.filename and self.line == other.line
        elif isinstance(other, six.string_types):
            return self.filename == other
        else:
            raise TypeError('Cannot compare FileLocation with %s:%s' % (type(other), other))

    def __ne__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return not self.__eq__(other)

    def __lt__(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, FileLocation):
            if self.filename < other.filename:
                return True
            elif self.filename > other.filename:
                return False
            else:
                assert self.filename == other.filename
                return self._line_lessthan(other.line)
        elif isinstance(other, six.string_types):
            return self.filename < other
        else:
            raise TypeError('Cannot compare FileLocation with %s:%s' % (type(other), other))

    def __le__(self, other):
        if False:
            while True:
                i = 10
        return other >= self

    def __gt__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other, FileLocation):
            return other < self
        else:
            return self.filename > other

    def __ge__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return not self.__lt__(other)

    def __repr__(self):
        if False:
            return 10
        return u'<FileLocation: filename="%s", line=%s>' % (self.filename, self.line)

    def __str__(self):
        if False:
            return 10
        filename = self.filename
        if isinstance(filename, six.binary_type):
            filename = _text(filename, 'utf-8')
        if self.line is None:
            return filename
        return u'%s:%d' % (filename, self.line)
    if six.PY2:
        __unicode__ = __str__
        __str__ = lambda self: self.__unicode__().encode('utf-8')

    @classmethod
    def for_function(cls, func, curdir=None):
        if False:
            print('Hello World!')
        'Extracts the location information from the function and builds\n        the location string (schema: "{source_filename}:{line_number}").\n\n        :param func: Function whose location should be determined.\n        :return: FileLocation object\n        '
        func = unwrap_function(func)
        function_code = six.get_function_code(func)
        filename = function_code.co_filename
        line_number = function_code.co_firstlineno
        curdir = curdir or os.getcwd()
        try:
            filename = os.path.relpath(filename, curdir)
        except ValueError:
            pass
        return cls(filename, line_number)

class BasicStatement(object):

    def __init__(self, filename, line, keyword, name):
        if False:
            while True:
                i = 10
        filename = filename or '<string>'
        filename = os.path.relpath(filename, os.getcwd())
        self.location = FileLocation(filename, line)
        assert isinstance(keyword, six.text_type)
        assert isinstance(name, six.text_type)
        self.keyword = keyword
        self.name = name
        self.captured = Captured()
        self.exception = None
        self.exc_traceback = None
        self.error_message = None

    @property
    def filename(self):
        if False:
            while True:
                i = 10
        return self.location.filename

    @property
    def line(self):
        if False:
            for i in range(10):
                print('nop')
        return self.location.line

    def reset(self):
        if False:
            i = 10
            return i + 15
        self.captured.reset()
        self.exception = None
        self.exc_traceback = None
        self.error_message = None

    def store_exception_context(self, exception):
        if False:
            for i in range(10):
                print('nop')
        self.exception = exception
        self.exc_traceback = sys.exc_info()[2]

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return id(self)

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        return (self.keyword, self.name) == (other.keyword, other.name)

    def __lt__(self, other):
        if False:
            return 10
        return (self.keyword, self.name) < (other.keyword, other.name)

    def __ne__(self, other):
        if False:
            while True:
                i = 10
        return not self.__eq__(other)

    def __le__(self, other):
        if False:
            print('Hello World!')
        return other >= self

    def __gt__(self, other):
        if False:
            while True:
                i = 10
        assert isinstance(other, BasicStatement)
        return other < self

    def __ge__(self, other):
        if False:
            return 10
        return not self < other

class TagStatement(BasicStatement):

    def __init__(self, filename, line, keyword, name, tags):
        if False:
            while True:
                i = 10
        if tags is None:
            tags = []
        super(TagStatement, self).__init__(filename, line, keyword, name)
        self.tags = tags

    def should_run_with_tags(self, tag_expression):
        if False:
            for i in range(10):
                print('nop')
        'Determines if statement should run when the tag expression is used.\n\n        :param tag_expression:  Runner/config environment tags to use.\n        :return: True, if examples should run. False, otherwise (skip it).\n        '
        return tag_expression.check(self.tags)

class TagAndStatusStatement(BasicStatement):
    """Base class for statements with:

    * tags (as: taggable statement)
    * status (has a result after a test run)
    """
    final_status = (Status.passed, Status.failed, Status.skipped)

    def __init__(self, filename, line, keyword, name, tags, parent=None):
        if False:
            return 10
        super(TagAndStatusStatement, self).__init__(filename, line, keyword, name)
        self.parent = parent
        self.tags = tags
        self.should_skip = False
        self.skip_reason = None
        self._cached_status = Status.untested

    @property
    def effective_tags(self):
        if False:
            while True:
                i = 10
        'Compute effective tags of this entity.\n        This is includes the own tags and the inherited tags from the parents.\n\n        :return: Set of effective tags\n\n        .. versionadded:: 1.2.7\n        '
        tags = set(self.tags)
        if self.parent:
            inherited_tags = self.parent.effective_tags
            tags.update(inherited_tags)
        return tags

    def should_run_with_tags(self, tag_expression):
        if False:
            i = 10
            return i + 15
        'Determines if statement should run when the tag expression is used.\n\n        :param tag_expression:  Runner/config environment tags to use.\n        :return: True, if this statement should run. False, otherwise (skip it).\n        '
        return tag_expression.check(self.effective_tags)

    @property
    def status(self):
        if False:
            print('Hello World!')
        if self._cached_status not in self.final_status:
            self._cached_status = self.compute_status()
        return self._cached_status

    def set_status(self, value):
        if False:
            print('Hello World!')
        if isinstance(value, six.string_types):
            value = Status.from_name(value)
        self._cached_status = value

    def clear_status(self):
        if False:
            return 10
        self._cached_status = Status.untested

    def reset(self):
        if False:
            i = 10
            return i + 15
        self.should_skip = False
        self.skip_reason = None
        self.clear_status()

    def compute_status(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

class Replayable(object):
    type = None

    def replay(self, formatter):
        if False:
            return 10
        getattr(formatter, self.type)(self)

def unwrap_function(func, max_depth=10):
    if False:
        print('Hello World!')
    'Unwraps a function that is wrapped with :func:`functools.partial()`'
    iteration = 0
    wrapped = getattr(func, '__wrapped__', None)
    while wrapped and iteration < max_depth:
        func = wrapped
        wrapped = getattr(func, '__wrapped__', None)
        iteration += 1
    return func