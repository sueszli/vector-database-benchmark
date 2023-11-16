import sys
import collections
import nose.case
import nose.inspector
import nose.loader
import nose.suite
import nose.plugins.attrib
if sys.version_info >= (3, 10) and (not hasattr(collections, 'Callable')):
    nose.case.collections = collections.abc
    nose.inspector.collections = collections.abc
    nose.loader.collections = collections.abc
    nose.suite.collections = collections.abc
    nose.plugins.attrib.collections = collections.abc
import nose.tools as tools
import re
import fnmatch

def glob_to_regex(glob):
    if False:
        return 10
    if not isinstance(glob, str):
        raise ValueError('Glob pattern must be a string')
    pattern = fnmatch.translate(glob)
    if pattern[-2:] == '\\Z':
        pattern = pattern[:-2]
    return pattern

def get_pattern(glob=None, regex=None, match_case=None):
    if False:
        return 10
    assert glob is not None or regex is not None
    if glob is not None and regex is not None:
        raise ValueError('You should specify at most one of `glob` and `regex` parameters but not both')
    if glob is not None:
        pattern = glob_to_regex(glob)
    else:
        if match_case is not None and (not isinstance(regex, str)):
            raise ValueError('Regex must be a string if `match_case` is specified when calling assert_raises_pattern')
        pattern = regex
    if isinstance(pattern, str) and (not match_case):
        pattern = re.compile(pattern, re.IGNORECASE)
    return pattern

def assert_raises(exception, *args, glob=None, regex=None, match_case=None, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Wrapper combining `nose.tools.assert_raises` and `nose.tools.assert_raises_regex`.\n    Specify ``regex=pattern`` or ``glob=pattern`` to check error message of expected exception\n    against the pattern.\n    Value for `glob` must be a string, `regex` can be either a literal or compiled regex pattern.\n    By default, the check will ignore case, if called with `glob` or a literal for `regex`.\n    To enforce case sensitive check pass ``match_case=True``.\n    Don't specify `match_case` if passing already compiled regex pattern.\n    "
    if glob is None and regex is None:
        return tools.assert_raises(exception, *args, **kwargs)
    pattern = get_pattern(glob, regex, match_case)
    return tools.assert_raises_regex(exception, pattern, *args, **kwargs)

def assert_warns(exception=Warning, *args, glob=None, regex=None, match_case=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    if glob is None and regex is None:
        return tools.assert_warns(exception, *args, **kwargs)
    pattern = get_pattern(glob, regex, match_case)
    return tools.assert_warns_regex(exception, pattern, *args, **kwargs)

def raises(exception, glob=None, regex=None, match_case=None):
    if False:
        return 10
    '\n    To assert that the test case raises Exception with the message matching given glob pattern\n        @raises(Exception, "abc * def")\n        def test():\n            raise Exception("It\'s: abc 42 def, and has some suffix.")\n\n    To assert that the test case raises Exception with the message matching given regex pattern\n        @raises(Exception, regex="abc[0-9]{2}def")\n        def test():\n            raise Exception("It\'s: abc42def, and has some suffix too.")\n\n    You can also use it like regular nose.raises\n        @raises(Exception)\n        def test():\n            raise Exception("This message is not checked")\n\n    By default, the check is not case-sensitive, to change that pass `match_case`=True.\n\n    You can pass a tuple of exception classes to assert that the raised exception is\n    an instance of at least one of the classes.\n    '

    def decorator(func):
        if False:
            i = 10
            return i + 15

        def new_func(*args, **kwargs):
            if False:
                while True:
                    i = 10
            with assert_raises(exception, glob=glob, regex=regex, match_case=match_case):
                return func(*args, **kwargs)
        return tools.make_decorator(func)(new_func)
    return decorator