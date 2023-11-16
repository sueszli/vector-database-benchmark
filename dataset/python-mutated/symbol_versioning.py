"""Symbol versioning

The methods here allow for api symbol versioning.
"""
from __future__ import absolute_import
__all__ = ['deprecated_function', 'deprecated_in', 'deprecated_list', 'deprecated_method', 'DEPRECATED_PARAMETER', 'deprecated_passed', 'set_warning_method', 'warn']
import warnings
from warnings import warn
import bzrlib
DEPRECATED_PARAMETER = 'A deprecated parameter marker.'

def deprecated_in(version_tuple):
    if False:
        return 10
    "Generate a message that something was deprecated in a release.\n\n    >>> deprecated_in((1, 4, 0))\n    '%s was deprecated in version 1.4.0.'\n    "
    return '%%s was deprecated in version %s.' % bzrlib._format_version_tuple(version_tuple)

def set_warning_method(method):
    if False:
        for i in range(10):
            print('nop')
    'Set the warning method to be used by this module.\n\n    It should take a message and a warning category as warnings.warn does.\n    '
    global warn
    warn = method

def deprecation_string(a_callable, deprecation_version):
    if False:
        return 10
    'Generate an automatic deprecation string for a_callable.\n\n    :param a_callable: The callable to substitute into deprecation_version.\n    :param deprecation_version: A deprecation format warning string. This should\n        have a single %s operator in it. a_callable will be turned into a nice\n        python symbol and then substituted into deprecation_version.\n    '
    if getattr(a_callable, 'im_class', None) is None:
        symbol = '%s.%s' % (a_callable.__module__, a_callable.__name__)
    else:
        symbol = '%s.%s.%s' % (a_callable.im_class.__module__, a_callable.im_class.__name__, a_callable.__name__)
    return deprecation_version % symbol

def deprecated_function(deprecation_version):
    if False:
        while True:
            i = 10
    'Decorate a function so that use of it will trigger a warning.'

    def function_decorator(callable):
        if False:
            return 10
        'This is the function python calls to perform the decoration.'

        def decorated_function(*args, **kwargs):
            if False:
                while True:
                    i = 10
            'This is the decorated function.'
            from bzrlib import trace
            trace.mutter_callsite(4, 'Deprecated function called')
            warn(deprecation_string(callable, deprecation_version), DeprecationWarning, stacklevel=2)
            return callable(*args, **kwargs)
        _populate_decorated(callable, deprecation_version, 'function', decorated_function)
        return decorated_function
    return function_decorator

def deprecated_method(deprecation_version):
    if False:
        i = 10
        return i + 15
    'Decorate a method so that use of it will trigger a warning.\n\n    To deprecate a static or class method, use\n\n        @staticmethod\n        @deprecated_function\n        def ...\n\n    To deprecate an entire class, decorate __init__.\n    '

    def method_decorator(callable):
        if False:
            while True:
                i = 10
        'This is the function python calls to perform the decoration.'

        def decorated_method(self, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            'This is the decorated method.'
            from bzrlib import trace
            if callable.__name__ == '__init__':
                symbol = '%s.%s' % (self.__class__.__module__, self.__class__.__name__)
            else:
                symbol = '%s.%s.%s' % (self.__class__.__module__, self.__class__.__name__, callable.__name__)
            trace.mutter_callsite(4, 'Deprecated method called')
            warn(deprecation_version % symbol, DeprecationWarning, stacklevel=2)
            return callable(self, *args, **kwargs)
        _populate_decorated(callable, deprecation_version, 'method', decorated_method)
        return decorated_method
    return method_decorator

def deprecated_passed(parameter_value):
    if False:
        print('Hello World!')
    'Return True if parameter_value was used.'
    return not parameter_value is DEPRECATED_PARAMETER

def _decorate_docstring(callable, deprecation_version, label, decorated_callable):
    if False:
        for i in range(10):
            print('nop')
    if callable.__doc__:
        docstring_lines = callable.__doc__.split('\n')
    else:
        docstring_lines = []
    if len(docstring_lines) == 0:
        decorated_callable.__doc__ = deprecation_version % ('This ' + label)
    elif len(docstring_lines) == 1:
        decorated_callable.__doc__ = callable.__doc__ + '\n' + '\n' + deprecation_version % ('This ' + label) + '\n'
    else:
        spaces = len(docstring_lines[-1])
        new_doc = callable.__doc__
        new_doc += '\n' + ' ' * spaces
        new_doc += deprecation_version % ('This ' + label)
        new_doc += '\n' + ' ' * spaces
        decorated_callable.__doc__ = new_doc

def _populate_decorated(callable, deprecation_version, label, decorated_callable):
    if False:
        while True:
            i = 10
    'Populate attributes like __name__ and __doc__ on the decorated callable.\n    '
    _decorate_docstring(callable, deprecation_version, label, decorated_callable)
    decorated_callable.__module__ = callable.__module__
    decorated_callable.__name__ = callable.__name__
    decorated_callable.is_deprecated = True

def _dict_deprecation_wrapper(wrapped_method):
    if False:
        print('Hello World!')
    'Returns a closure that emits a warning and calls the superclass'

    def cb(dep_dict, *args, **kwargs):
        if False:
            print('Hello World!')
        msg = 'access to %s' % (dep_dict._variable_name,)
        msg = dep_dict._deprecation_version % (msg,)
        if dep_dict._advice:
            msg += ' ' + dep_dict._advice
        warn(msg, DeprecationWarning, stacklevel=2)
        return wrapped_method(dep_dict, *args, **kwargs)
    return cb

class DeprecatedDict(dict):
    """A dictionary that complains when read or written."""
    is_deprecated = True

    def __init__(self, deprecation_version, variable_name, initial_value, advice):
        if False:
            for i in range(10):
                print('nop')
        'Create a dict that warns when read or modified.\n\n        :param deprecation_version: string for the warning format to raise,\n            typically from deprecated_in()\n        :param initial_value: The contents of the dict\n        :param variable_name: This allows better warnings to be printed\n        :param advice: String of advice on what callers should do instead\n            of using this variable.\n        '
        self._deprecation_version = deprecation_version
        self._variable_name = variable_name
        self._advice = advice
        dict.__init__(self, initial_value)
    __len__ = _dict_deprecation_wrapper(dict.__len__)
    __getitem__ = _dict_deprecation_wrapper(dict.__getitem__)
    __setitem__ = _dict_deprecation_wrapper(dict.__setitem__)
    __delitem__ = _dict_deprecation_wrapper(dict.__delitem__)
    keys = _dict_deprecation_wrapper(dict.keys)
    __contains__ = _dict_deprecation_wrapper(dict.__contains__)

def deprecated_list(deprecation_version, variable_name, initial_value, extra=None):
    if False:
        print('Hello World!')
    'Create a list that warns when modified\n\n    :param deprecation_version: string for the warning format to raise,\n        typically from deprecated_in()\n    :param initial_value: The contents of the list\n    :param variable_name: This allows better warnings to be printed\n    :param extra: Extra info to print when printing a warning\n    '
    subst_text = 'Modifying %s' % (variable_name,)
    msg = deprecation_version % (subst_text,)
    if extra:
        msg += ' ' + extra

    class _DeprecatedList(list):
        __doc__ = list.__doc__ + msg
        is_deprecated = True

        def _warn_deprecated(self, func, *args, **kwargs):
            if False:
                print('Hello World!')
            warn(msg, DeprecationWarning, stacklevel=3)
            return func(self, *args, **kwargs)

        def append(self, obj):
            if False:
                i = 10
                return i + 15
            'appending to %s is deprecated' % (variable_name,)
            return self._warn_deprecated(list.append, obj)

        def insert(self, index, obj):
            if False:
                while True:
                    i = 10
            'inserting to %s is deprecated' % (variable_name,)
            return self._warn_deprecated(list.insert, index, obj)

        def extend(self, iterable):
            if False:
                i = 10
                return i + 15
            'extending %s is deprecated' % (variable_name,)
            return self._warn_deprecated(list.extend, iterable)

        def remove(self, value):
            if False:
                return 10
            'removing from %s is deprecated' % (variable_name,)
            return self._warn_deprecated(list.remove, value)

        def pop(self, index=None):
            if False:
                for i in range(10):
                    print('nop')
            "pop'ing from %s is deprecated" % (variable_name,)
            if index:
                return self._warn_deprecated(list.pop, index)
            else:
                return self._warn_deprecated(list.pop)
    return _DeprecatedList(initial_value)

def _check_for_filter(error_only):
    if False:
        print('Hello World!')
    "Check if there is already a filter for deprecation warnings.\n\n    :param error_only: Only match an 'error' filter\n    :return: True if a filter is found, False otherwise\n    "
    for filter in warnings.filters:
        if issubclass(DeprecationWarning, filter[2]):
            if not error_only or filter[0] == 'error':
                return True
    return False

def _remove_filter_callable(filter):
    if False:
        i = 10
        return i + 15
    'Build and returns a callable removing filter from the warnings.\n\n    :param filter: The filter to remove (can be None).\n\n    :return: A callable that will remove filter from warnings.filters.\n    '

    def cleanup():
        if False:
            print('Hello World!')
        if filter:
            warnings.filters.remove(filter)
    return cleanup

def suppress_deprecation_warnings(override=True):
    if False:
        i = 10
        return i + 15
    "Call this function to suppress all deprecation warnings.\n\n    When this is a final release version, we don't want to annoy users with\n    lots of deprecation warnings. We only want the deprecation warnings when\n    running a dev or release candidate.\n\n    :param override: If True, always set the ignore, if False, only set the\n        ignore if there isn't already a filter.\n\n    :return: A callable to remove the new warnings this added.\n    "
    if not override and _check_for_filter(error_only=False):
        filter = None
    else:
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        filter = warnings.filters[0]
    return _remove_filter_callable(filter)

def activate_deprecation_warnings(override=True):
    if False:
        i = 10
        return i + 15
    "Call this function to activate deprecation warnings.\n\n    When running in a 'final' release we suppress deprecation warnings.\n    However, the test suite wants to see them. So when running selftest, we\n    re-enable the deprecation warnings.\n\n    Note: warnings that have already been issued under 'ignore' will not be\n    reported after this point. The 'warnings' module has already marked them as\n    handled, so they don't get issued again.\n\n    :param override: If False, only add a filter if there isn't an error filter\n        already. (This slightly differs from suppress_deprecation_warnings, in\n        because it always overrides everything but -Werror).\n\n    :return: A callable to remove the new warnings this added.\n    "
    if not override and _check_for_filter(error_only=True):
        filter = None
    else:
        warnings.filterwarnings('default', category=DeprecationWarning)
        filter = warnings.filters[0]
    return _remove_filter_callable(filter)