from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _

def _raise_error_if_not_of_type(arg, expected_type, arg_name=None):
    if False:
        i = 10
        return i + 15
    "\n    Check if the input is of expected type.\n\n    Parameters\n    ----------\n    arg            : Input argument.\n\n    expected_type  : A type OR a list of types that the argument is expected\n                     to be.\n\n    arg_name      : The name of the variable in the function being used. No\n                    name is assumed if set to None.\n\n    Examples\n    --------\n    _raise_error_if_not_of_type(sf, str, 'sf')\n    _raise_error_if_not_of_type(sf, [str, int], 'sf')\n    "
    display_name = '%s ' % arg_name if arg_name is not None else 'Argument '
    lst_expected_type = [expected_type] if type(expected_type) == type else expected_type
    err_msg = '%smust be of type %s ' % (display_name, ' or '.join([x.__name__ for x in lst_expected_type]))
    err_msg += '(not %s).' % type(arg).__name__
    if not any(map(lambda x: isinstance(arg, x), lst_expected_type)):
        raise TypeError(err_msg)

def _is_non_string_iterable(obj):
    if False:
        i = 10
        return i + 15
    return hasattr(obj, '__iter__') and (not isinstance(obj, str))