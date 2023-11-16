"""Custom descriptions and summaries for the builtin types.

The docstrings for objects of primitive types reflect the type of the object,
rather than the object itself. For example, the docstring for any dict is this:

> print({'key': 'value'}.__doc__)
dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

As you can see, this docstring is more pertinent to the function `dict` and
would be suitable as the result of `dict.__doc__`, but is wholely unsuitable
as a description for the dict `{'key': 'value'}`.

This modules aims to resolve that problem, providing custom summaries and
descriptions for primitive typed values.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import formatting
import six
TWO_DOUBLE_QUOTES = '""'
STRING_DESC_PREFIX = 'The string '

def NeedsCustomDescription(component):
    if False:
        for i in range(10):
            print('nop')
    'Whether the component should use a custom description and summary.\n\n  Components of primitive type, such as ints, floats, dicts, lists, and others\n  have messy builtin docstrings. These are inappropriate for display as\n  descriptions and summaries in a CLI. This function determines whether the\n  provided component has one of these docstrings.\n\n  Note that an object such as `int` has the same docstring as an int like `3`.\n  The docstring is OK for `int`, but is inappropriate as a docstring for `3`.\n\n  Args:\n    component: The component of interest.\n  Returns:\n    Whether the component should use a custom description and summary.\n  '
    type_ = type(component)
    if type_ in six.string_types or type_ in six.integer_types or type_ is six.text_type or (type_ is six.binary_type) or (type_ in (float, complex, bool)) or (type_ in (dict, tuple, list, set, frozenset)):
        return True
    return False

def GetStringTypeSummary(obj, available_space, line_length):
    if False:
        while True:
            i = 10
    'Returns a custom summary for string type objects.\n\n  This function constructs a summary for string type objects by double quoting\n  the string value. The double quoted string value will be potentially truncated\n  with ellipsis depending on whether it has enough space available to show the\n  full string value.\n\n  Args:\n    obj: The object to generate summary for.\n    available_space: Number of character spaces available.\n    line_length: The full width of the terminal, default is 80.\n\n  Returns:\n    A summary for the input object.\n  '
    if len(obj) + len(TWO_DOUBLE_QUOTES) <= available_space:
        content = obj
    else:
        additional_len_needed = len(TWO_DOUBLE_QUOTES) + len(formatting.ELLIPSIS)
        if available_space < additional_len_needed:
            available_space = line_length
        content = formatting.EllipsisTruncate(obj, available_space - len(TWO_DOUBLE_QUOTES), line_length)
    return formatting.DoubleQuote(content)

def GetStringTypeDescription(obj, available_space, line_length):
    if False:
        for i in range(10):
            print('nop')
    'Returns the predefined description for string obj.\n\n  This function constructs a description for string type objects in the format\n  of \'The string "<string_value>"\'. <string_value> could be potentially\n  truncated depending on whether it has enough space available to show the full\n  string value.\n\n  Args:\n    obj: The object to generate description for.\n    available_space: Number of character spaces available.\n    line_length: The full width of the terminal, default if 80.\n\n  Returns:\n    A description for input object.\n  '
    additional_len_needed = len(STRING_DESC_PREFIX) + len(TWO_DOUBLE_QUOTES) + len(formatting.ELLIPSIS)
    if available_space < additional_len_needed:
        available_space = line_length
    return STRING_DESC_PREFIX + formatting.DoubleQuote(formatting.EllipsisTruncate(obj, available_space - len(STRING_DESC_PREFIX) - len(TWO_DOUBLE_QUOTES), line_length))
CUSTOM_DESC_SUM_FN_DICT = {'str': (GetStringTypeSummary, GetStringTypeDescription), 'unicode': (GetStringTypeSummary, GetStringTypeDescription)}

def GetSummary(obj, available_space, line_length):
    if False:
        return 10
    obj_type_name = type(obj).__name__
    if obj_type_name in CUSTOM_DESC_SUM_FN_DICT:
        return CUSTOM_DESC_SUM_FN_DICT.get(obj_type_name)[0](obj, available_space, line_length)
    return None

def GetDescription(obj, available_space, line_length):
    if False:
        i = 10
        return i + 15
    obj_type_name = type(obj).__name__
    if obj_type_name in CUSTOM_DESC_SUM_FN_DICT:
        return CUSTOM_DESC_SUM_FN_DICT.get(obj_type_name)[1](obj, available_space, line_length)
    return None