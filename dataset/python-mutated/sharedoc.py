"""
Shared docstrings for parameters that should be documented identically
across different functions.
"""
import re
from six import iteritems
from textwrap import dedent
from toolz import curry
PIPELINE_DOWNSAMPLING_FREQUENCY_DOC = dedent("    frequency : {'year_start', 'quarter_start', 'month_start', 'week_start'}\n        A string indicating desired sampling dates:\n\n        * 'year_start'    -> first trading day of each year\n        * 'quarter_start' -> first trading day of January, April, July, October\n        * 'month_start'   -> first trading day of each month\n        * 'week_start'    -> first trading_day of each week\n    ")
PIPELINE_ALIAS_NAME_DOC = dedent('    name : str\n        The name to alias this term as.\n    ')

def pad_lines_after_first(prefix, s):
    if False:
        print('Hello World!')
    'Apply a prefix to each line in s after the first.'
    return ('\n' + prefix).join(s.splitlines())

def format_docstring(owner_name, docstring, formatters):
    if False:
        i = 10
        return i + 15
    '\n    Template ``formatters`` into ``docstring``.\n\n    Parameters\n    ----------\n    owner_name : str\n        The name of the function or class whose docstring is being templated.\n        Only used for error messages.\n    docstring : str\n        The docstring to template.\n    formatters : dict[str -> str]\n        Parameters for a a str.format() call on ``docstring``.\n\n        Multi-line values in ``formatters`` will have leading whitespace padded\n        to match the leading whitespace of the substitution string.\n    '
    format_params = {}
    for (target, doc_for_target) in iteritems(formatters):
        regex = re.compile('^(\\s*)' + '({' + target + '})$', re.MULTILINE)
        matches = regex.findall(docstring)
        if not matches:
            raise ValueError("Couldn't find template for parameter {!r} in docstring for {}.\nParameter name must be alone on a line surrounded by braces.".format(target, owner_name))
        elif len(matches) > 1:
            raise ValueError("Couldn't found multiple templates for parameter {!r}in docstring for {}.\nParameter should only appear once.".format(target, owner_name))
        (leading_whitespace, _) = matches[0]
        format_params[target] = pad_lines_after_first(leading_whitespace, doc_for_target)
    return docstring.format(**format_params)

def templated_docstring(**docs):
    if False:
        print('Hello World!')
    "\n    Decorator allowing the use of templated docstrings.\n\n    Examples\n    --------\n    >>> @templated_docstring(foo='bar')\n    ... def my_func(self, foo):\n    ...     '''{foo}'''\n    ...\n    >>> my_func.__doc__\n    'bar'\n    "

    def decorator(f):
        if False:
            print('Hello World!')
        f.__doc__ = format_docstring(f.__name__, f.__doc__, docs)
        return f
    return decorator

@curry
def copydoc(from_, to):
    if False:
        return 10
    'Copies the docstring from one function to another.\n    Parameters\n    ----------\n    from_ : any\n        The object to copy the docstring from.\n    to : any\n        The object to copy the docstring to.\n    Returns\n    -------\n    to : any\n        ``to`` with the docstring from ``from_``\n    '
    to.__doc__ = from_.__doc__
    return to