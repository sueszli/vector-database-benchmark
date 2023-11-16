"""A bunch of useful code utilities."""
import re

def extract_args(line):
    if False:
        return 10
    'Parse argument strings from all outer parentheses in a line of code.\n\n    Parameters\n    ----------\n    line : str\n        A line of code\n\n    Returns\n    -------\n    list of strings\n        Contents of the outer parentheses\n\n    Example\n    -------\n    >>> line = \'foo(bar, baz), "a", my(func)\'\n    >>> extract_args(line)\n    [\'bar, baz\', \'func\']\n\n    '
    stack = 0
    startIndex = None
    results = []
    for (i, c) in enumerate(line):
        if c == '(':
            if stack == 0:
                startIndex = i + 1
            stack += 1
        elif c == ')':
            stack -= 1
            if stack == 0:
                results.append(line[startIndex:i])
    return results

def get_method_args_from_code(args, line):
    if False:
        while True:
            i = 10
    "Parse arguments from a stringified arguments list inside parentheses\n\n    Parameters\n    ----------\n    args : list\n        A list where it's size matches the expected number of parsed arguments\n    line : str\n        Stringified line of code with method arguments inside parentheses\n\n    Returns\n    -------\n    list of strings\n        Parsed arguments\n\n    Example\n    -------\n    >>> line = 'foo(bar, baz, my(func, tion))'\n    >>>\n    >>> get_method_args_from_code(range(0, 3), line)\n    ['bar', 'baz', 'my(func, tion)']\n\n    "
    line_args = extract_args(line)[0]
    if len(args) > 1:
        inputs = re.split(',\\s*(?![^(){}[\\]]*\\))', line_args)
        assert len(inputs) == len(args), 'Could not split arguments'
    else:
        inputs = [line_args]
    return inputs