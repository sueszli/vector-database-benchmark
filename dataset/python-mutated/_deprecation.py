"""Utility for deprecating functions."""
import functools
import textwrap
import warnings

def deprecated(since: str, removed_in: str, instructions: str):
    if False:
        print('Hello World!')
    'Marks functions as deprecated.\n\n    It will result in a warning when the function is called and a note in the\n    docstring.\n\n    Args:\n        since: The version when the function was first deprecated.\n        removed_in: The version when the function will be removed.\n        instructions: The action users should take.\n    '

    def decorator(function):
        if False:
            return 10

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            warnings.warn(f"'{function.__module__}.{function.__name__}' is deprecated in version {since} and will be removed in {removed_in}. Please {instructions}.", category=FutureWarning, stacklevel=2)
            return function(*args, **kwargs)
        docstring = function.__doc__ or ''
        deprecation_note = textwrap.dedent(f'            .. deprecated:: {since}\n                Deprecated and will be removed in version {removed_in}.\n                Please {instructions}.\n            ')
        summary_and_body = docstring.split('\n\n', 1)
        if len(summary_and_body) > 1:
            (summary, body) = summary_and_body
            body = textwrap.dedent(body)
            new_docstring_parts = [deprecation_note, '\n\n', summary, body]
        else:
            summary = summary_and_body[0]
            new_docstring_parts = [deprecation_note, '\n\n', summary]
        wrapper.__doc__ = ''.join(new_docstring_parts)
        return wrapper
    return decorator