"""Decorators marks that a doctest should be skipped.

The IPython.testing.decorators module triggers various extra imports, including
numpy and sympy if they're present. Since this decorator is used in core parts
of IPython, it's in a separate module so that running IPython doesn't trigger
those imports."""

def skip_doctest(f):
    if False:
        print('Hello World!')
    'Decorator - mark a function or method for skipping its doctest.\n\n    This decorator allows you to mark a function whose docstring you wish to\n    omit from testing, while preserving the docstring for introspection, help,\n    etc.'
    f.__skip_doctest__ = True
    return f