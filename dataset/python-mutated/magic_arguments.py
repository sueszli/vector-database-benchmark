''' A decorator-based method of constructing IPython magics with `argparse`
option handling.

New magic functions can be defined like so::

    from IPython.core.magic_arguments import (argument, magic_arguments,
        parse_argstring)

    @magic_arguments()
    @argument('-o', '--option', help='An optional argument.')
    @argument('arg', type=int, help='An integer positional argument.')
    def magic_cool(self, arg):
        """ A really cool magic command.

    """
        args = parse_argstring(magic_cool, arg)
        ...

The `@magic_arguments` decorator marks the function as having argparse arguments.
The `@argument` decorator adds an argument using the same syntax as argparse's
`add_argument()` method. More sophisticated uses may also require the
`@argument_group` or `@kwds` decorator to customize the formatting and the
parsing.

Help text for the magic is automatically generated from the docstring and the
arguments::

    In[1]: %cool?
        %cool [-o OPTION] arg
        
        A really cool magic command.
        
        positional arguments:
          arg                   An integer positional argument.
        
        optional arguments:
          -o OPTION, --option OPTION
                                An optional argument.

Here is an elaborated example that uses default parameters in `argument` and calls the `args` in the cell magic::

    from IPython.core.magic import register_cell_magic
    from IPython.core.magic_arguments import (argument, magic_arguments,
                                            parse_argstring)


    @magic_arguments()
    @argument(
        "--option",
        "-o",
        help=("Add an option here"),
    )
    @argument(
        "--style",
        "-s",
        default="foo",
        help=("Add some style arguments"),
    )
    @register_cell_magic
    def my_cell_magic(line, cell):
        args = parse_argstring(my_cell_magic, line)
        print(f"{args.option=}")
        print(f"{args.style=}")
        print(f"{cell=}")

In a jupyter notebook, this cell magic can be executed like this::

    %%my_cell_magic -o Hello
    print("bar")
    i = 42

Inheritance diagram:

.. inheritance-diagram:: IPython.core.magic_arguments
   :parts: 3

'''
import argparse
import re
from IPython.core.error import UsageError
from IPython.utils.decorators import undoc
from IPython.utils.process import arg_split
from IPython.utils.text import dedent
NAME_RE = re.compile('[a-zA-Z][a-zA-Z0-9_-]*$')

@undoc
class MagicHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """A HelpFormatter with a couple of changes to meet our needs.
    """

    def _fill_text(self, text, width, indent):
        if False:
            i = 10
            return i + 15
        return argparse.RawDescriptionHelpFormatter._fill_text(self, dedent(text), width, indent)

    def _format_action_invocation(self, action):
        if False:
            while True:
                i = 10
        if not action.option_strings:
            (metavar,) = self._metavar_formatter(action, action.dest)(1)
            return metavar
        else:
            parts = []
            if action.nargs == 0:
                parts.extend(action.option_strings)
            else:
                default = action.dest.upper()
                args_string = self._format_args(action, default)
                if not NAME_RE.match(args_string):
                    args_string = '<%s>' % args_string
                for option_string in action.option_strings:
                    parts.append('%s %s' % (option_string, args_string))
            return ', '.join(parts)

    def add_usage(self, usage, actions, groups, prefix='::\n\n  %'):
        if False:
            return 10
        super(MagicHelpFormatter, self).add_usage(usage, actions, groups, prefix)

class MagicArgumentParser(argparse.ArgumentParser):
    """ An ArgumentParser tweaked for use by IPython magics.
    """

    def __init__(self, prog=None, usage=None, description=None, epilog=None, parents=None, formatter_class=MagicHelpFormatter, prefix_chars='-', argument_default=None, conflict_handler='error', add_help=False):
        if False:
            for i in range(10):
                print('nop')
        if parents is None:
            parents = []
        super(MagicArgumentParser, self).__init__(prog=prog, usage=usage, description=description, epilog=epilog, parents=parents, formatter_class=formatter_class, prefix_chars=prefix_chars, argument_default=argument_default, conflict_handler=conflict_handler, add_help=add_help)

    def error(self, message):
        if False:
            return 10
        ' Raise a catchable error instead of exiting.\n        '
        raise UsageError(message)

    def parse_argstring(self, argstring):
        if False:
            i = 10
            return i + 15
        ' Split a string into an argument list and parse that argument list.\n        '
        argv = arg_split(argstring)
        return self.parse_args(argv)

def construct_parser(magic_func):
    if False:
        print('Hello World!')
    ' Construct an argument parser using the function decorations.\n    '
    kwds = getattr(magic_func, 'argcmd_kwds', {})
    if 'description' not in kwds:
        kwds['description'] = getattr(magic_func, '__doc__', None)
    arg_name = real_name(magic_func)
    parser = MagicArgumentParser(arg_name, **kwds)
    group = None
    for deco in magic_func.decorators[::-1]:
        result = deco.add_to_parser(parser, group)
        if result is not None:
            group = result
    magic_func.__doc__ = parser.format_help()
    return parser

def parse_argstring(magic_func, argstring):
    if False:
        print('Hello World!')
    ' Parse the string of arguments for the given magic function.\n    '
    return magic_func.parser.parse_argstring(argstring)

def real_name(magic_func):
    if False:
        for i in range(10):
            print('nop')
    ' Find the real name of the magic.\n    '
    magic_name = magic_func.__name__
    if magic_name.startswith('magic_'):
        magic_name = magic_name[len('magic_'):]
    return getattr(magic_func, 'argcmd_name', magic_name)

class ArgDecorator(object):
    """ Base class for decorators to add ArgumentParser information to a method.
    """

    def __call__(self, func):
        if False:
            i = 10
            return i + 15
        if not getattr(func, 'has_arguments', False):
            func.has_arguments = True
            func.decorators = []
        func.decorators.append(self)
        return func

    def add_to_parser(self, parser, group):
        if False:
            for i in range(10):
                print('nop')
        " Add this object's information to the parser, if necessary.\n        "
        pass

class magic_arguments(ArgDecorator):
    """ Mark the magic as having argparse arguments and possibly adjust the
    name.
    """

    def __init__(self, name=None):
        if False:
            return 10
        self.name = name

    def __call__(self, func):
        if False:
            return 10
        if not getattr(func, 'has_arguments', False):
            func.has_arguments = True
            func.decorators = []
        if self.name is not None:
            func.argcmd_name = self.name
        func.parser = construct_parser(func)
        return func

class ArgMethodWrapper(ArgDecorator):
    """
    Base class to define a wrapper for ArgumentParser method.

    Child class must define either `_method_name` or `add_to_parser`.

    """
    _method_name: str

    def __init__(self, *args, **kwds):
        if False:
            return 10
        self.args = args
        self.kwds = kwds

    def add_to_parser(self, parser, group):
        if False:
            while True:
                i = 10
        " Add this object's information to the parser.\n        "
        if group is not None:
            parser = group
        getattr(parser, self._method_name)(*self.args, **self.kwds)
        return None

class argument(ArgMethodWrapper):
    """ Store arguments and keywords to pass to add_argument().

    Instances also serve to decorate command methods.
    """
    _method_name = 'add_argument'

class defaults(ArgMethodWrapper):
    """ Store arguments and keywords to pass to set_defaults().

    Instances also serve to decorate command methods.
    """
    _method_name = 'set_defaults'

class argument_group(ArgMethodWrapper):
    """ Store arguments and keywords to pass to add_argument_group().

    Instances also serve to decorate command methods.
    """

    def add_to_parser(self, parser, group):
        if False:
            i = 10
            return i + 15
        " Add this object's information to the parser.\n        "
        return parser.add_argument_group(*self.args, **self.kwds)

class kwds(ArgDecorator):
    """ Provide other keywords to the sub-parser constructor.
    """

    def __init__(self, **kwds):
        if False:
            return 10
        self.kwds = kwds

    def __call__(self, func):
        if False:
            while True:
                i = 10
        func = super(kwds, self).__call__(func)
        func.argcmd_kwds = self.kwds
        return func
__all__ = ['magic_arguments', 'argument', 'argument_group', 'kwds', 'parse_argstring']