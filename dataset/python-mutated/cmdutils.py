"""qutebrowser has the concept of functions, exposed to the user as commands.

Creating a new command is straightforward::

  from qutebrowser.api import cmdutils

  @cmdutils.register(...)
  def foo():
      ...

The commands arguments are automatically deduced by inspecting your function.

The types of the function arguments are inferred based on their default values,
e.g., an argument `foo=True` will be converted to a flag `-f`/`--foo` in
qutebrowser's commandline.

The type can be overridden using Python's function annotations::

  @cmdutils.register(...)
  def foo(bar: int, baz=True):
      ...

Possible values:

- A callable (``int``, ``float``, etc.): Gets called to validate/convert the
  value.
- A python enum type: All members of the enum are possible values.
- A ``typing.Union`` of multiple types above: Any of these types are valid
  values, e.g., ``Union[str, int]``.
"""
import inspect
from typing import Any, Callable, Iterable, Protocol, Optional, Dict, cast
from qutebrowser.utils import qtutils
from qutebrowser.commands import command, cmdexc
from qutebrowser.utils.usertypes import KeyMode, CommandValue as Value

class CommandError(cmdexc.Error):
    """Raised when a command encounters an error while running.

    If your command handler encounters an error and cannot continue, raise this
    exception with an appropriate error message::

        raise cmdexc.CommandError("Message")

    The message will then be shown in the qutebrowser status bar.

    .. note::

       You should only raise this exception while a command handler is run.
       Raising it at another point causes qutebrowser to crash due to an
       unhandled exception.
    """

def check_overflow(arg: int, ctype: str) -> None:
    if False:
        print('Hello World!')
    "Check if the given argument is in bounds for the given type.\n\n    Args:\n        arg: The argument to check.\n        ctype: The C++/Qt type to check as a string ('int'/'int64').\n    "
    try:
        qtutils.check_overflow(arg, ctype)
    except OverflowError:
        raise CommandError('Numeric argument is too large for internal {} representation.'.format(ctype))

def check_exclusive(flags: Iterable[bool], names: Iterable[str]) -> None:
    if False:
        while True:
            i = 10
    'Check if only one flag is set with exclusive flags.\n\n    Raise a CommandError if not.\n\n    Args:\n        flags: The flag values to check.\n        names: A list of names (corresponding to the flags argument).\n    '
    if sum((1 for e in flags if e)) > 1:
        argstr = '/'.join(('-' + e for e in names))
        raise CommandError('Only one of {} can be given!'.format(argstr))
_CmdHandlerFunc = Callable[..., Any]

class _CmdHandlerType(Protocol):
    """A qutebrowser command function, which had qute_args patched on it.

    Applying @cmdutils.argument to a function will patch it with a qute_args attribute.
    Below, we cast the decorated function to _CmdHandlerType to make mypy aware of this.
    """
    qute_args: Optional[Dict[str, 'command.ArgInfo']]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if False:
            for i in range(10):
                print('nop')
        ...

class register:
    """Decorator to register a new command handler."""

    def __init__(self, *, instance: str=None, name: str=None, deprecated_name: str=None, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        'Save decorator arguments.\n\n        Gets called on parse-time with the decorator arguments.\n\n        Args:\n            See class attributes.\n        '
        self._instance = instance
        self._name = name
        self._deprecated_name = deprecated_name
        self._kwargs = kwargs

    def __call__(self, func: _CmdHandlerFunc) -> _CmdHandlerType:
        if False:
            for i in range(10):
                print('nop')
        "Register the command before running the function.\n\n        Gets called when a function should be decorated.\n\n        Doesn't actually decorate anything, but creates a Command object and\n        registers it in the global commands dict.\n\n        Args:\n            func: The function to be decorated.\n\n        Return:\n            The original function (unmodified).\n        "
        if self._name is None:
            name = func.__name__.lower().replace('_', '-')
        else:
            assert isinstance(self._name, str), self._name
            name = self._name
        cmd = command.Command(name=name, instance=self._instance, handler=func, **self._kwargs)
        cmd.register()
        if self._deprecated_name is not None:
            deprecated_cmd = command.Command(name=self._deprecated_name, instance=self._instance, handler=func, deprecated=f'use {name} instead', **self._kwargs)
            deprecated_cmd.register()
        func = cast(_CmdHandlerType, func)
        func.qute_args = None
        return func

class argument:
    """Decorator to customize an argument.

    You can customize how an argument is handled using the
    ``@cmdutils.argument`` decorator *after* ``@cmdutils.register``. This can,
    for example, be used to customize the flag an argument should get::

      @cmdutils.register(...)
      @cmdutils.argument('bar', flag='c')
      def foo(bar):
          ...

    For a ``str`` argument, you can restrict the allowed strings using
    ``choices``::

      @cmdutils.register(...)
      @cmdutils.argument('bar', choices=['val1', 'val2'])
      def foo(bar: str):
          ...

    For ``Union`` types, the given ``choices`` are only checked if other
    types (like ``int``) don't match.

    The following arguments are supported for ``@cmdutils.argument``:

    - ``flag``: Customize the short flag (``-x``) the argument will get.
    - ``value``: Tell qutebrowser to fill the argument with special values:

      * ``value=cmdutils.Value.count``: The ``count`` given by the user to the
        command.
      * ``value=cmdutils.Value.win_id``: The window ID of the current window.
      * ``value=cmdutils.Value.cur_tab``: The tab object which is currently
        focused.

    - ``completion``: A completion function to use when completing arguments
      for the given command.
    - ``choices``: The allowed string choices for the argument.

    The name of an argument will always be the parameter name, with any
    trailing underscores stripped and underscores replaced by dashes.
    """

    def __init__(self, argname: str, **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._argname = argname
        self._kwargs = kwargs

    def __call__(self, func: _CmdHandlerFunc) -> _CmdHandlerType:
        if False:
            i = 10
            return i + 15
        funcname = func.__name__
        if self._argname not in inspect.signature(func).parameters:
            raise ValueError('{} has no argument {}!'.format(funcname, self._argname))
        func = cast(_CmdHandlerType, func)
        if not hasattr(func, 'qute_args'):
            func.qute_args = {}
        elif func.qute_args is None:
            raise ValueError('@cmdutils.argument got called above (after) @cmdutils.register for {}!'.format(funcname))
        arginfo = command.ArgInfo(**self._kwargs)
        func.qute_args[self._argname] = arginfo
        return func