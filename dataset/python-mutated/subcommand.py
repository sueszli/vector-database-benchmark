""" Provides a base class for defining subcommands of the Bokeh command
line application.

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from abc import ABCMeta, abstractmethod
from argparse import ArgumentParser, Namespace
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Sequence, Union
from ..util.dataclasses import NotRequired, Unspecified, dataclass, entries
if TYPE_CHECKING:
    from typing_extensions import TypeAlias
__all__ = ('Subcommand',)

@dataclass
class Argument:
    action: NotRequired[Literal['store', 'store_const', 'store_true', 'append', 'append_const', 'count', 'help', 'version', 'extend']] = Unspecified
    nargs: NotRequired[int | Literal['?', '*', '+', '...']] = Unspecified
    const: NotRequired[Any] = Unspecified
    default: NotRequired[Any] = Unspecified
    type: NotRequired[type[Any]] = Unspecified
    choices: NotRequired[Sequence[Any]] = Unspecified
    required: NotRequired[bool] = Unspecified
    help: NotRequired[str] = Unspecified
    metavar: NotRequired[str] = Unspecified
Arg: TypeAlias = tuple[Union[str, tuple[str, ...]], Argument]
Args: TypeAlias = tuple[Arg, ...]

class Subcommand(metaclass=ABCMeta):
    """ Abstract base class for subcommands

    Subclasses should implement an ``invoke(self, args)`` method that accepts
    a set of argparse processed arguments as input.

    Subclasses should also define the following class attributes:

    * ``name`` a name for this subcommand

    * ``help`` a help string for argparse to use for this subcommand

    * ``args`` the parameters to pass to ``parser.add_argument``

    The format of the ``args`` should be a sequence of tuples of the form:

    .. code-block:: python

        ('argname', Argument(
            metavar='ARGNAME',
            nargs='+',
        ))

    Example:

        A simple subcommand "foo" might look like this:

        .. code-block:: python

            class Foo(Subcommand):

                name = "foo"
                help = "performs the Foo action"
                args = (
                    ('--yell', Argument(
                        action='store_true',
                        help="Make it loud",
                    )),
                )

                def invoke(self, args):
                    if args.yell:
                        print("FOO!")
                    else:
                        print("foo")

        Then executing ``bokeh foo --yell`` would print ``FOO!`` at the console.

    """
    name: ClassVar[str]
    help: ClassVar[str]
    args: ClassVar[Args] = ()

    def __init__(self, parser: ArgumentParser) -> None:
        if False:
            return 10
        ' Initialize the subcommand with its parser\n\n        Args:\n            parser (Parser) : an Argparse ``Parser`` instance to configure\n                with the args for this subcommand.\n\n        This method will automatically add all the arguments described in\n        ``self.args``. Subclasses can perform any additional customizations\n        on ``self.parser``.\n\n        '
        self.parser = parser
        for arg in self.args:
            (flags, spec) = arg
            if not isinstance(flags, tuple):
                flags = (flags,)
            if not isinstance(spec, dict):
                kwargs = dict(entries(spec))
            else:
                kwargs = spec
            self.parser.add_argument(*flags, **kwargs)

    @abstractmethod
    def invoke(self, args: Namespace) -> bool | None:
        if False:
            print('Hello World!')
        ' Takes over main program flow to perform the subcommand.\n\n        *This method must be implemented by subclasses.*\n        subclassed overwritten methods return different types:\n        bool: Build\n        None: FileOutput (subclassed by HTML, SVG and JSON. PNG overwrites FileOutput.invoke method), Info, Init,                 Sampledata, Secret, Serve, Static\n\n\n        Args:\n            args (argparse.Namespace) : command line arguments for the subcommand to parse\n\n        Raises:\n            NotImplementedError\n\n        '
        raise NotImplementedError('implement invoke()')