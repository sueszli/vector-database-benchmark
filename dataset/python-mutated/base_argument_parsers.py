"""Base classes for the primary parsers for composite command line arguments."""
from __future__ import annotations
import abc
import typing as t
from ..argparsing.parsers import CompletionError, NamespaceParser, ParserState

class ControllerNamespaceParser(NamespaceParser, metaclass=abc.ABCMeta):
    """Base class for controller namespace parsers."""

    @property
    def dest(self) -> str:
        if False:
            while True:
                i = 10
        'The name of the attribute where the value should be stored.'
        return 'controller'

    def parse(self, state: ParserState) -> t.Any:
        if False:
            print('Hello World!')
        'Parse the input from the given state and return the result.'
        if state.root_namespace.targets:
            raise ControllerRequiredFirstError()
        return super().parse(state)

class TargetNamespaceParser(NamespaceParser, metaclass=abc.ABCMeta):
    """Base class for target namespace parsers involving a single target."""

    @property
    def option_name(self) -> str:
        if False:
            print('Hello World!')
        'The option name used for this parser.'
        return '--target'

    @property
    def dest(self) -> str:
        if False:
            i = 10
            return i + 15
        'The name of the attribute where the value should be stored.'
        return 'targets'

    @property
    def use_list(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'True if the destination is a list, otherwise False.'
        return True

    @property
    def limit_one(self) -> bool:
        if False:
            i = 10
            return i + 15
        'True if only one target is allowed, otherwise False.'
        return True

class TargetsNamespaceParser(NamespaceParser, metaclass=abc.ABCMeta):
    """Base class for controller namespace parsers involving multiple targets."""

    @property
    def option_name(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'The option name used for this parser.'
        return '--target'

    @property
    def dest(self) -> str:
        if False:
            return 10
        'The name of the attribute where the value should be stored.'
        return 'targets'

    @property
    def use_list(self) -> bool:
        if False:
            print('Hello World!')
        'True if the destination is a list, otherwise False.'
        return True

class ControllerRequiredFirstError(CompletionError):
    """Exception raised when controller and target options are specified out-of-order."""

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        super().__init__('The `--controller` option must be specified before `--target` option(s).')