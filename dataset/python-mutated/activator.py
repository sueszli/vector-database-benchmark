from __future__ import annotations
import os
from abc import ABCMeta, abstractmethod

class Activator(metaclass=ABCMeta):
    """Generates activate script for the virtual environment."""

    def __init__(self, options) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a new activator generator.\n\n        :param options: the parsed options as defined within :meth:`add_parser_arguments`\n        '
        self.flag_prompt = os.path.basename(os.getcwd()) if options.prompt == '.' else options.prompt

    @classmethod
    def supports(cls, interpreter):
        if False:
            return 10
        '\n        Check if the activation script is supported in the given interpreter.\n\n        :param interpreter: the interpreter we need to support\n        :return: ``True`` if supported, ``False`` otherwise\n        '
        return True

    @classmethod
    def add_parser_arguments(cls, parser, interpreter):
        if False:
            i = 10
            return i + 15
        '\n        Add CLI arguments for this activation script.\n\n        :param parser: the CLI parser\n        :param interpreter: the interpreter this virtual environment is based of\n        '

    @abstractmethod
    def generate(self, creator):
        if False:
            i = 10
            return i + 15
        '\n        Generate activate script for the given creator.\n\n        :param creator: the creator (based of :class:`virtualenv.create.creator.Creator`) we used to create this         virtual environment\n        '
        raise NotImplementedError
__all__ = ['Activator']