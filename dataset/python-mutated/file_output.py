""" Abstract base class for subcommands that output to a file (or stdout).

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
import argparse
import sys
from abc import abstractmethod
from os.path import splitext
from ...document import Document
from ..subcommand import Arg, Args, Argument, Subcommand
from ..util import build_single_handler_applications, die
__all__ = ('FileOutputSubcommand',)

class FileOutputSubcommand(Subcommand):
    """ Abstract subcommand to output applications as some type of file.

    """
    extension: str

    @classmethod
    def files_arg(cls, output_type_name: str) -> Arg:
        if False:
            return 10
        ' Returns a positional arg for ``files`` to specify file inputs to\n        the command.\n\n        Subclasses should include this to their class ``args``.\n\n        Example:\n\n            .. code-block:: python\n\n                class Foo(FileOutputSubcommand):\n\n                    args = (\n\n                        FileOutputSubcommand.files_arg("FOO"),\n\n                        # more args for Foo\n\n                    ) + FileOutputSubcommand.other_args()\n\n        '
        return ('files', Argument(metavar='DIRECTORY-OR-SCRIPT', nargs='+', help='The app directories or scripts to generate %s for' % output_type_name, default=None))

    @classmethod
    def other_args(cls) -> Args:
        if False:
            i = 10
            return i + 15
        ' Return args for ``-o`` / ``--output`` to specify where output\n        should be written, and for a ``--args`` to pass on any additional\n        command line args to the subcommand.\n\n        Subclasses should append these to their class ``args``.\n\n        Example:\n\n            .. code-block:: python\n\n                class Foo(FileOutputSubcommand):\n\n                    args = (\n\n                        FileOutputSubcommand.files_arg("FOO"),\n\n                        # more args for Foo\n\n                    ) + FileOutputSubcommand.other_args()\n\n        '
        return ((('-o', '--output'), Argument(metavar='FILENAME', action='append', type=str, help='Name of the output file or - for standard output.')), ('--args', Argument(metavar='COMMAND-LINE-ARGS', nargs='...', help='Any command line arguments remaining are passed on to the application handler')))

    def filename_from_route(self, route: str, ext: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n\n        '
        if route == '/':
            base = 'index'
        else:
            base = route[1:]
        return f'{base}.{ext}'

    def invoke(self, args: argparse.Namespace) -> None:
        if False:
            return 10
        '\n\n        '
        argvs = {f: args.args for f in args.files}
        applications = build_single_handler_applications(args.files, argvs)
        if args.output is None:
            outputs: list[str] = []
        else:
            outputs = list(args.output)
        if len(outputs) > len(applications):
            die('--output/-o was given too many times (%d times for %d applications)' % (len(outputs), len(applications)))
        for (route, app) in applications.items():
            doc = app.create_document()
            if len(outputs) > 0:
                filename = outputs.pop(0)
            else:
                filename = self.filename_from_route(route, self.extension)
            self.write_file(args, filename, doc)

    def write_file(self, args: argparse.Namespace, filename: str, doc: Document) -> None:
        if False:
            i = 10
            return i + 15
        '\n\n        '

        def write_str(content: str, filename: str) -> None:
            if False:
                i = 10
                return i + 15
            if filename == '-':
                print(content)
            else:
                with open(filename, 'w', encoding='utf-8') as file:
                    file.write(content)
            self.after_write_file(args, filename, doc)

        def write_bytes(content: bytes, filename: str) -> None:
            if False:
                return 10
            if filename == '-':
                sys.stdout.buffer.write(content)
            else:
                with open(filename, 'wb') as f:
                    f.write(content)
            self.after_write_file(args, filename, doc)
        contents = self.file_contents(args, doc)
        if isinstance(contents, str):
            write_str(contents, filename)
        elif isinstance(contents, bytes):
            write_bytes(contents, filename)
        else:
            if filename == '-' or len(contents) <= 1:

                def indexed(i: int) -> str:
                    if False:
                        while True:
                            i = 10
                    return filename
            else:

                def indexed(i: int) -> str:
                    if False:
                        i = 10
                        return i + 15
                    (root, ext) = splitext(filename)
                    return f'{root}_{i}{ext}'
            for (i, content) in enumerate(contents):
                if isinstance(content, str):
                    write_str(content, indexed(i))
                elif isinstance(content, bytes):
                    write_bytes(content, indexed(i))

    def after_write_file(self, args: argparse.Namespace, filename: str, doc: Document) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n\n        '
        pass

    @abstractmethod
    def file_contents(self, args: argparse.Namespace, doc: Document) -> str | bytes | list[str] | list[bytes]:
        if False:
            i = 10
            return i + 15
        ' Subclasses must override this method to return the contents of the output file for the given doc.\n        subclassed methods return different types:\n        str: html, json\n        bytes: SVG, png\n\n        Raises:\n            NotImplementedError\n\n        '
        raise NotImplementedError()