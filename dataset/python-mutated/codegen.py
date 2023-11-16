"""
Utility and driver module for C++ code generation.
"""
from datetime import datetime
from enum import Enum
from io import UnsupportedOperation
from itertools import chain
import os
from sys import modules
import sys
from typing import Generator
from ..log import err
from ..util.filelike.fifo import FIFO
from ..util.fslike.directory import Directory
from ..util.fslike.wrapper import Wrapper
from .listing import generate_all

class CodegenMode(Enum):
    """
    Modus operandi
    """
    CODEGEN = 'codegen'
    DRYRUN = 'dryrun'
    CLEAN = 'clean'

class WriteCatcher(FIFO):
    """
    Behaves like FIFO, but close() is converted to seteof(),
    and read() fails if eof is not set.
    """

    def close(self) -> None:
        if False:
            print('Hello World!')
        self.eof = True

    def read(self, size: int=-1) -> bytes:
        if False:
            i = 10
            return i + 15
        if not self.eof:
            raise UnsupportedOperation('can not read from WriteCatcher while not closed for writing')
        return super().read(size)

class CodegenDirWrapper(Wrapper):
    """
    Only allows pure-read and pure-write operations;

    Intercepts all writes for later inspection, and logs all reads.

    The constructor takes the to-be-wrapped fslike object.
    """

    def __init__(self, obj):
        if False:
            return 10
        super().__init__(obj)
        self.writes = []
        self.reads = []

    def open_r(self, parts):
        if False:
            for i in range(10):
                print('nop')
        self.reads.append(parts)
        return super().open_r(parts)

    def open_w(self, parts):
        if False:
            for i in range(10):
                print('nop')
        intercept_obj = WriteCatcher()
        self.writes.append((parts, intercept_obj))
        return intercept_obj

    def get_reads(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns an iterable of all path component tuples for files that have\n        been read.\n        '
        for parts in self.reads:
            yield parts
        self.reads.clear()

    def get_writes(self) -> None:
        if False:
            print('Hello World!')
        '\n        Returns an iterable of all (path components, data_written) tuples for\n        files that have been written.\n        '
        for (parts, intercept_obj) in self.writes:
            yield (parts, intercept_obj.read())
        self.writes.clear()

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return f'CodegenDirWrapper({repr(self.obj)})'

def codegen(mode: CodegenMode, input_dir: str, output_dir: str) -> tuple[list[str], list[str]]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Calls .listing.generate_all(), and post-processes the generated\n    data, checking them and adding a header.\n    Reads the input templates relative to input_dir.\n    Writes them to output_dir according to mode. output_dir is a path or str.\n\n    Returns ({generated}, {depends}), where\n    generated is a list of (absolute) filenames of generated files, and\n    depends is a list of (absolute) filenames of dependency files.\n    '
    input_dir = Directory(input_dir).root
    output_dir = Directory(output_dir).root
    wrapper = CodegenDirWrapper(input_dir)
    generate_all(wrapper.root)
    generated = set()
    for (parts, data) in wrapper.get_writes():
        generated.add(output_dir.fsobj.resolve(parts))
        wpath = output_dir[parts]
        data = postprocess_write(parts, data)
        if mode == CodegenMode.CODEGEN:
            try:
                with wpath.open('rb') as outfile:
                    if outfile.read() == data:
                        continue
            except FileNotFoundError:
                pass
            wpath.parent.mkdirs()
            with wpath.open('wb') as outfile:
                print(f"\x1b[36mcodegen: {b'/'.join(parts).decode(errors='replace')}\x1b[0m")
                outfile.write(data)
        elif mode == CodegenMode.DRYRUN:
            pass
        elif mode == CodegenMode.CLEAN:
            if wpath.is_file():
                print(b'/'.join(parts).decode(errors='replace'))
                wpath.unlink()
        else:
            err('unknown codegen mode: %s', mode)
            sys.exit(1)
    generated = {os.path.realpath(path).decode() for path in generated}
    depends = {os.path.realpath(path) for path in get_codegen_depends(wrapper)}
    return (generated, depends)

def depend_module_blacklist():
    if False:
        for i in range(10):
            print('nop')
    '\n    Yields all modules whose source files shall explicitly not appear in the\n    dependency list, even if they have been imported.\n    '
    try:
        import openage.config
        yield openage.config
    except ImportError:
        pass
    try:
        import openage.devmode
        yield openage.devmode
    except ImportError:
        pass

def get_codegen_depends(outputwrapper: CodegenDirWrapper) -> Generator[str, None, None]:
    if False:
        print('Hello World!')
    "\n    Yields all codegen dependencies.\n\n    outputwrapper is the CodegenDirWrapper that was passed to generate_all;\n    it's used to determine the files that have been read.\n\n    In addition, all imported python modules are yielded.\n    "
    for parts in outputwrapper.get_reads():
        yield outputwrapper.obj.fsobj.resolve(parts).decode()
    module_blacklist = set(depend_module_blacklist())
    for module in modules.values():
        if module in module_blacklist:
            continue
        try:
            filename = module.__file__
        except AttributeError:
            continue
        if filename is None:
            continue
        if module.__package__ == '':
            continue
        if not filename.endswith('.py'):
            if 'openage' in module.__name__:
                print('codegeneration depends on non-.py module ' + filename)
                sys.exit(1)
        yield filename

def get_header_lines() -> Generator[str, None, None]:
    if False:
        i = 10
        return i + 15
    '\n    Yields the lines for the automatically-added file header.\n    '
    yield f'Copyright 2013-{datetime.now().year} the openage authors. See copying.md for legal info.'
    yield ''
    yield 'Warning: this file was auto-generated; manual changes are futile.'
    yield 'For details, see buildsystem/codegen.cmake and openage/codegen.'
    yield ''

def postprocess_write(parts, data: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    '\n    Post-processes a single write operation, as intercepted during codegen.\n    '
    if parts[0] != b'libopenage':
        raise ValueError('Not in libopenage source directory')
    (name, extension) = os.path.splitext(parts[-1].decode())
    if not name.endswith('.gen'):
        raise ValueError("Doesn't match required filename format .gen.SUFFIX")
    if extension in {'.h', '.cpp'}:
        comment_prefix = '//'
    else:
        raise ValueError('Extension not in {.h, .cpp}')
    datalines = data.decode('ascii').split('\n')
    if 'Copyright' in datalines[0]:
        datalines = datalines[1:]
    headerlines = []
    for line in get_header_lines():
        if line:
            headerlines.append(comment_prefix + ' ' + line)
        else:
            headerlines.append('')
    return '\n'.join(chain(headerlines, datalines)).encode('ascii')