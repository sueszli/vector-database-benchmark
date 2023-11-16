"""Check the stable ABI manifest or generate files from it

By default, the tool only checks existing files/libraries.
Pass --generate to recreate auto-generated files instead.

For actions that take a FILENAME, the filename can be left out to use a default
(relative to the manifest file, as they appear in the CPython codebase).
"""
from functools import partial
from pathlib import Path
import dataclasses
import subprocess
import sysconfig
import argparse
import textwrap
import difflib
import shutil
import sys
import os
import os.path
import io
import re
import csv
MISSING = object()
EXCLUDED_HEADERS = {'bytes_methods.h', 'cellobject.h', 'classobject.h', 'code.h', 'compile.h', 'datetime.h', 'dtoa.h', 'frameobject.h', 'funcobject.h', 'genobject.h', 'longintrepr.h', 'parsetok.h', 'pyatomic.h', 'pytime.h', 'token.h', 'ucnhash.h'}
MACOS = sys.platform == 'darwin'
UNIXY = MACOS or sys.platform == 'linux'
IFDEF_DOC_NOTES = {'MS_WINDOWS': 'on Windows', 'HAVE_FORK': 'on platforms with fork()', 'USE_STACKCHECK': 'on platforms with USE_STACKCHECK'}

@dataclasses.dataclass
class Manifest:
    """Collection of `ABIItem`s forming the stable ABI/limited API."""
    kind = 'manifest'
    contents: dict = dataclasses.field(default_factory=dict)

    def add(self, item):
        if False:
            return 10
        if item.name in self.contents:
            raise ValueError(f'duplicate ABI item {item.name}')
        self.contents[item.name] = item

    @property
    def feature_defines(self):
        if False:
            print('Hello World!')
        "Return all feature defines which affect what's available\n\n        These are e.g. HAVE_FORK and MS_WINDOWS.\n        "
        return set((item.ifdef for item in self.contents.values())) - {None}

    def select(self, kinds, *, include_abi_only=True, ifdef=None):
        if False:
            while True:
                i = 10
        "Yield selected items of the manifest\n\n        kinds: set of requested kinds, e.g. {'function', 'macro'}\n        include_abi_only: if True (default), include all items of the\n            stable ABI.\n            If False, include only items from the limited API\n            (i.e. items people should use today)\n        ifdef: set of feature defines (e.g. {'HAVE_FORK', 'MS_WINDOWS'}).\n            If None (default), items are not filtered by this. (This is\n            different from the empty set, which filters out all such\n            conditional items.)\n        "
        for (name, item) in sorted(self.contents.items()):
            if item.kind not in kinds:
                continue
            if item.abi_only and (not include_abi_only):
                continue
            if ifdef is not None and item.ifdef is not None and (item.ifdef not in ifdef):
                continue
            yield item

    def dump(self):
        if False:
            print('Hello World!')
        'Yield lines to recreate the manifest file (sans comments/newlines)'
        for item in self.contents.values():
            yield from item.dump(indent=0)

@dataclasses.dataclass
class ABIItem:
    """Information on one item (function, macro, struct, etc.)"""
    kind: str
    name: str
    added: str = None
    contents: list = dataclasses.field(default_factory=list)
    abi_only: bool = False
    ifdef: str = None
    KINDS = frozenset({'struct', 'function', 'macro', 'data', 'const', 'typedef'})

    def dump(self, indent=0):
        if False:
            return 10
        yield f"{'    ' * indent}{self.kind} {self.name}"
        if self.added:
            yield f"{'    ' * (indent + 1)}added {self.added}"
        if self.ifdef:
            yield f"{'    ' * (indent + 1)}ifdef {self.ifdef}"
        if self.abi_only:
            yield f"{'    ' * (indent + 1)}abi_only"

def parse_manifest(file):
    if False:
        while True:
            i = 10
    'Parse the given file (iterable of lines) to a Manifest'
    LINE_RE = re.compile('(?P<indent>[ ]*)(?P<kind>[^ ]+)[ ]*(?P<content>.*)')
    manifest = Manifest()
    levels = [(manifest, -1)]

    def raise_error(msg):
        if False:
            for i in range(10):
                print('nop')
        raise SyntaxError(f'line {lineno}: {msg}')
    for (lineno, line) in enumerate(file, start=1):
        (line, sep, comment) = line.partition('#')
        line = line.rstrip()
        if not line:
            continue
        match = LINE_RE.fullmatch(line)
        if not match:
            raise_error(f'invalid syntax: {line}')
        level = len(match['indent'])
        kind = match['kind']
        content = match['content']
        while level <= levels[-1][1]:
            levels.pop()
        parent = levels[-1][0]
        entry = None
        if kind in ABIItem.KINDS:
            if parent.kind not in {'manifest'}:
                raise_error(f'{kind} cannot go in {parent.kind}')
            entry = ABIItem(kind, content)
            parent.add(entry)
        elif kind in {'added', 'ifdef'}:
            if parent.kind not in ABIItem.KINDS:
                raise_error(f'{kind} cannot go in {parent.kind}')
            setattr(parent, kind, content)
        elif kind in {'abi_only'}:
            if parent.kind not in {'function', 'data'}:
                raise_error(f'{kind} cannot go in {parent.kind}')
            parent.abi_only = True
        else:
            raise_error(f'unknown kind {kind!r}')
        levels.append((entry, level))
    return manifest
generators = []

def generator(var_name, default_path):
    if False:
        return 10
    'Decorates a file generator: function that writes to a file'

    def _decorator(func):
        if False:
            print('Hello World!')
        func.var_name = var_name
        func.arg_name = '--' + var_name.replace('_', '-')
        func.default_path = default_path
        generators.append(func)
        return func
    return _decorator

@generator('python3dll', 'PC/python3dll.c')
def gen_python3dll(manifest, args, outfile):
    if False:
        print('Hello World!')
    'Generate/check the source for the Windows stable ABI library'
    write = partial(print, file=outfile)
    write(textwrap.dedent('\n        /* Re-export stable Python ABI */\n\n        /* Generated by Tools/scripts/stable_abi.py */\n\n        #ifdef _M_IX86\n        #define DECORATE "_"\n        #else\n        #define DECORATE\n        #endif\n\n        #define EXPORT_FUNC(name) \\\n            __pragma(comment(linker, "/EXPORT:" DECORATE #name "=" PYTHON_DLL_NAME "." #name))\n        #define EXPORT_DATA(name) \\\n            __pragma(comment(linker, "/EXPORT:" DECORATE #name "=" PYTHON_DLL_NAME "." #name ",DATA"))\n    '))

    def sort_key(item):
        if False:
            return 10
        return item.name.lower()
    for item in sorted(manifest.select({'function'}, include_abi_only=True, ifdef={'MS_WINDOWS'}), key=sort_key):
        write(f'EXPORT_FUNC({item.name})')
    write()
    for item in sorted(manifest.select({'data'}, include_abi_only=True, ifdef={'MS_WINDOWS'}), key=sort_key):
        write(f'EXPORT_DATA({item.name})')
REST_ROLES = {'function': 'function', 'data': 'var', 'struct': 'type', 'macro': 'macro', 'typedef': 'type'}

@generator('doc_list', 'Doc/data/stable_abi.dat')
def gen_doc_annotations(manifest, args, outfile):
    if False:
        i = 10
        return i + 15
    'Generate/check the stable ABI list for documentation annotations'
    writer = csv.DictWriter(outfile, ['role', 'name', 'added', 'ifdef_note'], lineterminator='\n')
    writer.writeheader()
    for item in manifest.select(REST_ROLES.keys(), include_abi_only=False):
        if item.ifdef:
            ifdef_note = IFDEF_DOC_NOTES[item.ifdef]
        else:
            ifdef_note = None
        writer.writerow({'role': REST_ROLES[item.kind], 'name': item.name, 'added': item.added, 'ifdef_note': ifdef_note})

def generate_or_check(manifest, args, path, func):
    if False:
        for i in range(10):
            print('nop')
    'Generate/check a file with a single generator\n\n    Return True if successful; False if a comparison failed.\n    '
    outfile = io.StringIO()
    func(manifest, args, outfile)
    generated = outfile.getvalue()
    existing = path.read_text()
    if generated != existing:
        if args.generate:
            path.write_text(generated)
        else:
            print(f'File {path} differs from expected!')
            diff = difflib.unified_diff(generated.splitlines(), existing.splitlines(), str(path), '<expected>', lineterm='')
            for line in diff:
                print(line)
            return False
    return True

def do_unixy_check(manifest, args):
    if False:
        i = 10
        return i + 15
    'Check headers & library using "Unixy" tools (GCC/clang, binutils)'
    okay = True
    present_macros = gcc_get_limited_api_macros(['Include/Python.h'])
    feature_defines = manifest.feature_defines & present_macros
    expected_macros = set((item.name for item in manifest.select({'macro'})))
    missing_macros = expected_macros - present_macros
    okay &= _report_unexpected_items(missing_macros, 'Some macros from are not defined from "Include/Python.h"' + 'with Py_LIMITED_API:')
    expected_symbols = set((item.name for item in manifest.select({'function', 'data'}, include_abi_only=True, ifdef=feature_defines)))
    LIBRARY = sysconfig.get_config_var('LIBRARY')
    if not LIBRARY:
        raise Exception('failed to get LIBRARY variable from sysconfig')
    if os.path.exists(LIBRARY):
        okay &= binutils_check_library(manifest, LIBRARY, expected_symbols, dynamic=False)
    LDLIBRARY = sysconfig.get_config_var('LDLIBRARY')
    if not LDLIBRARY:
        raise Exception('failed to get LDLIBRARY variable from sysconfig')
    okay &= binutils_check_library(manifest, LDLIBRARY, expected_symbols, dynamic=False)
    expected_defs = set((item.name for item in manifest.select({'function', 'data'}, include_abi_only=False, ifdef=feature_defines)))
    found_defs = gcc_get_limited_api_definitions(['Include/Python.h'])
    missing_defs = expected_defs - found_defs
    okay &= _report_unexpected_items(missing_defs, 'Some expected declarations were not declared in ' + '"Include/Python.h" with Py_LIMITED_API:')
    private_symbols = {n for n in expected_symbols if n.startswith('_')}
    extra_defs = found_defs - expected_defs - private_symbols
    okay &= _report_unexpected_items(extra_defs, 'Some extra declarations were found in "Include/Python.h" ' + 'with Py_LIMITED_API:')
    return okay

def _report_unexpected_items(items, msg):
    if False:
        while True:
            i = 10
    'If there are any `items`, report them using "msg" and return false'
    if items:
        print(msg, file=sys.stderr)
        for item in sorted(items):
            print(' -', item, file=sys.stderr)
        return False
    return True

def binutils_get_exported_symbols(library, dynamic=False):
    if False:
        while True:
            i = 10
    'Retrieve exported symbols using the nm(1) tool from binutils'
    args = ['nm', '--no-sort']
    if dynamic:
        args.append('--dynamic')
    args.append(library)
    proc = subprocess.run(args, stdout=subprocess.PIPE, universal_newlines=True)
    if proc.returncode:
        sys.stdout.write(proc.stdout)
        sys.exit(proc.returncode)
    stdout = proc.stdout.rstrip()
    if not stdout:
        raise Exception('command output is empty')
    for line in stdout.splitlines():
        if not line:
            continue
        parts = line.split(maxsplit=2)
        if len(parts) < 3:
            continue
        symbol = parts[-1]
        if MACOS and symbol.startswith('_'):
            yield symbol[1:]
        else:
            yield symbol

def binutils_check_library(manifest, library, expected_symbols, dynamic):
    if False:
        while True:
            i = 10
    'Check that library exports all expected_symbols'
    available_symbols = set(binutils_get_exported_symbols(library, dynamic))
    missing_symbols = expected_symbols - available_symbols
    if missing_symbols:
        print(textwrap.dedent(f"            Some symbols from the limited API are missing from {library}:\n                {', '.join(missing_symbols)}\n\n            This error means that there are some missing symbols among the\n            ones exported in the library.\n            This normally means that some symbol, function implementation or\n            a prototype belonging to a symbol in the limited API has been\n            deleted or is missing.\n        "), file=sys.stderr)
        return False
    return True

def gcc_get_limited_api_macros(headers):
    if False:
        print('Hello World!')
    'Get all limited API macros from headers.\n\n    Runs the preprocessor over all the header files in "Include" setting\n    "-DPy_LIMITED_API" to the correct value for the running version of the\n    interpreter and extracting all macro definitions (via adding -dM to the\n    compiler arguments).\n\n    Requires Python built with a GCC-compatible compiler. (clang might work)\n    '
    api_hexversion = sys.version_info.major << 24 | sys.version_info.minor << 16
    preprocesor_output_with_macros = subprocess.check_output(sysconfig.get_config_var('CC').split() + ['-DSIZEOF_WCHAR_T=4', f'-DPy_LIMITED_API={api_hexversion}', '-I.', '-I./Include', '-dM', '-E'] + [str(file) for file in headers], text=True)
    return {target for target in re.findall('#define (\\w+)', preprocesor_output_with_macros)}

def gcc_get_limited_api_definitions(headers):
    if False:
        i = 10
        return i + 15
    'Get all limited API definitions from headers.\n\n    Run the preprocessor over all the header files in "Include" setting\n    "-DPy_LIMITED_API" to the correct value for the running version of the\n    interpreter.\n\n    The limited API symbols will be extracted from the output of this command\n    as it includes the prototypes and definitions of all the exported symbols\n    that are in the limited api.\n\n    This function does *NOT* extract the macros defined on the limited API\n\n    Requires Python built with a GCC-compatible compiler. (clang might work)\n    '
    api_hexversion = sys.version_info.major << 24 | sys.version_info.minor << 16
    preprocesor_output = subprocess.check_output(sysconfig.get_config_var('CC').split() + ['-DPyAPI_FUNC=__PyAPI_FUNC', '-DPyAPI_DATA=__PyAPI_DATA', '-DEXPORT_DATA=__EXPORT_DATA', '-D_Py_NO_RETURN=', '-DSIZEOF_WCHAR_T=4', f'-DPy_LIMITED_API={api_hexversion}', '-I.', '-I./Include', '-E'] + [str(file) for file in headers], text=True, stderr=subprocess.DEVNULL)
    stable_functions = set(re.findall('__PyAPI_FUNC\\(.*?\\)\\s*(.*?)\\s*\\(', preprocesor_output))
    stable_exported_data = set(re.findall('__EXPORT_DATA\\((.*?)\\)', preprocesor_output))
    stable_data = set(re.findall('__PyAPI_DATA\\(.*?\\)[\\s\\*\\(]*([^);]*)\\)?.*;', preprocesor_output))
    return stable_data | stable_exported_data | stable_functions

def check_private_names(manifest):
    if False:
        i = 10
        return i + 15
    "Ensure limited API doesn't contain private names\n\n    Names prefixed by an underscore are private by definition.\n    "
    for (name, item) in manifest.contents.items():
        if name.startswith('_') and (not item.abi_only):
            raise ValueError(f'`{name}` is private (underscore-prefixed) and should be ' + 'removed from the stable ABI list or or marked `abi_only`')

def main():
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('file', type=Path, metavar='FILE', help='file with the stable abi manifest')
    parser.add_argument('--generate', action='store_true', help='generate file(s), rather than just checking them')
    parser.add_argument('--generate-all', action='store_true', help='as --generate, but generate all file(s) using default filenames.' + ' (unlike --all, does not run any extra checks)')
    parser.add_argument('-a', '--all', action='store_true', help='run all available checks using default filenames')
    parser.add_argument('-l', '--list', action='store_true', help='list available generators and their default filenames; then exit')
    parser.add_argument('--dump', action='store_true', help='dump the manifest contents (used for debugging the parser)')
    actions_group = parser.add_argument_group('actions')
    for gen in generators:
        actions_group.add_argument(gen.arg_name, dest=gen.var_name, type=str, nargs='?', default=MISSING, metavar='FILENAME', help=gen.__doc__)
    actions_group.add_argument('--unixy-check', action='store_true', help=do_unixy_check.__doc__)
    args = parser.parse_args()
    base_path = args.file.parent.parent
    if args.list:
        for gen in generators:
            print(f'{gen.arg_name}: {base_path / gen.default_path}')
        sys.exit(0)
    run_all_generators = args.generate_all
    if args.generate_all:
        args.generate = True
    if args.all:
        run_all_generators = True
        args.unixy_check = True
    with args.file.open() as file:
        manifest = parse_manifest(file)
    check_private_names(manifest)
    results = {}
    if args.dump:
        for line in manifest.dump():
            print(line)
        results['dump'] = True
    for gen in generators:
        filename = getattr(args, gen.var_name)
        if filename is None or (run_all_generators and filename is MISSING):
            filename = base_path / gen.default_path
        elif filename is MISSING:
            continue
        results[gen.var_name] = generate_or_check(manifest, args, filename, gen)
    if args.unixy_check:
        results['unixy_check'] = do_unixy_check(manifest, args)
    if not results:
        if args.generate:
            parser.error('No file specified. Use --help for usage.')
        parser.error('No check specified. Use --help for usage.')
    failed_results = [name for (name, result) in results.items() if not result]
    if failed_results:
        raise Exception(f"\n        These checks related to the stable ABI did not succeed:\n            {', '.join(failed_results)}\n\n        If you see diffs in the output, files derived from the stable\n        ABI manifest the were not regenerated.\n        Run `make regen-limited-abi` to fix this.\n\n        Otherwise, see the error(s) above.\n\n        The stable ABI manifest is at: {args.file}\n        Note that there is a process to follow when modifying it.\n\n        You can read more about the limited API and its contracts at:\n\n        https://docs.python.org/3/c-api/stable.html\n\n        And in PEP 384:\n\n        https://www.python.org/dev/peps/pep-0384/\n        ")
if __name__ == '__main__':
    main()