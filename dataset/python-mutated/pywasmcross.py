"""Helper for cross-compiling distutils-based Python extensions.

distutils has never had a proper cross-compilation story. This is a hack, which
miraculously works, to get around that.

The gist is we compile the package replacing calls to the compiler and linker
with wrappers that adjusting include paths and flags as necessary for
cross-compiling and then pass the command long to emscripten.
"""
import json
import os
import sys
from pathlib import Path
from __main__ import __file__ as INVOKED_PATH_STR
INVOKED_PATH = Path(INVOKED_PATH_STR)
SYMLINKS = {'cc', 'c++', 'ld', 'lld', 'ar', 'gcc', 'ranlib', 'strip', 'gfortran', 'cargo', 'cmake', 'meson', 'install_name_tool', 'otool'}
IS_COMPILER_INVOCATION = INVOKED_PATH.name in SYMLINKS
if IS_COMPILER_INVOCATION:
    if 'PYWASMCROSS_ARGS' in os.environ:
        PYWASMCROSS_ARGS = json.loads(os.environ['PYWASMCROSS_ARGS'])
    else:
        try:
            with open(INVOKED_PATH.parent / 'pywasmcross_env.json') as f:
                PYWASMCROSS_ARGS = json.load(f)
        except FileNotFoundError:
            raise RuntimeError(f"Invalid invocation: can't find PYWASMCROSS_ARGS. Invoked from {INVOKED_PATH}.") from None
    sys.path = PYWASMCROSS_ARGS.pop('PYTHONPATH')
    os.environ['PATH'] = os.environ['BUILD_ENV_SCRIPTS_DIR'] + ':' + PYWASMCROSS_ARGS.pop('PATH')
    __name__ = PYWASMCROSS_ARGS.pop('orig__name__')
import dataclasses
import re
import shutil
import subprocess
from collections.abc import Iterable, Iterator
from typing import Literal, NoReturn

@dataclasses.dataclass(eq=False, order=False, kw_only=True)
class BuildArgs:
    """
    Common arguments for building a package.
    """
    pkgname: str = ''
    cflags: str = ''
    cxxflags: str = ''
    ldflags: str = ''
    target_install_dir: str = ''
    host_install_dir: str = ''
    builddir: str = ''
    pythoninclude: str = ''
    exports: Literal['whole_archive', 'requested', 'pyinit'] | list[str] = 'pyinit'
    compression_level: int = 6

def replay_f2c(args: list[str], dryrun: bool=False) -> list[str] | None:
    if False:
        return 10
    "Apply f2c to compilation arguments\n\n    Parameters\n    ----------\n    args\n       input compiler arguments\n    dryrun\n       if False run f2c on detected fortran files\n\n    Returns\n    -------\n    new_args\n       output compiler arguments\n\n\n    Examples\n    --------\n\n    >>> replay_f2c(['gfortran', 'test.f'], dryrun=True)\n    ['gcc', 'test.c']\n    "
    from pyodide_build._f2c_fixes import fix_f2c_input, fix_f2c_output
    new_args = ['gcc']
    found_source = False
    for arg in args[1:]:
        if arg.endswith('.f') or arg.endswith('.F'):
            filepath = Path(arg).resolve()
            if not dryrun:
                fix_f2c_input(arg)
                if arg.endswith('.F'):
                    subprocess.check_call(['gfortran', '-E', '-C', '-P', filepath, '-o', filepath.with_suffix('.f77')])
                    filepath = filepath.with_suffix('.f77')
                with open(filepath) as input_pipe, open(filepath.with_suffix('.c'), 'w') as output_pipe:
                    subprocess.check_call(['f2c', '-R'], stdin=input_pipe, stdout=output_pipe, cwd=filepath.parent)
                fix_f2c_output(arg[:-2] + '.c')
            new_args.append(arg[:-2] + '.c')
            found_source = True
        else:
            new_args.append(arg)
    new_args_str = ' '.join(args)
    if '.so' in new_args_str and 'libgfortran.so' not in new_args_str:
        found_source = True
    if not found_source:
        print(f'f2c: source not found, skipping: {new_args_str}')
        return None
    return new_args

def get_library_output(line: list[str]) -> str | None:
    if False:
        print('Hello World!')
    '\n    Check if the command is a linker invocation. If so, return the name of the\n    output file.\n    '
    SHAREDLIB_REGEX = re.compile('\\.so(.\\d+)*$')
    for arg in line:
        if not arg.startswith('-') and SHAREDLIB_REGEX.search(arg):
            return arg
    return None

def replay_genargs_handle_dashl(arg: str, used_libs: set[str]) -> str | None:
    if False:
        for i in range(10):
            print('nop')
    "\n    Figure out how to replace a `-lsomelib` argument.\n\n    Parameters\n    ----------\n    arg\n        The argument we are replacing. Must start with `-l`.\n\n    used_libs\n        The libraries we've used so far in this command. emcc fails out if `-lsomelib`\n        occurs twice, so we have to track this.\n\n    Returns\n    -------\n        The new argument, or None to delete the argument.\n    "
    assert arg.startswith('-l')
    if arg == '-lffi':
        return None
    if arg == '-lgfortran':
        return None
    if arg in used_libs:
        return None
    used_libs.add(arg)
    return arg

def replay_genargs_handle_dashI(arg: str, target_install_dir: str) -> str | None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Figure out how to replace a `-Iincludepath` argument.\n\n    Parameters\n    ----------\n    arg\n        The argument we are replacing. Must start with `-I`.\n\n    target_install_dir\n        The target_install_dir argument.\n\n    Returns\n    -------\n        The new argument, or None to delete the argument.\n    '
    assert arg.startswith('-I')
    if arg[2:].startswith('/usr'):
        return None
    include_path = str(Path(arg[2:]).resolve())
    if include_path.startswith(sys.prefix + '/include/python'):
        return arg.replace('-I' + sys.prefix, '-I' + target_install_dir)
    if include_path.startswith(sys.base_prefix + '/include/python'):
        return arg.replace('-I' + sys.base_prefix, '-I' + target_install_dir)
    return arg

def replay_genargs_handle_linker_opts(arg: str) -> str | None:
    if False:
        return 10
    '\n    ignore some link flags\n    it should not check if `arg == "-Wl,-xxx"` and ignore directly here,\n    because arg may be something like "-Wl,-xxx,-yyy" where we only want\n    to ignore "-xxx" but not "-yyy".\n    '
    assert arg.startswith('-Wl')
    link_opts = arg.split(',')[1:]
    new_link_opts = ['-Wl']
    for opt in link_opts:
        if opt in ['-Bsymbolic-functions', '--strip-all', '-strip-all', '--sort-common', '--as-needed']:
            continue
        if opt.startswith(('--sysroot=', '--version-script=', '-R/', '-R.', '--exclude-libs=')):
            continue
        new_link_opts.append(opt)
    if len(new_link_opts) > 1:
        return ','.join(new_link_opts)
    else:
        return None

def replay_genargs_handle_argument(arg: str) -> str | None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Figure out how to replace a general argument.\n\n    Parameters\n    ----------\n    arg\n        The argument we are replacing. Must not start with `-I` or `-l`.\n\n    Returns\n    -------\n        The new argument, or None to delete the argument.\n    '
    assert not arg.startswith('-I')
    assert not arg.startswith('-l')
    assert not arg.startswith('-Wl,')
    if arg.startswith('-L/usr'):
        return None
    if arg in ['-pthread', '-ffixed-form', '-fallow-argument-mismatch', '-bundle', '-undefined', 'dynamic_lookup', '-mpopcnt', '-Bsymbolic-functions', '-fno-second-underscore', '-fstack-protector', '-fno-strict-overflow', '-mno-sse2', '-mno-avx2']:
        return None
    return arg

def get_cmake_compiler_flags() -> list[str]:
    if False:
        print('Hello World!')
    '\n    GeneraTe cmake compiler flags.\n    emcmake will set these values to emcc, em++, ...\n    but we need to set them to cc, c++, in order to make them pass to pywasmcross.\n    Returns\n    -------\n    The commandline flags to pass to cmake.\n    '
    compiler_flags = {'CMAKE_C_COMPILER': 'cc', 'CMAKE_CXX_COMPILER': 'c++', 'CMAKE_AR': 'ar', 'CMAKE_C_COMPILER_AR': 'ar', 'CMAKE_CXX_COMPILER_AR': 'ar'}
    flags = []
    symlinks_dir = Path(sys.argv[0]).parent
    for (key, value) in compiler_flags.items():
        assert value in SYMLINKS
        flags.append(f'-D{key}={symlinks_dir / value}')
    return flags

def _calculate_object_exports_readobj_parse(output: str) -> list[str]:
    if False:
        return 10
    "\n    >>> _calculate_object_exports_readobj_parse(\n    ...     '''\n    ...     Format: WASM \\n Arch: wasm32 \\n AddressSize: 32bit\n    ...     Sections [\n    ...         Section { \\n Type: TYPE (0x1)   \\n Size: 5  \\n Offset: 8  \\n }\n    ...         Section { \\n Type: IMPORT (0x2) \\n Size: 32 \\n Offset: 19 \\n }\n    ...     ]\n    ...     Symbol {\n    ...         Name: g2 \\n Type: FUNCTION (0x0) \\n\n    ...         Flags [ (0x0) \\n ]\n    ...         ElementIndex: 0x2\n    ...     }\n    ...     Symbol {\n    ...         Name: f2 \\n Type: FUNCTION (0x0) \\n\n    ...         Flags [ (0x4) \\n VISIBILITY_HIDDEN (0x4) \\n ]\n    ...         ElementIndex: 0x1\n    ...     }\n    ...     Symbol {\n    ...         Name: l  \\n Type: FUNCTION (0x0)\n    ...         Flags [ (0x10)\\n UNDEFINED (0x10) \\n ]\n    ...         ImportModule: env\n    ...         ElementIndex: 0x0\n    ...     }\n    ...     '''\n    ... )\n    ['g2']\n    "
    result = []
    insymbol = False
    for line in output.split('\n'):
        line = line.strip()
        if line == 'Symbol {':
            insymbol = True
            export = True
            name = None
            symbol_lines = [line]
            continue
        if not insymbol:
            continue
        symbol_lines.append(line)
        if line.startswith('Name:'):
            name = line.removeprefix('Name:').strip()
        if line.startswith(('BINDING_LOCAL', 'UNDEFINED', 'VISIBILITY_HIDDEN')):
            export = False
        if line == '}':
            insymbol = False
            if export:
                if not name:
                    raise RuntimeError("Didn't find symbol's name:\n" + '\n'.join(symbol_lines))
                result.append(name)
    return result

def calculate_object_exports_readobj(objects: list[str]) -> list[str] | None:
    if False:
        return 10
    readobj_path = shutil.which('llvm-readobj')
    if not readobj_path:
        which_emcc = shutil.which('emcc')
        assert which_emcc
        emcc = Path(which_emcc)
        readobj_path = str((emcc / '../../bin/llvm-readobj').resolve())
    args = [readobj_path, '--section-details', '-st'] + objects
    completedprocess = subprocess.run(args, encoding='utf8', capture_output=True, env={'PATH': os.environ['PATH']})
    if completedprocess.returncode:
        print(f"Command '{' '.join(args)}' failed. Output to stderr was:")
        print(completedprocess.stderr)
        sys.exit(completedprocess.returncode)
    if 'bitcode files are not supported' in completedprocess.stderr:
        return None
    return _calculate_object_exports_readobj_parse(completedprocess.stdout)

def calculate_object_exports_nm(objects: list[str]) -> list[str]:
    if False:
        for i in range(10):
            print('nop')
    args = ['emnm', '-j', '--export-symbols'] + objects
    result = subprocess.run(args, encoding='utf8', capture_output=True, env={'PATH': os.environ['PATH']})
    if result.returncode:
        print(f"Command '{' '.join(args)}' failed. Output to stderr was:")
        print(result.stderr)
        sys.exit(result.returncode)
    return result.stdout.splitlines()

def filter_objects(line: list[str]) -> list[str]:
    if False:
        i = 10
        return i + 15
    '\n    Collect up all the object files and archive files being linked.\n    '
    return [arg for arg in line if arg.endswith(('.a', '.o')) or arg.startswith('@')]

def calculate_exports(line: list[str], export_all: bool) -> Iterable[str]:
    if False:
        i = 10
        return i + 15
    '\n    List out symbols from object files and archive files that are marked as public.\n    If ``export_all`` is ``True``, then return all public symbols.\n    If not, return only the public symbols that begin with `PyInit`.\n    '
    objects = filter_objects(line)
    exports = None
    if export_all:
        exports = calculate_object_exports_readobj(objects)
    if exports is None:
        exports = calculate_object_exports_nm(objects)
    if export_all:
        return exports
    return (x for x in exports if x.startswith('PyInit'))

def get_export_flags(line: list[str], exports: Literal['whole_archive', 'requested', 'pyinit'] | list[str]) -> Iterator[str]:
    if False:
        print('Hello World!')
    '\n    If "whole_archive" was requested, no action is needed. Otherwise, add\n    `-sSIDE_MODULE=2` and the appropriate export list.\n    '
    if exports == 'whole_archive':
        return
    yield '-sSIDE_MODULE=2'
    if isinstance(exports, str):
        export_list = calculate_exports(line, exports == 'requested')
    else:
        export_list = exports
    prefixed_exports = ['_' + x for x in export_list]
    yield f'-sEXPORTED_FUNCTIONS={prefixed_exports!r}'

def handle_command_generate_args(line: list[str], build_args: BuildArgs, is_link_command: bool) -> list[str]:
    if False:
        print('Hello World!')
    '\n    A helper command for `handle_command` that generates the new arguments for\n    the compilation.\n\n    Unlike `handle_command` this avoids I/O: it doesn\'t sys.exit, it doesn\'t run\n    subprocesses, it doesn\'t create any files, and it doesn\'t write to stdout.\n\n    Parameters\n    ----------\n    line The original compilation command as a list e.g., ["gcc", "-c",\n        "input.c", "-o", "output.c"]\n\n    build_args The arguments that pywasmcross was invoked with\n\n    is_link_command Is this a linker invocation?\n\n    Returns\n    -------\n        An updated argument list suitable for use with emscripten.\n\n\n    Examples\n    --------\n\n    >>> from collections import namedtuple\n    >>> Args = namedtuple(\'args\', [\'cflags\', \'cxxflags\', \'ldflags\', \'target_install_dir\'])\n    >>> args = Args(cflags=\'\', cxxflags=\'\', ldflags=\'\', target_install_dir=\'\')\n    >>> handle_command_generate_args([\'gcc\', \'test.c\'], args, False)\n    [\'emcc\', \'test.c\', \'-Werror=implicit-function-declaration\', \'-Werror=mismatched-parameter-types\', \'-Werror=return-type\']\n    '
    if '-print-multiarch' in line:
        return ['echo', 'wasm32-emscripten']
    for arg in line:
        if arg.startswith('-print-file-name'):
            return line
    if len(line) == 2 and line[1] == '-v':
        return ['emcc', '-v']
    cmd = line[0]
    if cmd == 'ar':
        line[0] = 'emar'
        return line
    elif cmd == 'c++' or cmd == 'g++':
        new_args = ['em++']
    elif cmd in ('cc', 'gcc', 'ld', 'lld'):
        new_args = ['emcc']
        if any((arg.endswith(('.cpp', '.cc')) for arg in line)):
            new_args = ['em++']
    elif cmd == 'cmake':
        if '--build' in line or '--install' in line or '-P' in line:
            return line
        flags = get_cmake_compiler_flags()
        line[:1] = ['emcmake', 'cmake', *flags, '--fresh']
        return line
    elif cmd == 'meson':
        if line[:2] != ['meson', 'setup']:
            return line
        if 'MESON_CROSS_FILE' in os.environ:
            line[:2] = ['meson', 'setup', '--cross-file', os.environ['MESON_CROSS_FILE']]
        return line
    elif cmd in ('install_name_tool', 'otool'):
        return ['echo', *line]
    elif cmd == 'ranlib':
        line[0] = 'emranlib'
        return line
    elif cmd == 'strip':
        line[0] = 'emstrip'
        return line
    else:
        return line
    used_libs: set[str] = set()
    for arg in line[1:]:
        if new_args[-1].startswith('-B') and 'compiler_compat' in arg:
            del new_args[-1]
            continue
        if arg.startswith('-l'):
            result = replay_genargs_handle_dashl(arg, used_libs)
        elif arg.startswith('-I'):
            result = replay_genargs_handle_dashI(arg, build_args.target_install_dir)
        elif arg.startswith('-Wl'):
            result = replay_genargs_handle_linker_opts(arg)
        else:
            result = replay_genargs_handle_argument(arg)
        if result:
            new_args.append(result)
    new_args.extend(['-Werror=implicit-function-declaration', '-Werror=mismatched-parameter-types', '-Werror=return-type'])
    if is_link_command:
        new_args.append('-Wl,--fatal-warnings')
        new_args.extend(build_args.ldflags.split())
        new_args.extend(get_export_flags(line, build_args.exports))
    if '-c' in line:
        if new_args[0] == 'emcc':
            new_args.extend(build_args.cflags.split())
        elif new_args[0] == 'em++':
            new_args.extend(build_args.cflags.split() + build_args.cxxflags.split())
        if build_args.pythoninclude:
            new_args.extend(['-I', build_args.pythoninclude])
    return new_args

def handle_command(line: list[str], build_args: BuildArgs) -> NoReturn:
    if False:
        while True:
            i = 10
    'Handle a compilation command. Exit with an appropriate exit code when done.\n\n    Parameters\n    ----------\n    line : iterable\n       an iterable with the compilation arguments\n    build_args : BuildArgs\n       a container with additional compilation options\n    '
    is_link_cmd = get_library_output(line) is not None
    if line[0] == 'gfortran':
        if '-dumpversion' in line:
            sys.exit(subprocess.run(line).returncode)
        tmp = replay_f2c(line)
        if tmp is None:
            sys.exit(0)
        line = tmp
    new_args = handle_command_generate_args(line, build_args, is_link_cmd)
    if build_args.pkgname == 'scipy':
        from pyodide_build._f2c_fixes import scipy_fixes
        scipy_fixes(new_args)
    returncode = subprocess.run(new_args).returncode
    sys.exit(returncode)

def compiler_main():
    if False:
        print('Hello World!')
    build_args = BuildArgs(**PYWASMCROSS_ARGS)
    basename = Path(sys.argv[0]).name
    args = list(sys.argv)
    args[0] = basename
    sys.exit(handle_command(args, build_args))
if IS_COMPILER_INVOCATION:
    compiler_main()