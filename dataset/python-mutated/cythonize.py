"""
Runs Cython on all modules that were listed via add_cython_module.
"""
import argparse
import os
import sys
from contextlib import redirect_stdout
from multiprocessing import cpu_count
from pathlib import Path
from Cython.Build import cythonize

class LineFilter:
    """ Proxy for a stream (default stdout) to filter out whole unwanted lines """

    def __init__(self, stream=None, filters=None):
        if False:
            print('Hello World!')
        self.stream = stream or sys.stdout
        self.filters = filters or []
        self.buf = ''

    def __getattr__(self, attr_name):
        if False:
            print('Hello World!')
        return getattr(self.stream, attr_name)

    def __enter__(self):
        if False:
            return 10
        return self

    def __exit__(self, *args):
        if False:
            print('Hello World!')
        self.stream.write(self.buf)
        self.buf = ''

    def write(self, data):
        if False:
            i = 10
            return i + 15
        '\n        Writes to output stream, buffered line-wise,\n        omitting lines given in to constructor in filter\n        '
        self.buf += data
        lines = self.buf.split('\n')
        for line in lines[:-1]:
            if not any((f(line) for f in self.filters)):
                self.stream.write(line + '\n')
        self.buf = lines[-1]

class CythonFilter(LineFilter):
    """ Filters output of cythonize for useless warnings """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        filters = [lambda x: 'put "# distutils: language=c++" in your .pyx or .pxd file(s)' in x, lambda x: x.startswith('Compiling ') and x.endswith(' because it changed.')]
        super().__init__(filters=filters)

def read_list_from_file(filename):
    if False:
        print('Hello World!')
    ' Reads a semicolon-separated list of file entires '
    with open(filename, encoding='utf8') as fileobj:
        data = fileobj.read().strip()
    return [Path(filename).resolve() for filename in data.split(';')]

def remove_if_exists(filename):
    if False:
        for i in range(10):
            print('nop')
    ' Deletes the file (if it exists) '
    if filename.is_file():
        print(filename.relative_to(os.getcwd()))
        filename.unlink()

def cythonize_wrapper(modules, **kwargs):
    if False:
        while True:
            i = 10
    ' Calls cythonize, filtering useless warnings '
    (bin_dir, bin_modules) = (kwargs['build_dir'], [])
    (src_dir, src_modules) = (Path.cwd(), [])
    for module in modules:
        if Path(bin_dir) in module.parents:
            bin_modules.append(str(module.relative_to(bin_dir)))
        else:
            src_modules.append(str(module.relative_to(src_dir)))
    with CythonFilter() as cython_filter:
        with redirect_stdout(cython_filter):
            if src_modules:
                cythonize(src_modules, **kwargs)
            if bin_modules:
                os.chdir(bin_dir)
                cythonize(bin_modules, **kwargs)
                os.chdir(src_dir)

def main():
    if False:
        print('Hello World!')
    ' CLI entry point '
    cli = argparse.ArgumentParser()
    cli.add_argument('module_list', help='Module list file (semicolon-separated).')
    cli.add_argument('embedded_module_list', help='Embedded module list file (semicolon-separated).\nModules in this list are compiled with the --embed option.')
    cli.add_argument('depends_list', help='Dependency list file (semicolon-separated).\nContains all .pxd and other files that may get included.\nUsed to verify that all dependencies are properly listed in the CMake build configuration.')
    cli.add_argument('--clean', action='store_true', help='Clean compilation results and exit.')
    cli.add_argument('--build-dir', help='Build output directory to generate the cpp files in.note: this is also added for module search path.')
    cli.add_argument('--memcleanup', type=int, default=0, help='Generate memory cleanup code to make valgrind happy:\n0: nothing, 1+: interned objects,\n2+: cdef globals, 3+: types objects')
    cli.add_argument('--threads', type=int, default=cpu_count(), help='number of compilation threads to use')
    args = cli.parse_args()
    modules = read_list_from_file(args.module_list)
    embedded_modules = read_list_from_file(args.embedded_module_list)
    depends = set(read_list_from_file(args.depends_list))
    if args.clean:
        for module in modules + embedded_modules:
            rel_module = module.relative_to(Path.cwd())
            build_module = args.build_dir / rel_module
            remove_if_exists(build_module.with_suffix('.cpp'))
            remove_if_exists(build_module.with_suffix('.html'))
        sys.exit(0)
    from Cython.Compiler import Options
    Options.annotate = True
    Options.fast_fail = True
    Options.generate_cleanup_code = args.memcleanup
    Options.cplus = 1
    cythonize_args = {'compiler_directives': {'language_level': 3}, 'build_dir': args.build_dir, 'include_path': [args.build_dir], 'nthreads': args.threads}
    cythonize_args['language'] = 'c++'
    cythonize_wrapper(modules, **cythonize_args)
    Options.embed = 'main'
    cythonize_wrapper(embedded_modules, **cythonize_args)
    from Cython.Build.Dependencies import _dep_tree
    depend_failed = False
    for (module, files) in _dep_tree.__cimported_files_cache.items():
        for filename in files:
            if not filename.startswith('.'):
                continue
            if os.path.realpath(os.path.abspath(filename)) not in depends:
                print('\x1b[31mERR\x1b[m unlisted dependency: ' + filename)
                depend_failed = True
    if depend_failed:
        sys.exit(1)
if __name__ == '__main__':
    main()