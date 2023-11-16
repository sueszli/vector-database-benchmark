from __future__ import print_function
import os
import sys
import re
import threading
if sys.version_info.major == 2:
    import distutils.core as setuptools
else:
    import setuptools
android = 'RENPY_ANDROID' in os.environ
ios = 'RENPY_IOS' in os.environ
raspi = 'RENPY_RASPBERRY_PI' in os.environ
emscripten = 'RENPY_EMSCRIPTEN' in os.environ
coverage = 'RENPY_COVERAGE' in os.environ
static = 'RENPY_STATIC' in os.environ
gen = 'gen'
if sys.version_info.major > 2:
    gen += '3'
    PY2 = False
else:
    PY2 = True
if coverage:
    gen += '-coverage'
if static:
    gen += '-static'
cython_command = os.environ.get('RENPY_CYTHON', 'cython')
if not (android or ios):
    install = os.environ.get('RENPY_DEPS_INSTALL', '/usr')
    if '::' in install:
        install = install.split('::')
    else:
        install = install.split(os.pathsep)
    install = [os.path.abspath(i) for i in install]
    if 'VIRTUAL_ENV' in os.environ:
        install.insert(0, os.environ['VIRTUAL_ENV'])
else:
    install = []
include_dirs = ['.']
library_dirs = []
extra_compile_args = []
extra_link_args = []

def include(header, directory=None, optional=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    Searches the install paths for `header`. If `directory` is given, we\n    will append that to each of the install paths when trying to find\n    the header. The directory the header is found in is added to include_dirs\n    if it's not present already.\n\n    `optional`\n        If given, returns False rather than abandoning the process.\n    "
    if android or ios or emscripten:
        return True
    for i in install:
        if directory is not None:
            idir = os.path.join(i, 'include', directory)
        else:
            idir = os.path.join(i, 'include')
        fn = os.path.join(idir, header)
        if os.path.exists(fn):
            if idir not in include_dirs:
                include_dirs.append(idir)
            return True
    if optional:
        return False
    if directory is None:
        print('Could not find required header {0}.'.format(header))
    else:
        print('Could not find required header {0}/{1}.'.format(directory, header))
    sys.exit(-1)

def library(name, optional=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Searches for `library`.\n\n    `optional`\n        If true, this function will return False if a library is not found,\n        rather than reporting an error.\n    '
    if android or ios or emscripten:
        return True
    for i in install:
        for ldir in [i, os.path.join(i, 'lib'), os.path.join(i, 'lib64'), os.path.join(i, 'lib32')]:
            for suffix in ('.so', '.a', '.dll.a', '.dylib'):
                fn = os.path.join(ldir, 'lib' + name + suffix)
                if os.path.exists(fn):
                    if ldir not in library_dirs:
                        library_dirs.append(ldir)
                    return True
    if optional:
        return False
    print('Could not find required library {0}.'.format(name))
    sys.exit(-1)
extensions = []
global_macros = []

def cmodule(name, source, libs=[], define_macros=[], includes=[], language='c', compile_args=[]):
    if False:
        while True:
            i = 10
    '\n    Compiles the python module `name` from the files given in\n    `source`, and the libraries in `libs`.\n    '
    eca = list(extra_compile_args) + compile_args
    if language == 'c':
        eca.insert(0, '-std=gnu99')
    extensions.append(setuptools.Extension(name, source, include_dirs=include_dirs + includes, library_dirs=library_dirs, extra_compile_args=eca, extra_link_args=extra_link_args, libraries=libs, define_macros=define_macros + global_macros, language=language))
necessary_gen = []
generate_cython_queue = []

def cython(name, source=[], libs=[], includes=[], compile_if=True, define_macros=[], pyx=None, language='c', compile_args=[]):
    if False:
        while True:
            i = 10
    '\n    Compiles a cython module. This takes care of regenerating it as necessary\n    when it, or any of the files it depends on, changes.\n    '
    mod_coverage = coverage
    split_name = name.split('.')
    if pyx is not None:
        fn = pyx
    else:
        fn = '/'.join(split_name) + '.pyx'
    if os.path.exists(os.path.join('..', fn)):
        fn = os.path.join('..', fn)
    elif os.path.exists(fn):
        pass
    else:
        print('Could not find {0}.'.format(fn))
        sys.exit(-1)
    module_dir = os.path.dirname(fn)
    deps = [fn]
    with open(fn) as f:
        for l in f:
            m = re.search('from\\s*([\\w.]+)\\s*cimport', l)
            if m:
                deps.append(m.group(1).replace('.', '/') + '.pxd')
                continue
            m = re.search('cimport\\s*([\\w.]+)', l)
            if m:
                deps.append(m.group(1).replace('.', '/') + '.pxd')
                continue
            m = re.search('include\\s*"(.*?)"', l)
            if m:
                deps.append(m.group(1))
                continue
    deps = [i for i in deps if not i.startswith('cpython/') and (not i.startswith('libc/'))]
    if language == 'c++':
        c_fn = os.path.join(gen, name + '.cc')
        necessary_gen.append(name + '.cc')
    else:
        c_fn = os.path.join(gen, name + '.c')
        necessary_gen.append(name + '.c')
    if os.path.exists(c_fn):
        c_mtime = os.path.getmtime(c_fn)
    else:
        c_mtime = 0
    out_of_date = False
    for dep_fn in deps:
        if os.path.exists(os.path.join(module_dir, dep_fn)):
            dep_fn = os.path.join(module_dir, dep_fn)
        elif os.path.exists(os.path.join('..', dep_fn)):
            dep_fn = os.path.join('..', dep_fn)
        elif os.path.exists(os.path.join('include', dep_fn)):
            dep_fn = os.path.join('include', dep_fn)
        elif os.path.exists(os.path.join(gen, dep_fn)):
            dep_fn = os.path.join(gen, dep_fn)
        elif os.path.exists(dep_fn):
            pass
        else:
            print("{0} depends on {1}, which can't be found.".format(fn, dep_fn))
            sys.exit(-1)
        if os.path.getmtime(dep_fn) > c_mtime:
            out_of_date = True
    if out_of_date and (not cython_command):
        print('WARNING:', name, "is out of date, but RENPY_CYTHON isn't set.")
        out_of_date = False
    if out_of_date:
        print(name, 'is out of date.')
        generate_cython_queue.append((name, language, mod_coverage, split_name, fn, c_fn))
    if compile_if:
        if mod_coverage:
            define_macros = define_macros + [('CYTHON_TRACE', '1')]
        cmodule(name, [c_fn] + source, libs=libs, includes=includes, define_macros=define_macros, language=language, compile_args=compile_args)
lock = threading.Condition()
cython_failure = False

def generate_cython(name, language, mod_coverage, split_name, fn, c_fn):
    if False:
        for i in range(10):
            print('nop')
    import subprocess
    global cython_failure
    if language == 'c++':
        lang_args = ['--cplus']
    else:
        lang_args = []
    if 'RENPY_ANNOTATE_CYTHON' in os.environ:
        annotate = ['-a']
    else:
        annotate = []
    if mod_coverage:
        coverage_args = ['-X', 'linetrace=true']
    else:
        coverage_args = []
    p = subprocess.Popen([cython_command, '-Iinclude', '-I' + gen, '-I..', '--3str'] + annotate + lang_args + coverage_args + ['-X', 'profile=False', '-X', 'embedsignature=True', fn, '-o', c_fn], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (stdout, stderr) = p.communicate()
    with lock:
        print('-', name, '-' * (76 - len(name)))
        if stdout:
            print(stdout.decode('utf-8', 'surrogateescape'))
            print('')
    if p.returncode:
        cython_failure = True
        return
    if static:
        parent_module = '.'.join(split_name[:-1])
        parent_module_identifier = parent_module.replace('.', '_')
        with open(c_fn, 'r') as f:
            ccode = f.read()
        with open(c_fn + '.dynamic', 'w') as f:
            f.write(ccode)
        if len(split_name) > 1:
            ccode = re.sub('Py_InitModule4\\("([^"]+)"', 'Py_InitModule4("' + parent_module + '.\\1"', ccode)
            ccode = re.sub('(__pyx_moduledef.*?"){}"'.format(re.escape(split_name[-1])), '\\1' + '.'.join(split_name) + '"', ccode, count=1, flags=re.DOTALL)
            ccode = re.sub('^__Pyx_PyMODINIT_FUNC init', '__Pyx_PyMODINIT_FUNC init' + parent_module_identifier + '_', ccode, 0, re.MULTILINE)
            ccode = re.sub('^__Pyx_PyMODINIT_FUNC PyInit_', '__Pyx_PyMODINIT_FUNC PyInit_' + parent_module_identifier + '_', ccode, 0, re.MULTILINE)
            ccode = re.sub('^PyMODINIT_FUNC init', 'PyMODINIT_FUNC init' + parent_module_identifier + '_', ccode, 0, re.MULTILINE)
        with open(c_fn, 'w') as f:
            f.write(ccode)

def generate_all_cython():
    if False:
        for i in range(10):
            print('nop')
    '\n    Run all of the cython that needs to be generated.\n    '
    threads = []
    for args in generate_cython_queue:
        if 'RENPY_CYTHON_SINGLETHREAD' in os.environ:
            generate_cython(*args)
            if cython_failure:
                sys.exit(1)
        else:
            t = threading.Thread(target=generate_cython, args=args)
            t.start()
            threads.append(t)
    for t in threads:
        t.join()
    if cython_failure:
        sys.exit(1)

def find_unnecessary_gen():
    if False:
        for i in range(10):
            print('nop')
    for i in os.listdir(gen):
        if not i.endswith('.c'):
            continue
        if i in necessary_gen:
            continue
        print('Unnecessary file', os.path.join(gen, i))
py_modules = []

def pymodule(name):
    if False:
        return 10
    '\n    Causes a python module to be included in the build.\n    '
    py_modules.append(name)

def copyfile(source, dest, replace=None, replace_with=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Copy `source` to `dest`, preserving the modification time.\n\n    If `replace` is given, instances of `replace` in the file contents are\n    replaced with `replace_with`.\n    '
    sfn = os.path.join('..', source)
    dfn = os.path.join('..', dest)
    if os.path.exists(dfn):
        if os.path.getmtime(sfn) <= os.path.getmtime(dfn):
            return
    with open(sfn, 'r') as sf:
        data = sf.read()
    if replace and replace_with is not None:
        data = data.replace(replace, replace_with)
    with open(dfn, 'w') as df:
        df.write('# This file was automatically generated from ' + source + '\n')
        df.write('# Modifications will be automatically overwritten.\n\n')
        df.write(data)
    import shutil
    shutil.copystat(sfn, dfn)

def setup(name, version):
    if False:
        print('Hello World!')
    '\n    Calls the distutils setup function.\n    '
    if len(sys.argv) >= 2 and sys.argv[1] == 'generate':
        return
    setuptools.setup(name=name, version=version, ext_modules=extensions, py_modules=py_modules, zip_safe=False)
if not os.path.exists(gen):
    os.mkdir(gen)