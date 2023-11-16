"""
test_pythonlib.py -- compile, uncompyle, and verify Python libraries

Usage-Examples:

  # decompile, and verify base set of python 2.7 byte-compiled files
  test_pythonlib.py --base-2.7 --verify

  # Same as above but compile the base set first
  test_pythonlib.py --base-2.7 --verify --compile

  # Same as above but use a longer set from the python 2.7 library
  test_pythonlib.py --ok-2.7 --verify --compile

  # Just deompile the longer set of files
  test_pythonlib.py --ok-2.7

Adding own test-trees:

Step 1) Edit this file and add a new entry to 'test_options', eg.
  test_options['mylib'] = ('/usr/lib/mylib', PYOC, 'mylib')
Step 2: Run the test:
  test_pythonlib.py --mylib	  # decompile 'mylib'
  test_pythonlib.py --mylib --verify # decompile verify 'mylib'
"""
from __future__ import print_function
import getopt
import os
import py_compile
import shutil
import sys
import tempfile
import time
from fnmatch import fnmatch
from xdis.version_info import PYTHON_VERSION_TRIPLE
from uncompyle6.main import main

def get_srcdir():
    if False:
        i = 10
        return i + 15
    filename = os.path.normcase(os.path.dirname(__file__))
    return os.path.realpath(filename)
src_dir = get_srcdir()
lib_prefix = '/usr/lib'
target_base = tempfile.mkdtemp(prefix='py-dis-')
PY = ('*.py',)
PYC = ('*.pyc',)
PYO = ('*.pyo',)
PYOC = ('*.pyc', '*.pyo')
test_options = {'test': ('test', PYC, 'test'), 'ok-2.6': (os.path.join(src_dir, 'ok_lib2.6'), PYOC, 'ok-2.6', 2.6), 'ok-2.7': (os.path.join(src_dir, 'ok_lib2.7'), PYOC, 'ok-2.7', 2.7), 'ok-3.2': (os.path.join(src_dir, 'ok_lib3.2'), PYOC, 'ok-3.2', 3.2), 'base-2.7': (os.path.join(src_dir, 'base_tests', 'python2.7'), PYOC, 'base_2.7', 2.7)}
for vers in (2.7, 3.4, 3.5, 3.6):
    pythonlib = 'ok_lib%s' % vers
    key = 'ok-%s' % vers
    test_options[key] = (os.path.join(src_dir, pythonlib), PYOC, key, vers)
    pass
for vers in (1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 'pypy3.2', 'pypy2.7', 'pypy3.6'):
    bytecode = 'bytecode_%s' % vers
    key = 'bytecode-%s' % vers
    test_options[key] = (bytecode, PYC, bytecode, vers)
    bytecode = 'bytecode_%s_run' % vers
    key = 'bytecode-%s-run' % vers
    test_options[key] = (bytecode, PYC, bytecode, vers)
    key = '%s' % vers
    pythonlib = 'python%s' % vers
    if isinstance(vers, float) and vers >= 3.0:
        pythonlib = os.path.join(pythonlib, '__pycache__')
    test_options[key] = (os.path.join(lib_prefix, pythonlib), PYOC, pythonlib, vers)
for (vers, vers_dot) in ((37, 3.7), (38, 3.8)):
    bytecode = 'bytecode_pypy%s_run' % vers
    key = 'bytecode-pypy%s' % vers
    test_options[key] = (bytecode, PYC, bytecode, vers_dot)
    key = 'bytecode-pypy%s' % vers_dot
    test_options[key] = (bytecode, PYC, bytecode, vers_dot)

def help():
    if False:
        while True:
            i = 10
    print("Usage-Examples:\n\n  # compile, decompyle and verify short tests for Python 2.7:\n  test_pythonlib.py --bytecode-2.7 --verify --compile\n\n  # decompile all of Python's installed lib files\n  test_pythonlib.py --2.7\n\n  # decompile and verify known good python 2.7\n  test_pythonlib.py --ok-2.7 --verify\n")
    sys.exit(1)

def do_tests(src_dir, obj_patterns, target_dir, opts):
    if False:
        return 10

    def file_matches(files, root, basenames, patterns):
        if False:
            while True:
                i = 10
        files.extend([os.path.normpath(os.path.join(root, n)) for n in basenames for pat in patterns if fnmatch(n, pat)])
    files = []
    cwd = os.getcwd()
    os.chdir(src_dir)
    if opts['do_compile']:
        compiled_version = opts['compiled_version']
        if compiled_version and PYTHON_VERSION_TRIPLE != compiled_version:
            print('Not compiling: desired Python version is %s but we are running %s' % (compiled_version, PYTHON_VERSION_TRIPLE), file=sys.stderr)
        else:
            for (root, dirs, basenames) in os.walk(src_dir):
                file_matches(files, root, basenames, PY)
                for sfile in files:
                    py_compile.compile(sfile)
                    pass
                pass
            files = []
            pass
        pass
    for (root, dirs, basenames) in os.walk('.'):
        dirname = root[2:]
        file_matches(files, dirname, basenames, obj_patterns)
    if not files:
        print("Didn't come up with any files to test! Try with --compile?", file=sys.stderr)
        exit(1)
    os.chdir(cwd)
    files.sort()
    if opts['start_with']:
        try:
            start_with = files.index(opts['start_with'])
            files = files[start_with:]
            print('>>> starting with file', files[0])
        except ValueError:
            pass
    print(time.ctime())
    print('Source directory: ', src_dir)
    print('Output directory: ', target_dir)
    try:
        (_, _, failed_files, failed_verify) = main(src_dir, target_dir, files, [], do_verify=opts['do_verify'])
        if failed_files != 0:
            sys.exit(2)
        elif failed_verify != 0:
            sys.exit(3)
    except (KeyboardInterrupt, OSError):
        print()
        sys.exit(1)
    if test_opts['rmtree']:
        parent_dir = os.path.dirname(target_dir)
        print('Everything good, removing %s' % parent_dir)
        shutil.rmtree(parent_dir)
if __name__ == '__main__':
    test_dirs = []
    checked_dirs = []
    start_with = None
    test_options_keys = list(test_options.keys())
    test_options_keys.sort()
    (opts, args) = getopt.getopt(sys.argv[1:], '', ['start-with=', 'verify', 'verify-run', 'syntax-verify', 'all', 'compile', 'coverage', 'no-rm'] + test_options_keys)
    if not opts:
        help()
    test_opts = {'do_compile': False, 'do_verify': False, 'start_with': None, 'rmtree': True, 'coverage': False}
    for (opt, val) in opts:
        if opt == '--verify':
            test_opts['do_verify'] = 'strong'
        elif opt == '--syntax-verify':
            test_opts['do_verify'] = 'weak'
        elif opt == '--verify-run':
            test_opts['do_verify'] = 'verify-run'
        elif opt == '--compile':
            test_opts['do_compile'] = True
        elif opt == '--start-with':
            test_opts['start_with'] = val
        elif opt == '--no-rm':
            test_opts['rmtree'] = False
        elif opt[2:] in test_options_keys:
            test_dirs.append(test_options[opt[2:]])
        elif opt == '--all':
            for val in test_options_keys:
                test_dirs.append(test_options[val])
        elif opt == '--coverage':
            test_opts['coverage'] = True
        else:
            help()
            pass
        pass
    if test_opts['coverage']:
        os.environ['SPARK_PARSER_COVERAGE'] = '/tmp/spark-grammar-python-lib%s.cover' % test_dirs[0][-1]
    last_compile_version = None
    for (src_dir, pattern, target_dir, compiled_version) in test_dirs:
        if os.path.isdir(src_dir):
            checked_dirs.append([src_dir, pattern, target_dir])
        else:
            print("Can't find directory %s. Skipping" % src_dir, file=sys.stderr)
            continue
        last_compile_version = compiled_version
        pass
    if not checked_dirs:
        print('No directories found to check', file=sys.stderr)
        sys.exit(1)
    test_opts['compiled_version'] = last_compile_version
    for (src_dir, pattern, target_dir) in checked_dirs:
        target_dir = os.path.join(target_base, target_dir)
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir, ignore_errors=1)
        do_tests(src_dir, pattern, target_dir, test_opts)