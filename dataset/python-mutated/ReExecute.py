"""Ability to restart Nuitka, needed for removing site module effects and using Python PGO after compile.

Note: This avoids imports at all costs, such that initial startup doesn't do more
than necessary.

spell-checker: ignore execl, Popen
"""
import os
import sys

def callExecProcess(args):
    if False:
        return 10
    'Do exec in a portable way preserving exit code.\n\n    On Windows, unfortunately there is no real exec, so we have to spawn\n    a new process instead.\n    '
    sys.stdout.flush()
    sys.stderr.flush()
    if os.name == 'nt':
        import subprocess
        args = list(args)
        del args[1]
        try:
            process = subprocess.Popen(args=args)
            process.communicate()
            try:
                os._exit(process.returncode)
            except OverflowError:
                os._exit(process.returncode - 2 ** 32)
        except KeyboardInterrupt:
            os._exit(2)
    else:
        os.execl(*args)

def reExecuteNuitka(pgo_filename):
    if False:
        i = 10
        return i + 15
    args = [sys.executable, sys.executable]
    if sys.version_info >= (3, 7) and sys.flags.utf8_mode:
        args += ['-X', 'utf8']
    if sys.version_info >= (3, 11):
        args += ['-X', 'frozen_modules=off']
    if 'nuitka.__main__' in sys.modules:
        our_filename = sys.modules['nuitka.__main__'].__file__
    else:
        our_filename = sys.modules['__main__'].__file__
    args += ['-S', our_filename]
    os.environ['NUITKA_BINARY_NAME'] = sys.modules['__main__'].__file__
    os.environ['NUITKA_PACKAGE_HOME'] = os.path.dirname(os.path.abspath(sys.modules['nuitka'].__path__[0]))
    if pgo_filename is not None:
        args.append('--pgo-python-input=%s' % pgo_filename)
    else:
        os.environ['NUITKA_SYS_PREFIX'] = sys.prefix
    args += sys.argv[1:]
    from nuitka.importing.PreloadedPackages import detectPreLoadedPackagePaths, detectPthImportedPackages
    os.environ['NUITKA_NAMESPACES'] = repr(detectPreLoadedPackagePaths())
    if 'site' in sys.modules:
        site_filename = sys.modules['site'].__file__
        if site_filename.endswith('.pyc'):
            site_filename = site_filename[:-4] + '.py'
        os.environ['NUITKA_SITE_FILENAME'] = site_filename
        os.environ['NUITKA_PTH_IMPORTED'] = repr(detectPthImportedPackages())
    os.environ['NUITKA_PYTHONPATH'] = repr(sys.path)
    import ast
    os.environ['NUITKA_PYTHONPATH_AST'] = os.path.dirname(ast.__file__)
    if sys.flags.no_site:
        os.environ['NUITKA_NOSITE_FLAG'] = '1'
    os.environ['PYTHONHASHSEED'] = '0'
    os.environ['NUITKA_REEXECUTION'] = '1'
    callExecProcess(args)