""" Test Nuitka compiling itself and compiling itself in compiled form again.

This should not only give no errors, but the same source for modules being
compiled when Nuitka is running compiled and uncompiled, so we can discover
changes in order of execution in this test.
"""
import os
import sys
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
import difflib
import shutil
import subprocess
import time
from nuitka.tools.Basics import addPYTHONPATH
from nuitka.tools.testing.Common import getPythonSysPath, getTempDir, my_print, setup, test_logger, withPythonPathChange
from nuitka.utils.Execution import wrapCommandForDebuggerForSubprocess
from nuitka.utils.FileOperations import copyTree, deleteFile, listDir, removeDirectory
from nuitka.utils.Importing import getSharedLibrarySuffix
from nuitka.Version import getCommercialVersion
nuitka_main_path = os.path.join('..', '..', 'bin', 'nuitka')
tmp_dir = getTempDir()
PACKAGE_LIST = ['nuitka', 'nuitka/nodes', 'nuitka/specs', 'nuitka/nodes/shapes', 'nuitka/tree', 'nuitka/importing', 'nuitka/build', 'nuitka/freezer', 'nuitka/code_generation', 'nuitka/code_generation/templates', 'nuitka/code_generation/c_types', 'nuitka/optimizations', 'nuitka/finalizations', 'nuitka/plugins', 'nuitka/plugins/standard', 'nuitka/plugins/commercial', 'nuitka/reports', 'nuitka/pgo', 'nuitka/containers', 'nuitka/utils']
if not getCommercialVersion():
    PACKAGE_LIST.remove('nuitka/plugins/commercial')
exe_suffix = '.exe' if os.name == 'nt' else '.bin'

def readSource(filename):
    if False:
        i = 10
        return i + 15
    if str is bytes:
        return open(filename, 'rb').read()
    else:
        return open(filename, 'rb').read().decode('latin1')

def diffRecursive(dir1, dir2):
    if False:
        return 10
    done = set()
    result = False
    for (path1, filename) in listDir(dir1):
        if 'cache-' in path1:
            continue
        path2 = os.path.join(dir2, filename)
        done.add(path1)
        if filename.endswith(('.o', '.os', '.obj', '.dblite', '.tmp', '.sconsign', '.txt', '.bin', '.const', '.exp')):
            continue
        if not os.path.exists(path2):
            test_logger.warning('Only in %s: %s' % (dir1, filename))
            result = False
            continue
        if os.path.isdir(path1):
            r = diffRecursive(path1, path2)
            if r:
                result = True
        elif os.path.isfile(path1):
            fromdate = time.ctime(os.stat(path1).st_mtime)
            todate = time.ctime(os.stat(path2).st_mtime)
            diff = difflib.unified_diff(a=readSource(path1).splitlines(), b=readSource(path2).splitlines(), fromfile=path1, tofile=path2, fromfiledate=fromdate, tofiledate=todate, n=3)
            diff_list = list(diff)
            if diff_list:
                for line in diff_list:
                    try:
                        my_print(line)
                    except UnicodeEncodeError:
                        my_print(repr(line))
                result = True
        else:
            assert False, path1
    for (path1, filename) in listDir(dir2):
        if 'cache-' in path1:
            continue
        path2 = os.path.join(dir2, filename)
        if path1 in done:
            continue
        if not os.path.exists(path1):
            test_logger.warning('Only in %s: %s' % (dir2, filename))
            result = False
            continue
    return result

def _traceCompilation(path, pass_number):
    if False:
        print('Hello World!')
    test_logger.info("Compiling '%s' (PASS %d)." % (path, pass_number))

def executePASS1():
    if False:
        while True:
            i = 10
    test_logger.info('PASS 1: Compiling to many compiled modules from compiler running from .py files.')
    base_dir = os.path.join('..', '..')
    for package in PACKAGE_LIST:
        package = package.replace('/', os.path.sep)
        source_dir = os.path.join(base_dir, package)
        target_dir = package
        removeDirectory(path=target_dir, ignore_errors=False)
        os.mkdir(target_dir)
        for (path, filename) in listDir(target_dir):
            if filename.endswith(('.so', '.dylib')):
                os.unlink(path)
        for (path, filename) in listDir(source_dir):
            if not filename.endswith('.py'):
                continue
            if filename.startswith('.#'):
                continue
            if filename != '__init__.py':
                _traceCompilation(path=path, pass_number=1)
                command = [os.environ['PYTHON'], nuitka_main_path, '--module', '--nofollow-imports', '--output-dir=%s' % target_dir, '--no-pyi-file', path]
                command += os.environ.get('NUITKA_EXTRA_OPTIONS', '').split()
                my_print('Command: ', ' '.join(command))
                result = subprocess.call(command)
                if result != 0:
                    sys.exit(result)
            else:
                shutil.copyfile(path, os.path.join(target_dir, filename))
    _traceCompilation(path=nuitka_main_path, pass_number=1)
    shutil.copyfile(nuitka_main_path, 'nuitka-runner.py')
    command = [os.environ['PYTHON'], nuitka_main_path, '--nofollow-imports', '--enable-plugin=pylint-warnings', '--output-dir=.', '--python-flag=no_site', 'nuitka-runner.py']
    command += os.environ.get('NUITKA_EXTRA_OPTIONS', '').split()
    my_print('Command: ', ' '.join(command))
    result = subprocess.call(command)
    if result != 0:
        sys.exit(result)
    shutil.move('nuitka-runner' + exe_suffix, 'nuitka' + exe_suffix)
    scons_inline_copy_path = os.path.join(base_dir, 'nuitka', 'build', 'inline_copy')
    if os.path.exists(scons_inline_copy_path):
        copyTree(scons_inline_copy_path, os.path.join('nuitka', 'build', 'inline_copy'))
    for filename in ('nuitka/build/Backend.scons', 'nuitka/plugins/standard/standard.nuitka-package.config.yml', 'nuitka/plugins/standard/stdlib3.nuitka-package.config.yml', 'nuitka/plugins/standard/stdlib2.nuitka-package.config.yml'):
        shutil.copyfile(os.path.join(base_dir, filename), filename)
    copyTree(os.path.join(base_dir, 'nuitka', 'code_generation', 'templates_c'), os.path.join('nuitka', 'code_generation', 'templates_c'))
    copyTree(os.path.join(base_dir, 'nuitka', 'build', 'static_src'), os.path.join('nuitka', 'build', 'static_src'))
    copyTree(os.path.join(base_dir, 'nuitka', 'build', 'include'), os.path.join('nuitka', 'build', 'include'))
    copyTree(os.path.join(base_dir, 'nuitka', 'tools'), os.path.join('nuitka', 'tools'))
    test_logger.info('OK.')

def compileAndCompareWith(nuitka, pass_number):
    if False:
        for i in range(10):
            print('nop')
    if 'PYTHONHASHSEED' not in os.environ:
        os.environ['PYTHONHASHSEED'] = '0'
    if 'PYTHON_FROZEN_MODULES' not in os.environ:
        os.environ['PYTHON_FROZEN_MODULES'] = 'off'
    base_dir = os.path.join('..', '..')
    for package in PACKAGE_LIST:
        package = package.replace('/', os.path.sep)
        source_dir = os.path.join(base_dir, package)
        for (path, filename) in listDir(source_dir):
            if not filename.endswith('.py'):
                continue
            if filename.startswith('.#'):
                continue
            path = os.path.join(source_dir, filename)
            if filename != '__init__.py':
                _traceCompilation(path=path, pass_number=pass_number)
                target = filename.replace('.py', '.build')
                target_dir = os.path.join(tmp_dir, target)
                removeDirectory(path=target_dir, ignore_errors=False)
                command = [nuitka, '--module', '--enable-plugin=pylint-warnings', '--output-dir=%s' % tmp_dir, '--no-pyi-file', '--nofollow-imports', path]
                command += os.environ.get('NUITKA_EXTRA_OPTIONS', '').split()
                my_print('Command: ', ' '.join(command))
                exit_nuitka = subprocess.call(command)
                if exit_nuitka in (-11, -6) and sys.platform != 'nt':
                    command2 = wrapCommandForDebuggerForSubprocess(*command)
                    subprocess.call(command2)
                if exit_nuitka != 0:
                    my_print('An error exit %s occurred, aborting.' % exit_nuitka)
                    sys.exit(exit_nuitka)
                has_diff = diffRecursive(os.path.join(package, target), target_dir)
                if has_diff:
                    sys.exit('There were differences!')
                shutil.rmtree(target_dir)
                for preferred in (True, False):
                    target_filename = filename.replace('.py', getSharedLibrarySuffix(preferred=preferred))
                    deleteFile(path=os.path.join(tmp_dir, target_filename), must_exist=False)

def executePASS2():
    if False:
        while True:
            i = 10
    test_logger.info("PASS 2: Compiling from compiler running from entry '.exe' and many extension files.")
    with withPythonPathChange(getPythonSysPath()):
        if os.name == 'nt':
            addPYTHONPATH(PACKAGE_LIST)
        compileAndCompareWith(nuitka=os.path.join('.', 'nuitka' + exe_suffix), pass_number=2)
    test_logger.info('OK.')

def executePASS3():
    if False:
        i = 10
        return i + 15
    test_logger.info('PASS 3: Compiling from compiler running from .py files to single .exe.')
    exe_path = os.path.join(tmp_dir, 'nuitka' + exe_suffix)
    if os.path.exists(exe_path):
        os.unlink(exe_path)
    build_path = os.path.join(tmp_dir, 'nuitka.build')
    if os.path.exists(build_path):
        shutil.rmtree(build_path)
    path = os.path.join('..', '..', 'bin', 'nuitka')
    _traceCompilation(path=path, pass_number=3)
    command = [os.environ['PYTHON'], nuitka_main_path, path, '--output-dir=%s' % tmp_dir, '--python-flag=-S', '--follow-imports']
    my_print('Command: ', ' '.join(command))
    result = subprocess.call(command)
    if result != 0:
        sys.exit(result)
    shutil.rmtree(build_path)
    test_logger.info('OK.')

def executePASS4():
    if False:
        print('Hello World!')
    test_logger.info('PASS 4: Compiling the compiler running from single exe.')
    exe_path = os.path.join(tmp_dir, 'nuitka' + exe_suffix)
    with withPythonPathChange(getPythonSysPath()):
        if os.name == 'nt':
            addPYTHONPATH(PACKAGE_LIST)
        compileAndCompareWith(exe_path, pass_number=4)
    test_logger.info('OK.')

def executePASS5():
    if False:
        print('Hello World!')
    my_print("PASS 5: Compiling the compiler 'nuitka' package to single extension module.")
    path = os.path.join('..', '..', 'nuitka')
    command = [os.environ['PYTHON'], nuitka_main_path, '--enable-plugin=pylint-warnings', '--output-dir=%s' % tmp_dir, '--include-plugin-dir=%s' % path, '--nofollow-import-to=nuitka.build.inline_copy', '--nofollow-import-to=nuitka.build.include', '--nofollow-import-to=nuitka.build.static_src', '--nofollow-import-to=nuitka.tools', '--module', path]
    result = subprocess.call(command)
    if result != 0:
        sys.exit(result)
    for preferred in (True, False):
        candidate = 'nuitka' + getSharedLibrarySuffix(preferred=preferred)
        deleteFile(candidate, must_exist=False)
    os.unlink(os.path.join(tmp_dir, 'nuitka.pyi'))
    shutil.rmtree(os.path.join(tmp_dir, 'nuitka.build'))

def main():
    if False:
        print('Hello World!')
    setup(needs_io_encoding=True)
    executePASS1()
    executePASS2()
    executePASS3()
    executePASS4()
    shutil.rmtree('nuitka')
    executePASS5()
if __name__ == '__main__':
    main()