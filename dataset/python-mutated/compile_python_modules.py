""" This test runner compiles all Python files as a module.

This is a test to achieve some coverage, it will only find assertions of
within Nuitka or warnings from the C compiler. Code will not be run
normally.

"""
import os
import sys
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')))
import subprocess
from nuitka.tools.testing.Common import checkCompilesNotWithCPython, compileLibraryTest, createSearchMode, getPythonArch, getPythonVendor, getTempDir, my_print, setup
from nuitka.utils.Importing import getSharedLibrarySuffix
python_version = setup(suite='python_modules', needs_io_encoding=True)
python_vendor = getPythonVendor()
python_arch = getPythonArch()
search_mode = createSearchMode()
tmp_dir = getTempDir()
ignore_list = ('__phello__.foo.py', 'idnadata', 'joined_strings.py', 'test_spin.py', 'cheshire_tomography.py')
late_syntax_errors = ('_identifier.py', 'bench.py', '_tweedie_compound_poisson.py', 'session.py')

def decide(_root, filename):
    if False:
        i = 10
        return i + 15
    return filename.endswith('.py') and filename not in ignore_list and ('(' not in filename) and (filename.count('.') == 1)

def action(stage_dir, _root, path):
    if False:
        i = 10
        return i + 15
    command = [sys.executable, os.path.join('..', '..', 'bin', 'nuitka'), '--module', '--output-dir=%s' % stage_dir, '--remove-output', '--quiet', '--nofollow-imports', '--no-progressbar']
    command += os.environ.get('NUITKA_EXTRA_OPTIONS', '').split()
    suffix = getSharedLibrarySuffix(preferred=True)
    if os.path.basename(path) == '__init__.py':
        source_filename = os.path.dirname(path)
        target_filename = os.path.basename(source_filename) + suffix
    else:
        source_filename = path
        target_filename = os.path.basename(source_filename)[:-3] + suffix
    target_filename = target_filename.replace('(', '').replace(')', '')
    command.append(source_filename)
    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError:
        basename = os.path.basename(path)
        if basename in late_syntax_errors:
            my_print('Syntax error is known unreliable with file %s.' % basename)
        else:
            my_print('Falling back to full comparison due to error exit.')
            checkCompilesNotWithCPython(dirname=None, filename=path, search_mode=search_mode)
    else:
        my_print('OK')
        os.unlink(os.path.join(stage_dir, target_filename))
compileLibraryTest(search_mode=search_mode, stage_dir=os.path.join(tmp_dir, 'compile_library_%s-%s-%s' % ('.'.join((str(d) for d in python_version)), python_arch, python_vendor)), decide=decide, action=action)
search_mode.finish()