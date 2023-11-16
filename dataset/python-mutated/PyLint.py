""" PyLint handling for Nuitka.

Our usage of PyLint also works around a few issues that PyLint
has.

"""
import os
import sys
from nuitka.tools.testing.Common import hasModule, my_print
from nuitka.utils.Execution import check_output, executeProcess, getNullOutput
_pylint_version = None

def checkVersion():
    if False:
        i = 10
        return i + 15
    global _pylint_version
    if not hasModule('pylint'):
        sys.exit("Error, pylint is not installed for this interpreter '%s' version." % os.environ['PYTHON'])
    if _pylint_version is None:
        _pylint_version = check_output([os.environ['PYTHON'], '-m', 'pylint', '--version'], stderr=getNullOutput())
        if str is not bytes:
            _pylint_version = _pylint_version.decode('utf8')
        _pylint_version = _pylint_version.split('\n')[0].split()[-1].strip(',')
    my_print('Using PyLint version:', _pylint_version)
    return tuple((int(d) for d in _pylint_version.split('.')))

def getOptions():
    if False:
        i = 10
        return i + 15
    pylint_version = checkVersion()
    default_pylint_options = '--init-hook=import sys;sys.setrecursionlimit(1024*sys.getrecursionlimit())\n--disable=I0011,E1103,W0632,C0123,C0411,C0413,cyclic-import,duplicate-code,deprecated-module,deprecated-method,deprecated-argument,assignment-from-none,ungrouped-imports,no-else-return,c-extension-no-member,inconsistent-return-statements,raise-missing-from,import-outside-toplevel,useless-object-inheritance,useless-return,assignment-from-no-return,redundant-u-string-prefix,consider-using-f-string,consider-using-dict-comprehension,\n--enable=useless-suppression\n--msg-template="{path}:{line} {msg_id} {symbol} {obj} {msg}"\n--reports=no\n--persistent=no\n--method-rgx=[a-z_][a-zA-Z0-9_]{2,55}$\n--module-rgx=.*\n--function-rgx=.*\n--variable-rgx=.*\n--argument-rgx=.*\n--dummy-variables-rgx=_.*|trace_collection\n--ignored-argument-names=_.*|trace_collection\n--const-rgx=.*\n--max-line-length=125\n--no-docstring-rgx=.*\n--max-module-lines=6000\n--min-public-methods=0\n--max-public-methods=100\n--max-args=11\n--max-parents=14\n--max-statements=50\n--max-nested-blocks=10\n--max-bool-expr=10\n--score=no'.split('\n')
    if os.name != 'nt':
        default_pylint_options.append('--rcfile=%s' % os.devnull)
    if pylint_version < (2, 17):
        default_pylint_options.append('--disable=bad-whitespace')
        default_pylint_options.append('--disable=bad-continuation')
        default_pylint_options.append('--disable=no-init')
        default_pylint_options.append('--disable=similar-code')
        default_pylint_options.append('--disable=I0012')
        default_pylint_options.append('--disable=W1504')
        default_pylint_options.append('--disable=R0204')
    else:
        default_pylint_options.append('--load-plugins=pylint.extensions.no_self_use')
        default_pylint_options.append('--disable=unnecessary-lambda-assignment')
        default_pylint_options.append('--disable=unnecessary-dunder-call')
        default_pylint_options.append('--disable=arguments-differ')
        default_pylint_options.append('--disable=redefined-slots-in-subclass')
    return default_pylint_options
our_exit_code = 0

def _cleanupPylintOutput(output):
    if False:
        print('Hello World!')
    if str is not bytes:
        output = output.decode('utf8')
    output = output.replace('\r\n', '\n')
    lines = [line for line in output.split('\n') if line if 'Using config file' not in line if "Unable to import 'resource'" not in line if "Bad option value 'self-assigning-variable'" not in line]
    try:
        error_line = lines.index('No config file found, using default configuration')
        del lines[error_line]
        if error_line < len(lines):
            del lines[error_line]
    except ValueError:
        pass
    return lines

def _executePylint(filenames, pylint_options, extra_options):
    if False:
        for i in range(10):
            print('nop')
    global our_exit_code
    command = [os.environ['PYTHON'], '-m', 'pylint'] + pylint_options + extra_options + filenames
    (stdout, stderr, exit_code) = executeProcess(command)
    if exit_code == -11:
        sys.exit('Error, segfault from pylint.')
    stdout = _cleanupPylintOutput(stdout)
    stderr = _cleanupPylintOutput(stderr)
    if stderr:
        our_exit_code = 1
        for line in stderr:
            my_print(line)
    if stdout:
        while stdout and stdout[-1].startswith('******'):
            del stdout[-1]
        for line in stdout:
            my_print(line)
        if stdout:
            our_exit_code = 1
    sys.stdout.flush()

def hasPyLintBugTrigger(filename):
    if False:
        i = 10
        return i + 15
    'Decide if a filename should be skipped.'
    if filename == 'nuitka/distutils/Build.py':
        return True
    return False

def isSpecificPythonOnly(filename):
    if False:
        print('Hello World!')
    'Decide if something is not used for this specific Python.'
    return False

def executePyLint(filenames, show_todo, verbose, one_by_one):
    if False:
        i = 10
        return i + 15
    filenames = list(filenames)
    if verbose:
        my_print('Checking', filenames, '...')
    pylint_options = getOptions()
    if not show_todo:
        pylint_options.append('--notes=')
    filenames = [filename for filename in filenames if not hasPyLintBugTrigger(filename) if not isSpecificPythonOnly(filename)]
    extra_options = os.environ.get('PYLINT_EXTRA_OPTIONS', '').split()
    if '' in extra_options:
        extra_options.remove('')
    if one_by_one:
        for filename in filenames:
            my_print('Checking', filename, ':')
            _executePylint([filename], pylint_options, extra_options)
    else:
        _executePylint(filenames, pylint_options, extra_options)