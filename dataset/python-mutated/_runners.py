"""Test running functions."""
from __future__ import print_function
import sys
import os
import warnings
from os import path as op
from copy import deepcopy
from functools import partial
from ..util import use_log_level, run_subprocess
from ..util.ptime import time
from ._testing import has_application, nottest, IS_CI, IS_TRAVIS_CI
_line_sep = '-' * 70

class VispySkipSuite(Exception):
    """Class we use to internally signal skipping a test suite."""

    def __init__(self, msg=''):
        if False:
            while True:
                i = 10
        if msg:
            print(msg)
        super(VispySkipSuite, self).__init__(msg)

def _get_import_dir():
    if False:
        i = 10
        return i + 15
    import_dir = op.abspath(op.join(op.dirname(__file__), '..'))
    up_dir = op.join(import_dir, '..')
    if op.isfile(op.join(up_dir, 'setup.py')) and op.isdir(op.join(up_dir, 'vispy')) and op.isdir(op.join(up_dir, 'examples')):
        dev = True
    else:
        dev = False
    return (import_dir, dev)

def _unit(mode, extra_arg_string='', coverage=False):
    if False:
        while True:
            i = 10
    'Run unit tests using a particular mode'
    if isinstance(extra_arg_string, str):
        if len(extra_arg_string):
            extra_args = extra_arg_string.split(' ')
        else:
            extra_args = ()
    else:
        extra_args = extra_arg_string
    del extra_arg_string
    assert isinstance(extra_args, (list, tuple))
    assert all((isinstance(e, str) for e in extra_args))
    import_dir = _get_import_dir()[0]
    cwd = op.abspath(op.join(import_dir, '..'))
    extra_args = list(extra_args)
    try:
        import pytest
    except ImportError:
        raise VispySkipSuite('Skipping unit tests, pytest not installed')
    if mode == 'nobackend':
        msg = 'Running tests with no backend'
        extra_args += ['-m', 'not vispy_app_test']
    else:
        (stdout, stderr, invalid) = run_subprocess([sys.executable, '-c', 'import vispy.app; vispy.app.use_app("%s"); exit(0)' % mode], return_code=True)
        if invalid:
            stdout = stdout + '\n' + stderr
            stdout = '\n'.join(('    ' + x for x in stdout.split('\n')))
            raise VispySkipSuite('\n%s\n%s\n%s' % (_line_sep, 'Skipping backend %s, not installed or working properly:\n%s' % (mode, stdout), _line_sep))
        msg = 'Running tests with %s backend' % mode
        extra_args += ['-m', 'vispy_app_test']
    if coverage:
        extra_args += ['--cov', 'vispy', '--cov-report=']
    if not any((e.startswith('-r') for e in extra_args)):
        extra_args.append('-ra')
    extra_args += [import_dir]
    cmd = [sys.executable, '-m', 'pytest'] + extra_args
    env = deepcopy(os.environ)
    env.update(dict(_VISPY_TESTING_APP=mode, VISPY_IGNORE_OLD_VERSION='true'))
    env_str = '_VISPY_TESTING_APP=%s ' % mode
    if len(msg) > 0:
        cmd_string = ' '.join(cmd)
        msg = '%s\n%s:\n%s%s' % (_line_sep, 'msg', env_str, cmd_string)
        print(msg)
    sys.stdout.flush()
    return_code = run_subprocess(cmd, return_code=True, cwd=cwd, env=env, stdout=None, stderr=None)[2]
    if return_code:
        raise RuntimeError('unit failure (%s)' % return_code)
    if coverage:
        out_name = op.join(cwd, '.vispy-coverage.%s' % mode)
        if op.isfile(out_name):
            os.remove(out_name)
        os.rename(op.join(cwd, '.coverage'), out_name)

def _docs():
    if False:
        return 10
    'Test docstring parameters\n    using vispy/utils/tests/test_docstring_parameters.py\n    '
    dev = _get_import_dir()[1]
    if not dev:
        warnings.warn("Docstring test imports Vispy from Vispy's installation. It is recommended to setup Vispy using 'python setup.py develop' so that the latest sources are used automatically")
    try:
        from ..util.tests import test_docstring_parameters
        print('Running docstring test...')
        test_docstring_parameters.test_docstring_parameters()
    except AssertionError as docstring_violations:
        raise RuntimeError(docstring_violations)

def _flake():
    if False:
        return 10
    'Test flake8'
    orig_dir = os.getcwd()
    (import_dir, dev) = _get_import_dir()
    os.chdir(op.join(import_dir, '..'))
    if dev:
        sys.argv[1:] = ['vispy', 'examples', 'make']
    else:
        sys.argv[1:] = [op.basename(import_dir)]
    try:
        try:
            from flake8.main import main
        except ImportError:
            from flake8.main.cli import main
    except ImportError:
        print('Skipping flake8 test, flake8 not installed')
    else:
        print('Running flake8... ')
        sys.stdout.flush()
        try:
            main()
        except SystemExit as ex:
            if ex.code in (None, 0):
                pass
            else:
                raise RuntimeError('flake8 failed')
    finally:
        os.chdir(orig_dir)

def _check_line_endings():
    if False:
        i = 10
        return i + 15
    'Check all files in the repository for CR characters'
    if sys.platform == 'win32':
        print('Skipping line endings check on Windows')
        sys.stdout.flush()
        return
    print('Running line endings check... ')
    sys.stdout.flush()
    report = []
    (import_dir, dev) = _get_import_dir()
    for (dirpath, dirnames, filenames) in os.walk(import_dir):
        for fname in filenames:
            if op.splitext(fname)[1] in ('.pyc', '.pyo', '.so', '.dll'):
                continue
            filename = op.join(dirpath, fname)
            relfilename = op.relpath(filename, import_dir)
            try:
                with open(filename, 'rb') as fid:
                    text = fid.read().decode('utf-8')
            except UnicodeDecodeError:
                continue
            crcount = text.count('\r')
            if crcount:
                lfcount = text.count('\n')
                report.append('In %s found %i/%i CR/LF' % (relfilename, crcount, lfcount))
    if len(report) > 0:
        raise RuntimeError('Found %s files with incorrect endings:\n%s' % (len(report), '\n'.join(report)))
_script = "\nimport sys\nimport time\nimport warnings\nimport os\ntry:\n    import faulthandler\n    faulthandler.enable()\nexcept Exception:\n    pass\nfrom vispy.util.gallery_scraper import get_canvaslike_from_globals, FrameGrabber\nos.environ['VISPY_IGNORE_OLD_VERSION'] = 'true'\nos.environ['_VISPY_RUNNING_GALLERY_EXAMPLES'] = 'true'\nimport {0}\n\ncanvas_or_widget = get_canvaslike_from_globals({0}.__dict__)\nif canvas_or_widget is None:\n    raise RuntimeError('Bad example formatting: fix or add `# vispy: testskip`'\n                       ' to the top of the file.')\n\nframe_grabber = FrameGrabber(canvas_or_widget, [1, 2, 3, 4, 5])\nframe_grabber.collect_frames()\n"
bad_examples = []
if IS_TRAVIS_CI and sys.platform == 'darwin':
    bad_examples = ['examples/basics/plotting/colorbar.py', 'examples/basics/plotting/plot.py', 'examples/demo/gloo/high_frequency.py', 'examples/basics/scene/shared_context.py']
elif IS_CI and 'linux' in sys.platform:
    bad_examples = ['examples/basics/scene/shared_context.py']
if IS_CI:
    bad_examples += ['examples/basics/gloo/geometry_shader.py', 'examples/gloo/geometry_shader.py']

def _skip_example(fname):
    if False:
        while True:
            i = 10
    for bad_ex in bad_examples:
        if fname.endswith(bad_ex):
            return True
    return False

def _examples(fnames_str):
    if False:
        print('Hello World!')
    'Run examples and make sure they work.\n\n    Parameters\n    ----------\n    fnames_str : str\n        Can be a space-separated list of paths to test, or an empty string to\n        auto-detect and run all examples.\n    '
    (import_dir, dev) = _get_import_dir()
    reason = None
    if not dev:
        reason = 'Cannot test examples unless in vispy git directory'
    else:
        with use_log_level('warning', print_msg=False):
            (good, backend) = has_application(capable=('multi_window',))
        if not good:
            reason = 'Must have suitable app backend'
    if reason is not None:
        raise VispySkipSuite('Skipping example test: %s' % reason)
    if fnames_str:
        examples_dir = ''
        fnames = fnames_str.split(' ')
    else:
        examples_dir = op.join(import_dir, '..', 'examples')
        fnames = [op.join(d[0], fname) for d in os.walk(examples_dir) for fname in d[2] if fname.endswith('.py')]
    fnames = sorted(fnames, key=lambda x: x.lower())
    print(_line_sep + '\nRunning examples using %s backend' % (backend,))
    (op.join('tutorial', 'app', 'shared_context.py'),)
    fails = []
    n_ran = n_skipped = 0
    t0 = time()
    for (fi, fname) in enumerate(fnames):
        n_ran += 1
        root_name = fname[-len(fname) + len(examples_dir):]
        good = True
        with open(fname, 'rb') as fid:
            for _ in range(10):
                line = fid.readline().decode('utf-8')
                if line == '':
                    break
                elif line.startswith('# vispy: ') and 'testskip' in line:
                    good = False
                    break
        if _skip_example(fname):
            print('Skipping example that fails on Travis CI: {}'.format(fname))
            good = False
        if not good:
            n_ran -= 1
            n_skipped += 1
            continue
        line_str = '[%3d/%3d] %s' % (fi + 1, len(fnames), root_name)
        print(line_str.ljust(len(_line_sep) - 1), end='')
        sys.stdout.flush()
        cwd = op.dirname(fname)
        cmd = [sys.executable, '-c', _script.format(op.split(fname)[1][:-3])]
        sys.stdout.flush()
        (stdout, stderr, retcode) = run_subprocess(cmd, return_code=True, cwd=cwd, env=os.environ)
        if retcode or len(stderr.strip()) > 0:
            if 'ImportError: ' in stderr:
                print('S')
            else:
                ext = '\n' + _line_sep + '\n'
                fails.append('X%sExample %s failed (%s):%s%s%s' % (ext, root_name, retcode, ext, stderr, ext))
                print(fails[-1])
        else:
            print('✓')
        sys.stdout.flush()
    print('')
    t = ': %s failed, %s succeeded, %s skipped in %s seconds' % (len(fails), n_ran - len(fails), n_skipped, round(time() - t0))
    if len(fails) > 0:
        raise RuntimeError('Failed%s' % t)
    print('Success%s' % t)

@nottest
def test(label='full', extra_arg_string='', coverage=False):
    if False:
        while True:
            i = 10
    "Test vispy software\n\n    Parameters\n    ----------\n    label : str\n        Can be one of 'full', 'unit', 'nobackend', 'extra', 'lineendings',\n        'flake', 'docs', or any backend name (e.g., 'qt').\n    extra_arg_string : str | list of str\n        Extra arguments to sent to ``pytest``.\n        Can also be a list of str to more explicitly provide the\n        arguments.\n    coverage : bool\n        If True, collect coverage data.\n    "
    if label == 'osmesa':
        from ..util.osmesa_gl import fix_osmesa_gl_lib
        fix_osmesa_gl_lib()
    from ..app.backends import BACKEND_NAMES as backend_names
    label = label.lower()
    label = 'pytest' if label == 'nose' else label
    known_types = ['full', 'unit', 'lineendings', 'extra', 'flake', 'docs', 'nobackend', 'examples']
    if label not in known_types + backend_names:
        raise ValueError("label must be one of %s, or a backend name %s, not '%s'" % (known_types, backend_names, label))
    backend_names.remove('tkinter')
    runs = []
    if label in ('full', 'unit'):
        for backend in backend_names:
            runs.append([partial(_unit, backend, extra_arg_string, coverage), backend])
    elif label in backend_names:
        runs.append([partial(_unit, label, extra_arg_string, coverage), label])
    if label in ('full', 'unit', 'nobackend'):
        runs.append([partial(_unit, 'nobackend', extra_arg_string, coverage), 'nobackend'])
    if label == 'examples':
        runs.append([partial(_examples, extra_arg_string), 'examples'])
    elif label == 'full':
        runs.append([partial(_examples, ''), 'examples'])
    if label in ('full', 'extra', 'lineendings'):
        runs.append([_check_line_endings, 'lineendings'])
    if label in ('full', 'extra', 'flake'):
        runs.append([_flake, 'flake'])
    if label in ('extra', 'docs'):
        runs.append([_docs, 'docs'])
    t0 = time()
    fail = []
    skip = []
    for run in runs:
        try:
            run[0]()
        except RuntimeError as exp:
            print('Failed: %s' % str(exp))
            fail += [run[1]]
        except VispySkipSuite:
            skip += [run[1]]
        except Exception as exp:
            fail += [run[1]]
            print('Failed strangely (%s): %s\n' % (type(exp), str(exp)))
            import traceback
            (type_, value, tb) = sys.exc_info()
            traceback.print_exception(type_, value, tb)
        else:
            print('Passed\n')
        sys.stdout.flush()
    dt = time() - t0
    stat = '%s failed, %s skipped' % (fail if fail else 0, skip if skip else 0)
    extra = 'failed' if fail else 'succeeded'
    print('Testing %s (%s) in %0.3f seconds' % (extra, stat, dt))
    sys.stdout.flush()
    if len(fail) > 0:
        raise RuntimeError('FAILURE')