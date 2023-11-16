"""
Nose test running.

This module implements ``test()`` and ``bench()`` functions for NumPy modules.

"""
from __future__ import division, absolute_import, print_function
import os
import sys
import warnings
from numpy.compat import basestring
import numpy as np
from .utils import import_nose, suppress_warnings
__all__ = ['get_package_name', 'run_module_suite', 'NoseTester', '_numpy_tester', 'get_package_name', 'import_nose', 'suppress_warnings']

def get_package_name(filepath):
    if False:
        while True:
            i = 10
    '\n    Given a path where a package is installed, determine its name.\n\n    Parameters\n    ----------\n    filepath : str\n        Path to a file. If the determination fails, "numpy" is returned.\n\n    Examples\n    --------\n    >>> np.testing.nosetester.get_package_name(\'nonsense\')\n    \'numpy\'\n\n    '
    fullpath = filepath[:]
    pkg_name = []
    while 'site-packages' in filepath or 'dist-packages' in filepath:
        (filepath, p2) = os.path.split(filepath)
        if p2 in ('site-packages', 'dist-packages'):
            break
        pkg_name.append(p2)
    if not pkg_name:
        if 'scipy' in fullpath:
            return 'scipy'
        else:
            return 'numpy'
    pkg_name.reverse()
    if pkg_name[0].endswith('.egg'):
        pkg_name.pop(0)
    return '.'.join(pkg_name)

def run_module_suite(file_to_run=None, argv=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Run a test module.\n\n    Equivalent to calling ``$ nosetests <argv> <file_to_run>`` from\n    the command line\n\n    Parameters\n    ----------\n    file_to_run : str, optional\n        Path to test module, or None.\n        By default, run the module from which this function is called.\n    argv : list of strings\n        Arguments to be passed to the nose test runner. ``argv[0]`` is\n        ignored. All command line arguments accepted by ``nosetests``\n        will work. If it is the default value None, sys.argv is used.\n\n        .. versionadded:: 1.9.0\n\n    Examples\n    --------\n    Adding the following::\n\n        if __name__ == "__main__" :\n            run_module_suite(argv=sys.argv)\n\n    at the end of a test module will run the tests when that module is\n    called in the python interpreter.\n\n    Alternatively, calling::\n\n    >>> run_module_suite(file_to_run="numpy/tests/test_matlib.py")  # doctest: +SKIP\n\n    from an interpreter will run all the test routine in \'test_matlib.py\'.\n    '
    if file_to_run is None:
        f = sys._getframe(1)
        file_to_run = f.f_locals.get('__file__', None)
        if file_to_run is None:
            raise AssertionError
    if argv is None:
        argv = sys.argv + [file_to_run]
    else:
        argv = argv + [file_to_run]
    nose = import_nose()
    from .noseclasses import KnownFailurePlugin
    nose.run(argv=argv, addplugins=[KnownFailurePlugin()])

class NoseTester(object):
    """
    Nose test runner.

    This class is made available as numpy.testing.Tester, and a test function
    is typically added to a package's __init__.py like so::

      from numpy.testing import Tester
      test = Tester().test

    Calling this test function finds and runs all tests associated with the
    package and all its sub-packages.

    Attributes
    ----------
    package_path : str
        Full path to the package to test.
    package_name : str
        Name of the package to test.

    Parameters
    ----------
    package : module, str or None, optional
        The package to test. If a string, this should be the full path to
        the package. If None (default), `package` is set to the module from
        which `NoseTester` is initialized.
    raise_warnings : None, str or sequence of warnings, optional
        This specifies which warnings to configure as 'raise' instead
        of being shown once during the test execution.  Valid strings are:

          - "develop" : equals ``(Warning,)``
          - "release" : equals ``()``, don't raise on any warnings.

        Default is "release".
    depth : int, optional
        If `package` is None, then this can be used to initialize from the
        module of the caller of (the caller of (...)) the code that
        initializes `NoseTester`. Default of 0 means the module of the
        immediate caller; higher values are useful for utility routines that
        want to initialize `NoseTester` objects on behalf of other code.

    """

    def __init__(self, package=None, raise_warnings='release', depth=0, check_fpu_mode=False):
        if False:
            for i in range(10):
                print('nop')
        if raise_warnings is None:
            raise_warnings = 'release'
        package_name = None
        if package is None:
            f = sys._getframe(1 + depth)
            package_path = f.f_locals.get('__file__', None)
            if package_path is None:
                raise AssertionError
            package_path = os.path.dirname(package_path)
            package_name = f.f_locals.get('__name__', None)
        elif isinstance(package, type(os)):
            package_path = os.path.dirname(package.__file__)
            package_name = getattr(package, '__name__', None)
        else:
            package_path = str(package)
        self.package_path = package_path
        if package_name is None:
            package_name = get_package_name(package_path)
        self.package_name = package_name
        self.raise_warnings = raise_warnings
        self.check_fpu_mode = check_fpu_mode

    def _test_argv(self, label, verbose, extra_argv):
        if False:
            print('Hello World!')
        " Generate argv for nosetest command\n\n        Parameters\n        ----------\n        label : {'fast', 'full', '', attribute identifier}, optional\n            see ``test`` docstring\n        verbose : int, optional\n            Verbosity value for test outputs, in the range 1-10. Default is 1.\n        extra_argv : list, optional\n            List with any extra arguments to pass to nosetests.\n\n        Returns\n        -------\n        argv : list\n            command line arguments that will be passed to nose\n        "
        argv = [__file__, self.package_path, '-s']
        if label and label != 'full':
            if not isinstance(label, basestring):
                raise TypeError('Selection label should be a string')
            if label == 'fast':
                label = 'not slow'
            argv += ['-A', label]
        argv += ['--verbosity', str(verbose)]
        argv += ['--exe']
        if extra_argv:
            argv += extra_argv
        return argv

    def _show_system_info(self):
        if False:
            while True:
                i = 10
        nose = import_nose()
        import numpy
        print('NumPy version %s' % numpy.__version__)
        relaxed_strides = numpy.ones((10, 1), order='C').flags.f_contiguous
        print('NumPy relaxed strides checking option:', relaxed_strides)
        npdir = os.path.dirname(numpy.__file__)
        print('NumPy is installed in %s' % npdir)
        if 'scipy' in self.package_name:
            import scipy
            print('SciPy version %s' % scipy.__version__)
            spdir = os.path.dirname(scipy.__file__)
            print('SciPy is installed in %s' % spdir)
        pyversion = sys.version.replace('\n', '')
        print('Python version %s' % pyversion)
        print('nose version %d.%d.%d' % nose.__versioninfo__)

    def _get_custom_doctester(self):
        if False:
            i = 10
            return i + 15
        ' Return instantiated plugin for doctests\n\n        Allows subclassing of this class to override doctester\n\n        A return value of None means use the nose builtin doctest plugin\n        '
        from .noseclasses import NumpyDoctest
        return NumpyDoctest()

    def prepare_test_args(self, label='fast', verbose=1, extra_argv=None, doctests=False, coverage=False, timer=False):
        if False:
            print('Hello World!')
        '\n        Run tests for module using nose.\n\n        This method does the heavy lifting for the `test` method. It takes all\n        the same arguments, for details see `test`.\n\n        See Also\n        --------\n        test\n\n        '
        import_nose()
        argv = self._test_argv(label, verbose, extra_argv)
        if coverage:
            argv += ['--cover-package=%s' % self.package_name, '--with-coverage', '--cover-tests', '--cover-erase']
        if timer:
            if timer is True:
                argv += ['--with-timer']
            elif isinstance(timer, int):
                argv += ['--with-timer', '--timer-top-n', str(timer)]
        import nose.plugins.builtin
        from nose.plugins import EntryPointPluginManager
        from .noseclasses import KnownFailurePlugin, Unplugger, FPUModeCheckPlugin
        plugins = [KnownFailurePlugin()]
        plugins += [p() for p in nose.plugins.builtin.plugins]
        if self.check_fpu_mode:
            plugins += [FPUModeCheckPlugin()]
            argv += ['--with-fpumodecheckplugin']
        try:
            entrypoint_manager = EntryPointPluginManager()
            entrypoint_manager.loadPlugins()
            plugins += [p for p in entrypoint_manager.plugins]
        except ImportError:
            pass
        doctest_argv = '--with-doctest' in argv
        if doctests == False and doctest_argv:
            doctests = True
        plug = self._get_custom_doctester()
        if plug is None:
            if doctests and (not doctest_argv):
                argv += ['--with-doctest']
        else:
            if doctest_argv:
                argv.remove('--with-doctest')
            plugins += [Unplugger('doctest'), plug]
            if doctests:
                argv += ['--with-' + plug.name]
        return (argv, plugins)

    def test(self, label='fast', verbose=1, extra_argv=None, doctests=False, coverage=False, raise_warnings=None, timer=False):
        if False:
            i = 10
            return i + 15
        '\n        Run tests for module using nose.\n\n        Parameters\n        ----------\n        label : {\'fast\', \'full\', \'\', attribute identifier}, optional\n            Identifies the tests to run. This can be a string to pass to\n            the nosetests executable with the \'-A\' option, or one of several\n            special values.  Special values are:\n\n            * \'fast\' - the default - which corresponds to the ``nosetests -A``\n              option of \'not slow\'.\n            * \'full\' - fast (as above) and slow tests as in the\n              \'no -A\' option to nosetests - this is the same as \'\'.\n            * None or \'\' - run all tests.\n            * attribute_identifier - string passed directly to nosetests as \'-A\'.\n\n        verbose : int, optional\n            Verbosity value for test outputs, in the range 1-10. Default is 1.\n        extra_argv : list, optional\n            List with any extra arguments to pass to nosetests.\n        doctests : bool, optional\n            If True, run doctests in module. Default is False.\n        coverage : bool, optional\n            If True, report coverage of NumPy code. Default is False.\n            (This requires the\n            `coverage module <https://nedbatchelder.com/code/modules/coveragehtml>`_).\n        raise_warnings : None, str or sequence of warnings, optional\n            This specifies which warnings to configure as \'raise\' instead\n            of being shown once during the test execution. Valid strings are:\n\n            * "develop" : equals ``(Warning,)``\n            * "release" : equals ``()``, do not raise on any warnings.\n        timer : bool or int, optional\n            Timing of individual tests with ``nose-timer`` (which needs to be\n            installed).  If True, time tests and report on all of them.\n            If an integer (say ``N``), report timing results for ``N`` slowest\n            tests.\n\n        Returns\n        -------\n        result : object\n            Returns the result of running the tests as a\n            ``nose.result.TextTestResult`` object.\n\n        Notes\n        -----\n        Each NumPy module exposes `test` in its namespace to run all tests for it.\n        For example, to run all tests for numpy.lib:\n\n        >>> np.lib.test() #doctest: +SKIP\n\n        Examples\n        --------\n        >>> result = np.lib.test() #doctest: +SKIP\n        Running unit tests for numpy.lib\n        ...\n        Ran 976 tests in 3.933s\n\n        OK\n\n        >>> result.errors #doctest: +SKIP\n        []\n        >>> result.knownfail #doctest: +SKIP\n        []\n        '
        verbose = min(verbose, 3)
        from . import utils
        utils.verbose = verbose
        (argv, plugins) = self.prepare_test_args(label, verbose, extra_argv, doctests, coverage, timer)
        if doctests:
            print('Running unit tests and doctests for %s' % self.package_name)
        else:
            print('Running unit tests for %s' % self.package_name)
        self._show_system_info()
        import doctest
        doctest.master = None
        if raise_warnings is None:
            raise_warnings = self.raise_warnings
        _warn_opts = dict(develop=(Warning,), release=())
        if isinstance(raise_warnings, basestring):
            raise_warnings = _warn_opts[raise_warnings]
        with suppress_warnings('location') as sup:
            warnings.resetwarnings()
            warnings.filterwarnings('always')
            for warningtype in raise_warnings:
                warnings.filterwarnings('error', category=warningtype)
            sup.filter(message='Not importing directory')
            sup.filter(message='numpy.dtype size changed')
            sup.filter(message='numpy.ufunc size changed')
            sup.filter(category=np.ModuleDeprecationWarning)
            sup.filter(message='.*boolean negative.*')
            sup.filter(message='.*boolean subtract.*')
            with warnings.catch_warnings():
                warnings.simplefilter('always')
                from ...distutils import cpuinfo
            sup.filter(category=UserWarning, module=cpuinfo)
            if sys.version_info.major == 2 and sys.py3kwarning:
                import threading
                sup.filter(DeprecationWarning, 'sys\\.exc_clear\\(\\) not supported in 3\\.x', module=threading)
                sup.filter(DeprecationWarning, message='in 3\\.x, __setslice__')
                sup.filter(DeprecationWarning, message='in 3\\.x, __getslice__')
                sup.filter(DeprecationWarning, message='buffer\\(\\) not supported in 3\\.x')
                sup.filter(DeprecationWarning, message='CObject type is not supported in 3\\.x')
                sup.filter(DeprecationWarning, message='comparing unequal types not supported in 3\\.x')
            warnings.filterwarnings('ignore', message='.*getargspec.*', category=DeprecationWarning, module='nose\\.')
            from .noseclasses import NumpyTestProgram
            t = NumpyTestProgram(argv=argv, exit=False, plugins=plugins)
        return t.result

    def bench(self, label='fast', verbose=1, extra_argv=None):
        if False:
            return 10
        '\n        Run benchmarks for module using nose.\n\n        Parameters\n        ----------\n        label : {\'fast\', \'full\', \'\', attribute identifier}, optional\n            Identifies the benchmarks to run. This can be a string to pass to\n            the nosetests executable with the \'-A\' option, or one of several\n            special values.  Special values are:\n\n            * \'fast\' - the default - which corresponds to the ``nosetests -A``\n              option of \'not slow\'.\n            * \'full\' - fast (as above) and slow benchmarks as in the\n              \'no -A\' option to nosetests - this is the same as \'\'.\n            * None or \'\' - run all tests.\n            * attribute_identifier - string passed directly to nosetests as \'-A\'.\n\n        verbose : int, optional\n            Verbosity value for benchmark outputs, in the range 1-10. Default is 1.\n        extra_argv : list, optional\n            List with any extra arguments to pass to nosetests.\n\n        Returns\n        -------\n        success : bool\n            Returns True if running the benchmarks works, False if an error\n            occurred.\n\n        Notes\n        -----\n        Benchmarks are like tests, but have names starting with "bench" instead\n        of "test", and can be found under the "benchmarks" sub-directory of the\n        module.\n\n        Each NumPy module exposes `bench` in its namespace to run all benchmarks\n        for it.\n\n        Examples\n        --------\n        >>> success = np.lib.bench() #doctest: +SKIP\n        Running benchmarks for numpy.lib\n        ...\n        using 562341 items:\n        unique:\n        0.11\n        unique1d:\n        0.11\n        ratio: 1.0\n        nUnique: 56230 == 56230\n        ...\n        OK\n\n        >>> success #doctest: +SKIP\n        True\n\n        '
        print('Running benchmarks for %s' % self.package_name)
        self._show_system_info()
        argv = self._test_argv(label, verbose, extra_argv)
        argv += ['--match', '(?:^|[\\\\b_\\\\.%s-])[Bb]ench' % os.sep]
        nose = import_nose()
        from .noseclasses import Unplugger
        add_plugins = [Unplugger('doctest')]
        return nose.run(argv=argv, addplugins=add_plugins)

def _numpy_tester():
    if False:
        i = 10
        return i + 15
    if hasattr(np, '__version__') and '.dev0' in np.__version__:
        mode = 'develop'
    else:
        mode = 'release'
    return NoseTester(raise_warnings=mode, depth=1, check_fpu_mode=True)