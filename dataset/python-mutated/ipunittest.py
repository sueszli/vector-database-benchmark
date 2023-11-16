"""Experimental code for cleaner support of IPython syntax with unittest.

In IPython up until 0.10, we've used very hacked up nose machinery for running
tests with IPython special syntax, and this has proved to be extremely slow.
This module provides decorators to try a different approach, stemming from a
conversation Brian and I (FP) had about this problem Sept/09.

The goal is to be able to easily write simple functions that can be seen by
unittest as tests, and ultimately for these to support doctests with full
IPython syntax.  Nose already offers this based on naming conventions and our
hackish plugins, but we are seeking to move away from nose dependencies if
possible.

This module follows a different approach, based on decorators.

- A decorator called @ipdoctest can mark any function as having a docstring
  that should be viewed as a doctest, but after syntax conversion.

Authors
-------

- Fernando Perez <Fernando.Perez@berkeley.edu>
"""
import re
import unittest
from doctest import DocTestFinder, DocTestRunner, TestResults
from IPython.terminal.interactiveshell import InteractiveShell

def count_failures(runner):
    if False:
        while True:
            i = 10
    'Count number of failures in a doctest runner.\n\n    Code modeled after the summarize() method in doctest.\n    '
    return [TestResults(f, t) for (f, t) in runner._name2ft.values() if f > 0]

class IPython2PythonConverter(object):
    """Convert IPython 'syntax' to valid Python.

    Eventually this code may grow to be the full IPython syntax conversion
    implementation, but for now it only does prompt conversion."""

    def __init__(self):
        if False:
            return 10
        self.rps1 = re.compile('In\\ \\[\\d+\\]: ')
        self.rps2 = re.compile('\\ \\ \\ \\.\\.\\.+: ')
        self.rout = re.compile('Out\\[\\d+\\]: \\s*?\\n?')
        self.pyps1 = '>>> '
        self.pyps2 = '... '
        self.rpyps1 = re.compile('(\\s*%s)(.*)$' % self.pyps1)
        self.rpyps2 = re.compile('(\\s*%s)(.*)$' % self.pyps2)

    def __call__(self, ds):
        if False:
            return 10
        'Convert IPython prompts to python ones in a string.'
        from . import globalipapp
        pyps1 = '>>> '
        pyps2 = '... '
        pyout = ''
        dnew = ds
        dnew = self.rps1.sub(pyps1, dnew)
        dnew = self.rps2.sub(pyps2, dnew)
        dnew = self.rout.sub(pyout, dnew)
        ip = InteractiveShell.instance()
        out = []
        newline = out.append
        for line in dnew.splitlines():
            mps1 = self.rpyps1.match(line)
            if mps1 is not None:
                (prompt, text) = mps1.groups()
                newline(prompt + ip.prefilter(text, False))
                continue
            mps2 = self.rpyps2.match(line)
            if mps2 is not None:
                (prompt, text) = mps2.groups()
                newline(prompt + ip.prefilter(text, True))
                continue
            newline(line)
        newline('')
        return '\n'.join(out)

class Doc2UnitTester(object):
    """Class whose instances act as a decorator for docstring testing.

    In practice we're only likely to need one instance ever, made below (though
    no attempt is made at turning it into a singleton, there is no need for
    that).
    """

    def __init__(self, verbose=False):
        if False:
            print('Hello World!')
        'New decorator.\n\n        Parameters\n        ----------\n\n        verbose : boolean, optional (False)\n          Passed to the doctest finder and runner to control verbosity.\n        '
        self.verbose = verbose
        self.finder = DocTestFinder(verbose=verbose, recurse=False)

    def __call__(self, func):
        if False:
            for i in range(10):
                print('nop')
        "Use as a decorator: doctest a function's docstring as a unittest.\n        \n        This version runs normal doctests, but the idea is to make it later run\n        ipython syntax instead."
        d2u = self
        if func.__doc__ is not None:
            func.__doc__ = ip2py(func.__doc__)

        class Tester(unittest.TestCase):

            def test(self):
                if False:
                    return 10
                runner = DocTestRunner(verbose=d2u.verbose)
                for the_test in d2u.finder.find(func, func.__name__):
                    runner.run(the_test)
                failed = count_failures(runner)
                if failed:
                    if len(failed) > 1:
                        err = 'Invalid number of test results: %s' % failed
                        raise ValueError(err)
                    self.fail('failed doctests: %s' % str(failed[0]))
        Tester.__name__ = func.__name__
        return Tester

def ipdocstring(func):
    if False:
        while True:
            i = 10
    'Change the function docstring via ip2py.\n    '
    if func.__doc__ is not None:
        func.__doc__ = ip2py(func.__doc__)
    return func
ipdoctest = Doc2UnitTester()
ip2py = IPython2PythonConverter()