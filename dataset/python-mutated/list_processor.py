""" Processes the raw test lists from the testlist module. """
from collections import OrderedDict
from importlib import import_module
from typing import Callable
from ..util.strings import lstrip_once

def list_targets(test_lister: Callable, demo_lister: Callable, benchmark_lister: Callable):
    if False:
        return 10
    '\n    Yields tuples of (testname, type, description, condition_function)\n    for given test and demo listers.\n\n    A processing step between the raw lists in testlist, and get_all_tests().\n    '

    def default_cond(_):
        if False:
            while True:
                i = 10
        ' default condition test to enable a test '
        return True
    for test in test_lister():
        if not isinstance(test, tuple):
            test = (test,)
        if not test:
            raise ValueError('empty test definition encountered')
        if len(test) == 3:
            condfun = test[2]
        else:
            condfun = default_cond
        if len(test) >= 2:
            desc = test[1]
        else:
            desc = ''
        name = test[0]
        yield (name, 'test', desc, condfun)
    for demo in demo_lister():
        (name, desc) = demo
        yield (name, 'demo', desc, default_cond)
    for benchmark in benchmark_lister():
        (name, desc) = benchmark
        yield (name, 'benchmark', desc, default_cond)

def list_targets_py():
    if False:
        print('Hello World!')
    ' Invokes list_targets() with the py-specific listers. '
    from .testlist import tests_py, demos_py, benchmark_py
    for val in list_targets(tests_py, demos_py, benchmark_py):
        yield val

def list_targets_cpp():
    if False:
        while True:
            i = 10
    ' Invokes list_targets() with the C++-specific listers. '
    from .testlist import tests_cpp, demos_cpp, benchmark_cpp
    for val in list_targets(tests_cpp, demos_cpp, benchmark_cpp):
        yield val

def get_all_targets() -> OrderedDict:
    if False:
        print('Hello World!')
    "\n    Reads the Python and C++ testspec.\n\n    returns an OrderedDict of\n    {(testname, type): conditionfun, lang, description, testfun}.\n\n    type is in {'demo', 'test'},\n    lang is in {'cpp', 'py'},\n    conditionfun is a callable which determines if the test is\n        to be run in the given environment\n    description is a str, and\n    testfun is callable and takes 0 args for tests / list(str) for demos.\n    "
    from .cpp_testing import run_cpp_method
    result = OrderedDict()
    for (name, type_, description, conditionfun) in list_targets_py():
        (modulename, objectname) = name.rsplit('.', maxsplit=1)
        try:
            module = import_module(modulename)
            func = getattr(module, objectname)
        except Exception as exc:
            raise ValueError('no such function: ' + name) from exc
        try:
            name = lstrip_once(name, 'openage.')
        except ValueError as exc:
            raise ValueError('Unexpected Python test/demo name') from exc
        result[name, type_] = (conditionfun, 'py', description, func)
    for (name, type_, description, conditionfun) in list_targets_cpp():
        if type_ == 'demo':

            def runner(args, name=name):
                if False:
                    while True:
                        i = 10
                ' runs the demo func, and ensures that args is empty. '
                if args:
                    raise ValueError("C++ demos can't take arguments. You should write a Python demo that calls to C++ then, with arguments.")
                run_cpp_method(name)
        elif type_ in ['test', 'benchmark']:

            def runner(name=name):
                if False:
                    print('Hello World!')
                ' simply runs the func. '
                run_cpp_method(name)
        else:
            raise ValueError('Unknown type ' + type_)
        try:
            name = lstrip_once(name, 'openage::')
        except ValueError as exc:
            raise ValueError('Unexpected C++ test/demo name') from exc
        result[name, type_] = (conditionfun, 'cpp', description, runner)
    return result