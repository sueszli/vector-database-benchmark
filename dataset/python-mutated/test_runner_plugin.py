"""
Unit tests for :mod:`behave.runner_plugin`.
"""
from __future__ import absolute_import, print_function
import sys
from contextlib import contextmanager
import os
from pathlib import Path
from behave import configuration
from behave.api.runner import ITestRunner
from behave.configuration import Configuration
from behave.exception import ClassNotFoundError, InvalidClassError, ModuleNotFoundError
from behave.runner import Runner as DefaultRunnerClass
from behave.runner_plugin import RunnerPlugin
import pytest
PYTHON_VERSION = sys.version_info[:2]

@contextmanager
def use_current_directory(directory_path):
    if False:
        print('Hello World!')
    'Use directory as current directory.\n\n    ::\n\n        with use_current_directory("/tmp/some_directory"):\n            pass # DO SOMETHING in current directory.\n        # -- ON EXIT: Restore old current-directory.\n    '
    initial_directory = str(Path.cwd())
    try:
        os.chdir(str(directory_path))
        yield directory_path
    finally:
        os.chdir(initial_directory)

def make_exception_message4abstract_method(class_name, method_name):
    if False:
        while True:
            i = 10
    '\n    Creates a regexp matcher object for the TypeError exception message\n    that is raised if an abstract method is encountered.\n    '
    message = "\nCan't instantiate abstract class {class_name} (with|without an implementation for) abstract method(s)? (')?{method_name}(')?\n".format(class_name=class_name, method_name=method_name).strip()
    return message

class CustomTestRunner(ITestRunner):
    """Custom, dummy runner"""

    def __init__(self, config, **kwargs):
        if False:
            return 10
        self.config = config

    def run(self):
        if False:
            print('Hello World!')
        return True

    @property
    def undefined_steps(self):
        if False:
            return 10
        return []

class PhoenixTestRunner(ITestRunner):

    def __init__(self, config, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.config = config
        self.the_runner = DefaultRunnerClass(config)

    def run(self, features=None):
        if False:
            i = 10
            return i + 15
        return self.the_runner.run(features=features)

    @property
    def undefined_steps(self):
        if False:
            return 10
        return self.the_runner.undefined_steps

class RegisteredTestRunner(object):
    """Not derived from :class:`behave.api.runner:ITestrunner`.
    In this case, you need to register this class to the interface class.
    """

    def __init__(self, config, **kwargs):
        if False:
            i = 10
            return i + 15
        self.config = config

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        return True

    @property
    def undefined_steps(self):
        if False:
            return 10
        return self.the_runner.undefined_steps
ITestRunner.register(RegisteredTestRunner)
INVALID_TEST_RUNNER_CLASS0 = True

class InvalidTestRunnerNotSubclass(object):
    """SYNDROME: Missing ITestRunner.register(InvalidTestRunnerNotSubclass)."""

    def __int__(self, config):
        if False:
            return 10
        self.undefined_steps = []

    def run(self, features=None):
        if False:
            while True:
                i = 10
        return True

class InvalidTestRunnerWithoutCtor(ITestRunner):
    """SYNDROME: ctor() method is missing"""

    def run(self, features=None):
        if False:
            while True:
                i = 10
        pass

    @property
    def undefined_steps(self):
        if False:
            i = 10
            return i + 15
        return []

class InvalidTestRunnerWithoutRun(ITestRunner):
    """SYNDROME: run() method is missing"""

    def __init__(self, config, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.config = config

    @property
    def undefined_steps(self):
        if False:
            while True:
                i = 10
        return []

class InvalidTestRunnerWithoutUndefinedSteps(ITestRunner):
    """SYNDROME: undefined_steps property is missing"""

    def __init__(self, config, **kwargs):
        if False:
            return 10
        self.config = config

    def run(self, features=None):
        if False:
            for i in range(10):
                print('nop')
        pass

class TestRunnerPlugin(object):
    """Test the runner-plugin configuration."""
    THIS_MODULE_NAME = CustomTestRunner.__module__

    def test_make_runner_with_default(self, tmp_path):
        if False:
            for i in range(10):
                print('nop')
        with use_current_directory(tmp_path):
            config_file = tmp_path / 'behave.ini'
            config = Configuration('')
            runner = RunnerPlugin().make_runner(config)
            assert config.runner == configuration.DEFAULT_RUNNER_CLASS_NAME
            assert isinstance(runner, DefaultRunnerClass)
            assert not config_file.exists()

    def test_make_runner_with_default_from_configfile(self, tmp_path):
        if False:
            for i in range(10):
                print('nop')
        config_file = tmp_path / 'behave.ini'
        config_file.write_text(u'\n[behave]\nrunner = behave.runner:Runner\n')
        with use_current_directory(tmp_path):
            config = Configuration('')
            runner = RunnerPlugin().make_runner(config)
            assert config.runner == configuration.DEFAULT_RUNNER_CLASS_NAME
            assert isinstance(runner, DefaultRunnerClass)
            assert config_file.exists()

    def test_make_runner_with_normal_runner_class(self):
        if False:
            print('Hello World!')
        config = Configuration(['--runner=behave.runner:Runner'])
        runner = RunnerPlugin().make_runner(config)
        assert isinstance(runner, DefaultRunnerClass)

    def test_make_runner_with_own_runner_class(self):
        if False:
            while True:
                i = 10
        config = Configuration(['--runner=%s:CustomTestRunner' % self.THIS_MODULE_NAME])
        runner = RunnerPlugin().make_runner(config)
        assert isinstance(runner, CustomTestRunner)

    def test_make_runner_with_registered_runner_class(self):
        if False:
            for i in range(10):
                print('nop')
        config = Configuration(['--runner=%s:RegisteredTestRunner' % self.THIS_MODULE_NAME])
        runner = RunnerPlugin().make_runner(config)
        assert isinstance(runner, RegisteredTestRunner)
        assert isinstance(runner, ITestRunner)
        assert issubclass(RegisteredTestRunner, ITestRunner)

    def test_make_runner_with_runner_alias(self):
        if False:
            while True:
                i = 10
        config = Configuration(['--runner=custom'])
        config.runner_aliases['custom'] = '%s:CustomTestRunner' % self.THIS_MODULE_NAME
        runner = RunnerPlugin().make_runner(config)
        assert isinstance(runner, CustomTestRunner)

    def test_make_runner_with_runner_alias_from_configfile(self, tmp_path):
        if False:
            print('Hello World!')
        config_file = tmp_path / 'behave.ini'
        config_file.write_text(u'\n[behave.runners]\ncustom = {this_module}:CustomTestRunner\n'.format(this_module=self.THIS_MODULE_NAME))
        with use_current_directory(tmp_path):
            config = Configuration(['--runner=custom'])
            runner = RunnerPlugin().make_runner(config)
            assert isinstance(runner, CustomTestRunner)
            assert config_file.exists()

    def test_make_runner_fails_with_unknown_module(self, capsys):
        if False:
            return 10
        with pytest.raises(ModuleNotFoundError) as exc_info:
            config = Configuration(['--runner=unknown_module:Runner'])
            runner = RunnerPlugin().make_runner(config)
        captured = capsys.readouterr()
        expected = 'unknown_module'
        assert exc_info.type is ModuleNotFoundError
        assert exc_info.match(expected)
        print('CAPTURED-OUTPUT: %s;' % captured.out)
        print('CAPTURED-ERROR:  %s;' % captured.err)

    def test_make_runner_fails_with_unknown_class(self, capsys):
        if False:
            i = 10
            return i + 15
        with pytest.raises(ClassNotFoundError) as exc_info:
            config = Configuration(['--runner=behave.runner:UnknownRunner'])
            RunnerPlugin().make_runner(config)
        captured = capsys.readouterr()
        assert 'FAILED to load runner.class' in captured.out
        assert 'behave.runner:UnknownRunner (ClassNotFoundError)' in captured.out
        expected = 'behave.runner:UnknownRunner'
        assert exc_info.type is ClassNotFoundError
        assert exc_info.match(expected)

    def test_make_runner_fails_if_runner_class_is_not_a_class(self):
        if False:
            return 10
        with pytest.raises(InvalidClassError) as exc_info:
            config = Configuration(['--runner=%s:INVALID_TEST_RUNNER_CLASS0' % self.THIS_MODULE_NAME])
            RunnerPlugin().make_runner(config)
        expected = 'is not a class'
        assert exc_info.type is InvalidClassError
        assert exc_info.match(expected)

    def test_make_runner_fails_if_runner_class_is_not_subclass_of_runner_interface(self):
        if False:
            while True:
                i = 10
        with pytest.raises(InvalidClassError) as exc_info:
            config = Configuration(['--runner=%s:InvalidTestRunnerNotSubclass' % self.THIS_MODULE_NAME])
            RunnerPlugin().make_runner(config)
        expected = "is not a subclass-of 'behave.api.runner:ITestRunner'"
        assert exc_info.type is InvalidClassError
        assert exc_info.match(expected)

    def test_make_runner_fails_if_runner_class_has_no_ctor(self):
        if False:
            while True:
                i = 10
        class_name = 'InvalidTestRunnerWithoutCtor'
        with pytest.raises(TypeError) as exc_info:
            config = Configuration(['--runner=%s:%s' % (self.THIS_MODULE_NAME, class_name)])
            RunnerPlugin().make_runner(config)
        expected = make_exception_message4abstract_method(class_name, method_name='__init__')
        assert exc_info.type is TypeError
        assert exc_info.match(expected)

    def test_make_runner_fails_if_runner_class_has_no_run_method(self):
        if False:
            for i in range(10):
                print('nop')
        class_name = 'InvalidTestRunnerWithoutRun'
        with pytest.raises(TypeError) as exc_info:
            config = Configuration(['--runner=%s:%s' % (self.THIS_MODULE_NAME, class_name)])
            RunnerPlugin().make_runner(config)
        expected = make_exception_message4abstract_method(class_name, method_name='run')
        assert exc_info.type is TypeError
        assert exc_info.match(expected)

    @pytest.mark.skipif(PYTHON_VERSION < (3, 0), reason='TypeError is not raised.')
    def test_make_runner_fails_if_runner_class_has_no_undefined_steps(self):
        if False:
            return 10
        class_name = 'InvalidTestRunnerWithoutUndefinedSteps'
        with pytest.raises(TypeError) as exc_info:
            config = Configuration(['--runner=%s:%s' % (self.THIS_MODULE_NAME, class_name)])
            RunnerPlugin().make_runner(config)
        expected = make_exception_message4abstract_method(class_name, 'undefined_steps')
        assert exc_info.type is TypeError
        assert exc_info.match(expected)