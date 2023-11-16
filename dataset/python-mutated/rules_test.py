from __future__ import annotations
from textwrap import dedent
import pytest
from pants.backend.python.goals.setup_py import SetupKwargs
from pants.backend.python.macros.python_artifact import PythonArtifact
from pants.backend.python.target_types import PythonDistribution, PythonSourceTarget, PythonSourcesGeneratorTarget
from pants.backend.python.target_types_rules import rules as python_target_types_rules
from pants.engine.addresses import Address
from pants.engine.internals.scheduler import ExecutionError
from pants.testutil.rule_runner import QueryRule, RuleRunner
from pants.util.frozendict import FrozenDict
from release.rules import StackStormSetupKwargsRequest
from release.rules import PROJECT_URLS, META_CLASSIFIERS, LINUX_CLASSIFIER
from release.rules import rules as release_rules

@pytest.fixture
def rule_runner() -> RuleRunner:
    if False:
        for i in range(10):
            print('nop')
    rule_runner = RuleRunner(rules=[*python_target_types_rules(), *release_rules(), QueryRule(SetupKwargs, (StackStormSetupKwargsRequest,))], target_types=[PythonDistribution, PythonSourceTarget, PythonSourcesGeneratorTarget], objects={'python_artifact': PythonArtifact})
    rule_runner.write_files({'runners/foobar_runner/BUILD': dedent('                python_distribution(\n                    provides=python_artifact(\n                        name="stackstorm-runner-foobar",\n                    ),\n                    dependencies=["./foobar_runner"],\n                    entry_points={\n                        "st2common.runners.runner": {\n                            "foobar": "foobar_runner.foobar_runner",\n                        },\n                    },\n                )\n                '), 'runners/foobar_runner/foobar_runner/BUILD': 'python_sources()', 'runners/foobar_runner/foobar_runner/__init__.py': '', 'runners/foobar_runner/foobar_runner/foobar_runner.py': '', 'runners/foobar_runner/foobar_runner/thing1.py': '', 'runners/foobar_runner/foobar_runner/thing2.py': ''})
    args = ['--source-root-patterns=runners/*_runner']
    rule_runner.set_options(args, env_inherit={'PATH', 'PYENV_ROOT', 'HOME'})
    return rule_runner

def gen_setup_kwargs(address: Address, rule_runner: RuleRunner) -> SetupKwargs:
    if False:
        return 10
    target = rule_runner.get_target(address)
    return rule_runner.request(SetupKwargs, [StackStormSetupKwargsRequest(target)])

def test_setup_kwargs_plugin_no_description_kwarg(rule_runner: RuleRunner) -> None:
    if False:
        for i in range(10):
            print('nop')
    rule_runner.write_files({'runners/foobar_runner/BUILD': dedent('                python_distribution(\n                    provides=python_artifact(\n                        name="stackstorm-runner-foobar",\n                    ),\n                    dependencies=["./foobar_runner"],\n                )\n                ')})
    address = Address('runners/foobar_runner')
    with pytest.raises(ExecutionError) as e:
        _ = gen_setup_kwargs(address, rule_runner)
    exc = e.value.wrapped_exceptions[0]
    assert isinstance(exc, ValueError)
    assert 'Missing a `description` kwarg in the `provides` field' in str(exc)

def test_setup_kwargs_plugin_no_version_file_kwarg(rule_runner: RuleRunner) -> None:
    if False:
        return 10
    rule_runner.write_files({'runners/foobar_runner/BUILD': dedent('                python_distribution(\n                    provides=python_artifact(\n                        name="stackstorm-runner-foobar",\n                        description="Foobar runner for ST2",\n                    ),\n                    dependencies=["./foobar_runner"],\n                )\n                ')})
    address = Address('runners/foobar_runner')
    with pytest.raises(ExecutionError) as e:
        _ = gen_setup_kwargs(address, rule_runner)
    exc = e.value.wrapped_exceptions[0]
    assert isinstance(exc, ValueError)
    assert 'Missing a `version_file` kwarg in the `provides` field' in str(exc)

def test_setup_kwargs_plugin_no_version_file(rule_runner: RuleRunner) -> None:
    if False:
        for i in range(10):
            print('nop')
    rule_runner.write_files({'runners/foobar_runner/BUILD': dedent('                python_distribution(\n                    provides=python_artifact(\n                        name="stackstorm-runner-foobar",\n                        description="Foobar runner for ST2",\n                        version_file="foobar_runner/__missing__.py",\n                    ),\n                    dependencies=["./foobar_runner"],\n                )\n                ')})
    address = Address('runners/foobar_runner')
    with pytest.raises(ExecutionError) as e:
        _ = gen_setup_kwargs(address, rule_runner)
    exc = e.value.wrapped_exceptions[0]
    assert 'Unmatched glob from StackStorm version file: foobar_runner/__missing__.py' in str(exc)

def test_setup_kwargs_plugin_no_version(rule_runner: RuleRunner) -> None:
    if False:
        for i in range(10):
            print('nop')
    rule_runner.write_files({'runners/foobar_runner/BUILD': dedent('                python_distribution(\n                    provides=python_artifact(\n                        name="stackstorm-runner-foobar",\n                        description="Foobar runner for ST2",\n                        version_file="foobar_runner/__init__.py",\n                    ),\n                )\n                '), 'runners/foobar_runner/foobar_runner/__init__.py': 'contents do not have version'})
    address = Address('runners/foobar_runner')
    with pytest.raises(ExecutionError) as e:
        _ = gen_setup_kwargs(address, rule_runner)
    exc = e.value.wrapped_exceptions[0]
    assert isinstance(exc, ValueError)
    assert 'Could not find the __version__' in str(exc)

def test_setup_kwargs_plugin_conflicting_kwargs(rule_runner: RuleRunner) -> None:
    if False:
        while True:
            i = 10
    rule_runner.write_files({'runners/foobar_runner/BUILD': dedent('                python_distribution(\n                    provides=python_artifact(\n                        name="stackstorm-runner-foobar",\n                        description="Foobar runner for ST2",\n                        version_file="foobar_runner/__init__.py",\n                        # these conflict with auto args\n                        version="1.2bad3",\n                        author="Anonymous",\n                        license="MIT",\n                        project_urls={"Foo": "bar://baz"},\n                        long_description="conflict",\n                    ),\n                )\n                '), 'runners/foobar_runner/foobar_runner/__init__.py': '__version__ = "0.0test0"', 'runners/foobar_runner/README.rst': 'lorem ipsum'})
    conflicting = sorted({'version', 'author', 'license', 'project_urls', 'long_description'})
    address = Address('runners/foobar_runner')
    with pytest.raises(ExecutionError) as e:
        _ = gen_setup_kwargs(address, rule_runner)
    exc = e.value.wrapped_exceptions[0]
    assert isinstance(exc, ValueError)
    assert 'These kwargs should not be set in the `provides` field' in str(exc)
    assert str(conflicting) in str(exc)

def test_setup_kwargs_plugin(rule_runner: RuleRunner) -> None:
    if False:
        print('Hello World!')
    rule_runner.write_files({'runners/foobar_runner/BUILD': dedent('                python_distribution(\n                    provides=python_artifact(\n                        name="stackstorm-runner-foobar",\n                        description="Foobar runner for ST2",\n                        version_file="foobar_runner/__init__.py",\n                        classifiers=["Qwerty :: Asdf :: Zxcv"],\n                    ),\n                    dependencies=[\n                        "./foobar_runner",\n                    ],\n                    entry_points={\n                        "st2common.runners.runner": {\n                            "foobar": "foobar_runner.foobar_runner",\n                        },\n                    },\n                )\n                '), 'runners/foobar_runner/foobar_runner/__init__.py': '__version__ = "0.0test0"'})
    address = Address('runners/foobar_runner')
    assert gen_setup_kwargs(address, rule_runner) == SetupKwargs(FrozenDict({'name': 'stackstorm-runner-foobar', 'description': 'Foobar runner for ST2', 'author': 'StackStorm', 'author_email': 'info@stackstorm.com', 'url': 'https://stackstorm.com', 'license': 'Apache License, Version 2.0', 'project_urls': FrozenDict(PROJECT_URLS), 'version': '0.0test0', 'classifiers': (*META_CLASSIFIERS, LINUX_CLASSIFIER, 'Programming Language :: Python', 'Programming Language :: Python :: 3', 'Programming Language :: Python :: 3.6', 'Programming Language :: Python :: 3.8', 'Qwerty :: Asdf :: Zxcv')}), address=address)

def test_setup_kwargs_plugin_with_readme(rule_runner: RuleRunner) -> None:
    if False:
        print('Hello World!')
    rule_runner.write_files({'runners/foobar_runner/BUILD': dedent('                python_distribution(\n                    provides=python_artifact(\n                        name="stackstorm-runner-foobar",\n                        description="Foobar runner for ST2",\n                        version_file="foobar_runner/__init__.py",\n                        classifiers=["Qwerty :: Asdf :: Zxcv"],\n                    ),\n                    dependencies=[\n                        "./foobar_runner",\n                    ],\n                    entry_points={\n                        "st2common.runners.runner": {\n                            "foobar": "foobar_runner.foobar_runner",\n                        },\n                    },\n                )\n                '), 'runners/foobar_runner/foobar_runner/__init__.py': '__version__ = "0.0test0"', 'runners/foobar_runner/README.rst': 'lorem ipsum'})
    address = Address('runners/foobar_runner')
    assert gen_setup_kwargs(address, rule_runner) == SetupKwargs(FrozenDict({'name': 'stackstorm-runner-foobar', 'description': 'Foobar runner for ST2', 'author': 'StackStorm', 'author_email': 'info@stackstorm.com', 'url': 'https://stackstorm.com', 'license': 'Apache License, Version 2.0', 'project_urls': FrozenDict(PROJECT_URLS), 'version': '0.0test0', 'long_description_content_type': 'text/x-rst', 'long_description': 'lorem ipsum', 'classifiers': (*META_CLASSIFIERS, LINUX_CLASSIFIER, 'Programming Language :: Python', 'Programming Language :: Python :: 3', 'Programming Language :: Python :: 3.6', 'Programming Language :: Python :: 3.8', 'Qwerty :: Asdf :: Zxcv')}), address=address)