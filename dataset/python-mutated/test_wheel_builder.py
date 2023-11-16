import logging
import os
from pathlib import Path
from typing import Optional, cast
import pytest
from pip._internal import wheel_builder
from pip._internal.models.link import Link
from pip._internal.operations.build.wheel_legacy import format_command_result
from pip._internal.req.req_install import InstallRequirement
from pip._internal.vcs.git import Git
from tests.lib import _create_test_package

@pytest.mark.parametrize('s, expected', [('pip-18.0', True), ('foo-2-2', True), ('im-valid', True), ('invalid', False), ('im_invalid', False)])
def test_contains_egg_info(s: str, expected: bool) -> None:
    if False:
        print('Hello World!')
    result = wheel_builder._contains_egg_info(s)
    assert result == expected

class ReqMock:

    def __init__(self, name: str='pendulum', is_wheel: bool=False, editable: bool=False, link: Optional[Link]=None, constraint: bool=False, source_dir: Optional[str]='/tmp/pip-install-123/pendulum', use_pep517: bool=True, supports_pyproject_editable: bool=False) -> None:
        if False:
            while True:
                i = 10
        self.name = name
        self.is_wheel = is_wheel
        self.editable = editable
        self.link = link
        self.constraint = constraint
        self.source_dir = source_dir
        self.use_pep517 = use_pep517
        self._supports_pyproject_editable = supports_pyproject_editable

    def supports_pyproject_editable(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self._supports_pyproject_editable

@pytest.mark.parametrize('req, expected', [(ReqMock(use_pep517=True), True), (ReqMock(use_pep517=False), True), (ReqMock(constraint=True), False), (ReqMock(is_wheel=True), False), (ReqMock(editable=True, use_pep517=False), False), (ReqMock(editable=True, use_pep517=True, supports_pyproject_editable=True), True), (ReqMock(editable=True, use_pep517=True, supports_pyproject_editable=False), False), (ReqMock(source_dir=None), False), (ReqMock(link=Link('git+https://g.c/org/repo'), use_pep517=True), True), (ReqMock(link=Link('git+https://g.c/org/repo'), use_pep517=False), True)])
def test_should_build_for_install_command(req: ReqMock, expected: bool) -> None:
    if False:
        i = 10
        return i + 15
    should_build = wheel_builder.should_build_for_install_command(cast(InstallRequirement, req))
    assert should_build is expected

@pytest.mark.parametrize('req, expected', [(ReqMock(), True), (ReqMock(constraint=True), False), (ReqMock(is_wheel=True), False), (ReqMock(editable=True, use_pep517=False), True), (ReqMock(editable=True, use_pep517=True), True), (ReqMock(source_dir=None), True), (ReqMock(link=Link('git+https://g.c/org/repo')), True)])
def test_should_build_for_wheel_command(req: ReqMock, expected: bool) -> None:
    if False:
        i = 10
        return i + 15
    should_build = wheel_builder.should_build_for_wheel_command(cast(InstallRequirement, req))
    assert should_build is expected

@pytest.mark.parametrize('req, expected', [(ReqMock(editable=True, use_pep517=False), False), (ReqMock(editable=True, use_pep517=True), False), (ReqMock(source_dir=None), False), (ReqMock(link=Link('git+https://g.c/org/repo')), False), (ReqMock(link=Link('https://g.c/dist.tgz')), False), (ReqMock(link=Link('https://g.c/dist-2.0.4.tgz')), True)])
def test_should_cache(req: ReqMock, expected: bool) -> None:
    if False:
        for i in range(10):
            print('nop')
    assert wheel_builder._should_cache(cast(InstallRequirement, req)) is expected

def test_should_cache_git_sha(tmpdir: Path) -> None:
    if False:
        while True:
            i = 10
    repo_path = os.fspath(_create_test_package(tmpdir, name='mypkg'))
    commit = Git.get_revision(repo_path)
    url = 'git+https://g.c/o/r@' + commit + '#egg=mypkg'
    req = ReqMock(link=Link(url), source_dir=repo_path)
    assert wheel_builder._should_cache(cast(InstallRequirement, req))
    url = 'git+https://g.c/o/r@master#egg=mypkg'
    req = ReqMock(link=Link(url), source_dir=repo_path)
    assert not wheel_builder._should_cache(cast(InstallRequirement, req))

def test_format_command_result__INFO(caplog: pytest.LogCaptureFixture) -> None:
    if False:
        return 10
    caplog.set_level(logging.INFO)
    actual = format_command_result(command_args=['arg1', 'second arg'], command_output='output line 1\noutput line 2\n')
    assert actual.splitlines() == ["Command arguments: arg1 'second arg'", 'Command output: [use --verbose to show]']

@pytest.mark.parametrize('command_output', ['output line 1\noutput line 2\n', 'output line 1\noutput line 2'])
def test_format_command_result__DEBUG(caplog: pytest.LogCaptureFixture, command_output: str) -> None:
    if False:
        return 10
    caplog.set_level(logging.DEBUG)
    actual = format_command_result(command_args=['arg1', 'arg2'], command_output=command_output)
    assert actual.splitlines() == ['Command arguments: arg1 arg2', 'Command output:', 'output line 1', 'output line 2']

@pytest.mark.parametrize('log_level', ['DEBUG', 'INFO'])
def test_format_command_result__empty_output(caplog: pytest.LogCaptureFixture, log_level: str) -> None:
    if False:
        i = 10
        return i + 15
    caplog.set_level(log_level)
    actual = format_command_result(command_args=['arg1', 'arg2'], command_output='')
    assert actual.splitlines() == ['Command arguments: arg1 arg2', 'Command output: None']