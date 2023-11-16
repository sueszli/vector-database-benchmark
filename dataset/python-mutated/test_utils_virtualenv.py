import logging
import os
import site
import sys
from pathlib import Path
from typing import List, Optional
import pytest
from pip._internal.utils import virtualenv

@pytest.mark.parametrize('real_prefix, base_prefix, expected', [(None, None, False), (None, sys.prefix, False), (None, 'not_sys_prefix', True), (sys.prefix, None, True), (sys.prefix, sys.prefix, True), (sys.prefix, 'not_sys_prefix', True), ('not_sys_prefix', None, True), ('not_sys_prefix', sys.prefix, True), ('not_sys_prefix', 'not_sys_prefix', True)])
def test_running_under_virtualenv(monkeypatch: pytest.MonkeyPatch, real_prefix: Optional[str], base_prefix: Optional[str], expected: bool) -> None:
    if False:
        while True:
            i = 10
    if real_prefix is None:
        monkeypatch.delattr(sys, 'real_prefix', raising=False)
    else:
        monkeypatch.setattr(sys, 'real_prefix', real_prefix, raising=False)
    if base_prefix is None:
        monkeypatch.delattr(sys, 'base_prefix', raising=False)
    else:
        monkeypatch.setattr(sys, 'base_prefix', base_prefix, raising=False)
    assert virtualenv.running_under_virtualenv() == expected

@pytest.mark.parametrize('under_virtualenv, no_global_file, expected', [(False, False, False), (False, True, False), (True, False, False), (True, True, True)])
def test_virtualenv_no_global_with_regular_virtualenv(monkeypatch: pytest.MonkeyPatch, tmpdir: Path, under_virtualenv: bool, no_global_file: bool, expected: bool) -> None:
    if False:
        print('Hello World!')
    monkeypatch.setattr(virtualenv, '_running_under_venv', lambda : False)
    monkeypatch.setattr(site, '__file__', os.fspath(tmpdir / 'site.py'))
    monkeypatch.setattr(virtualenv, '_running_under_legacy_virtualenv', lambda : under_virtualenv)
    if no_global_file:
        (tmpdir / 'no-global-site-packages.txt').touch()
    assert virtualenv.virtualenv_no_global() == expected

@pytest.mark.parametrize('pyvenv_cfg_lines, under_venv, expect_no_global, expect_warning', [(None, False, False, False), (None, True, True, True), (['home = <we do not care>', 'include-system-site-packages = true', 'version = <we do not care>'], True, False, False), (['home = <we do not care>', 'include-system-site-packages = false', 'version = <we do not care>'], True, True, False)])
def test_virtualenv_no_global_with_pep_405_virtual_environment(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, pyvenv_cfg_lines: Optional[List[str]], under_venv: bool, expect_no_global: bool, expect_warning: bool) -> None:
    if False:
        i = 10
        return i + 15
    monkeypatch.setattr(virtualenv, '_running_under_legacy_virtualenv', lambda : False)
    monkeypatch.setattr(virtualenv, '_get_pyvenv_cfg_lines', lambda : pyvenv_cfg_lines)
    monkeypatch.setattr(virtualenv, '_running_under_venv', lambda : under_venv)
    with caplog.at_level(logging.WARNING):
        assert virtualenv.virtualenv_no_global() == expect_no_global
    if expect_warning:
        assert caplog.records
        message = caplog.records[-1].getMessage().lower()
        assert "could not access 'pyvenv.cfg'" in message
        assert 'assuming global site-packages is not accessible' in message

@pytest.mark.parametrize('contents, expected', [(None, None), ('', []), ('a = b\nc = d\n', ['a = b', 'c = d']), ('a = b\nc = d', ['a = b', 'c = d'])])
def test_get_pyvenv_cfg_lines_for_pep_405_virtual_environment(monkeypatch: pytest.MonkeyPatch, tmpdir: Path, contents: Optional[str], expected: Optional[List[str]]) -> None:
    if False:
        return 10
    monkeypatch.setattr(sys, 'prefix', str(tmpdir))
    if contents is not None:
        tmpdir.joinpath('pyvenv.cfg').write_text(contents)
    assert virtualenv._get_pyvenv_cfg_lines() == expected