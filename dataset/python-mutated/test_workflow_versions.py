"""Unit tests for workflow version determination."""
from __future__ import print_function, unicode_literals
import pytest
from workflow.update import Version
from workflow.workflow import Workflow
from workflow.workflow3 import Workflow3
from .conftest import env, WORKFLOW_VERSION
from .util import VersionFile

def test_info_plist(infopl):
    if False:
        print('Hello World!')
    'Version from info.plist.'
    wf = Workflow3()
    assert wf.version == Version('1.1.1'), 'unexpected version'

def test_envvar(infopl):
    if False:
        while True:
            i = 10
    'Version from environment variable.'
    v = '1.1.2'
    with env(alfred_workflow_version=v):
        wf = Workflow3()
        assert wf.version == Version(v), 'unexpected version'
        wf = Workflow3(update_settings={'version': '1.1.3'})
        assert wf.version == Version(v), 'unexpected version'

def test_update_settings(infopl):
    if False:
        i = 10
        return i + 15
    'Version from update_settings.'
    v = '1.1.3'
    wf = Workflow3(update_settings={'version': v})
    assert wf.version == Version(v), 'unexpected version'

def test_versions_from_settings(alfred4, infopl2):
    if False:
        print('Hello World!')
    'Workflow: version from `update_settings`'
    vstr = '1.9.7'
    d = {'github_slug': 'deanishe/alfred-workflow', 'version': vstr}
    with env(alfred_workflow_version=None):
        wf = Workflow(update_settings=d)
        assert str(wf.version) == vstr
        assert isinstance(wf.version, Version)
        assert wf.version == Version(vstr)

def test_versions_from_file(alfred4, infopl2):
    if False:
        while True:
            i = 10
    'Workflow: version from `version` file'
    vstr = '1.9.7'
    with env(alfred_workflow_version=None):
        with VersionFile(vstr):
            wf = Workflow()
            assert str(wf.version) == vstr
            assert isinstance(wf.version, Version)
            assert wf.version == Version(vstr)

def test_versions_from_info(alfred4, infopl):
    if False:
        for i in range(10):
            print('nop')
    'Workflow: version from info.plist'
    with env(alfred_workflow_version=None):
        wf = Workflow()
        assert str(wf.version) == WORKFLOW_VERSION
        assert isinstance(wf.version, Version)
        assert wf.version == Version(WORKFLOW_VERSION)

def test_first_run_no_version(alfred4, infopl2):
    if False:
        i = 10
        return i + 15
    'Workflow: first_run fails on no version'
    with env(alfred_workflow_version=None):
        wf = Workflow()
        try:
            with pytest.raises(ValueError):
                wf.first_run
        finally:
            wf.reset()

def test_first_run_with_version(alfred4, infopl):
    if False:
        for i in range(10):
            print('nop')
    'Workflow: first_run'
    vstr = '1.9.7'
    with env(alfred_workflow_version=vstr):
        wf = Workflow()
        assert wf.first_run is True
        wf.reset()

def test_first_run_with_previous_run(alfred4, infopl):
    if False:
        print('Hello World!')
    'Workflow: first_run with previous run'
    vstr = '1.9.7'
    last_vstr = '1.9.6'
    with env(alfred_workflow_version=vstr):
        wf = Workflow()
        wf.set_last_version(last_vstr)
        assert wf.first_run is True
        assert wf.last_version_run == Version(last_vstr)
        wf.reset()

def test_last_version_empty(wf):
    if False:
        for i in range(10):
            print('nop')
    'Workflow: last_version_run empty'
    assert wf.last_version_run is None

def test_last_version_on(alfred4, infopl):
    if False:
        for i in range(10):
            print('nop')
    'Workflow: last_version_run not empty'
    vstr = '1.9.7'
    with env(alfred_workflow_version=vstr):
        wf = Workflow()
        wf.set_last_version(vstr)
        assert Version(vstr) == wf.last_version_run
        wf.reset()
    with env(alfred_workflow_version=vstr):
        wf = Workflow()
        wf.set_last_version()
        assert Version(vstr) == wf.last_version_run
        wf.reset()

def test_versions_no_version(alfred4, infopl2):
    if False:
        i = 10
        return i + 15
    'Workflow: version is `None`'
    with env(alfred_workflow_version=None):
        wf = Workflow()
        assert wf.version is None
        wf.reset()

def test_last_version_no_version(alfred4, infopl2):
    if False:
        return 10
    'Workflow: last_version no version'
    with env(alfred_workflow_version=None):
        wf = Workflow()
        assert wf.set_last_version() is False
        wf.reset()

def test_last_version_explicit_version(alfred4, infopl):
    if False:
        print('Hello World!')
    'Workflow: last_version explicit version'
    vstr = '1.9.6'
    wf = Workflow()
    assert wf.set_last_version(vstr) is True
    assert wf.last_version_run == Version(vstr)
    wf.reset()

def test_last_version_auto_version(alfred4, infopl):
    if False:
        return 10
    'Workflow: last_version auto version'
    vstr = '1.9.7'
    with env(alfred_workflow_version=vstr):
        wf = Workflow()
        assert wf.set_last_version() is True
        assert wf.last_version_run == Version(vstr)
        wf.reset()

def test_last_version_set_after_run(alfred4, infopl):
    if False:
        i = 10
        return i + 15
    'Workflow: last_version set after `run()`'
    vstr = '1.9.7'

    def cb(wf):
        if False:
            print('Hello World!')
        return
    with env(alfred_workflow_version=vstr):
        wf = Workflow()
        assert wf.last_version_run is None
        wf.run(cb)
        wf = Workflow()
        assert wf.last_version_run == Version(vstr)
        wf.reset()

def test_alfred_version(wf):
    if False:
        i = 10
        return i + 15
    'Workflow: alfred_version correct.'
    assert wf.alfred_version == Version('4.0')