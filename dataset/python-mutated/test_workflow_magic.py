"""Unit tests for magic arguments."""
from __future__ import print_function
import os
import pytest
from workflow import Workflow
from .conftest import env
from .util import VersionFile, WorkflowMock

def test_list_magic(infopl):
    if False:
        print('Hello World!')
    'Magic: list magic'
    with WorkflowMock(['script', 'workflow:magic']) as c:
        wf = Workflow()
        wf.args
        assert not c.cmd
        wf.reset()

def test_version_magic(infopl):
    if False:
        for i in range(10):
            print('nop')
    'Magic: version magic'
    vstr = '1.9.7'
    with env(alfred_workflow_version=None):
        with WorkflowMock(['script', 'workflow:version']) as c:
            with VersionFile(vstr):
                wf = Workflow()
                wf.args
                assert not c.cmd
                wf.reset()
        with WorkflowMock(['script', 'workflow:version']) as c:
            wf = Workflow()
            wf.args
            assert not c.cmd
            wf.reset()
    with env(alfred_workflow_version=vstr):
        with WorkflowMock(['script', 'workflow:version']) as c:
            wf = Workflow()
            wf.args
            assert not c.cmd
            wf.reset()

def test_openhelp(infopl):
    if False:
        return 10
    'Magic: open help URL'
    url = 'http://www.deanishe.net/alfred-workflow/'
    with WorkflowMock(['script', 'workflow:help']) as c:
        wf = Workflow(help_url=url)
        wf.args
        assert c.cmd == ['open', url]
        wf.reset()

def test_openhelp_no_url(infopl):
    if False:
        i = 10
        return i + 15
    'Magic: no help URL'
    with WorkflowMock(['script', 'workflow:help']) as c:
        wf = Workflow()
        wf.args
        assert not c.cmd
        wf.reset()

def test_openlog(infopl):
    if False:
        while True:
            i = 10
    'Magic: open logfile'
    with WorkflowMock(['script', 'workflow:openlog']) as c:
        wf = Workflow()
        wf.args
        assert c.cmd == ['open', wf.logfile]
        wf.reset()

def test_cachedir(infopl):
    if False:
        print('Hello World!')
    'Magic: open cachedir'
    with WorkflowMock(['script', 'workflow:opencache']) as c:
        wf = Workflow()
        wf.args
        assert c.cmd == ['open', wf.cachedir]
        wf.reset()

def test_datadir(infopl):
    if False:
        i = 10
        return i + 15
    'Magic: open datadir'
    with WorkflowMock(['script', 'workflow:opendata']) as c:
        wf = Workflow()
        wf.args
        assert c.cmd == ['open', wf.datadir]
        wf.reset()

def test_workflowdir(infopl):
    if False:
        for i in range(10):
            print('nop')
    'Magic: open workflowdir'
    with WorkflowMock(['script', 'workflow:openworkflow']) as c:
        wf = Workflow()
        wf.args
        assert c.cmd == ['open', wf.workflowdir]
        wf.reset()

def test_open_term(infopl):
    if False:
        return 10
    'Magic: open Terminal'
    with WorkflowMock(['script', 'workflow:openterm']) as c:
        wf = Workflow()
        wf.args
        assert c.cmd == ['open', '-a', 'Terminal', wf.workflowdir]
        wf.reset()

def test_delete_data(infopl):
    if False:
        i = 10
        return i + 15
    'Magic: delete data'
    with WorkflowMock(['script', 'workflow:deldata']):
        wf = Workflow()
        testpath = wf.datafile('file.test')
        with open(testpath, 'wb') as fp:
            fp.write('test!')
        assert os.path.exists(testpath)
        wf.args
        assert not os.path.exists(testpath)
        wf.reset()

def test_delete_cache(infopl):
    if False:
        return 10
    'Magic: delete cache'
    with WorkflowMock(['script', 'workflow:delcache']):
        wf = Workflow()
        testpath = wf.cachefile('file.test')
        with open(testpath, 'wb') as fp:
            fp.write('test!')
        assert os.path.exists(testpath)
        wf.args
        assert not os.path.exists(testpath)
        wf.reset()

def test_reset(infopl):
    if False:
        return 10
    'Magic: reset'
    with WorkflowMock(['script', 'workflow:reset']):
        wf = Workflow()
        wf.settings['key'] = 'value'
        datatest = wf.datafile('data.test')
        cachetest = wf.cachefile('cache.test')
        settings_path = wf.datafile('settings.json')
        for p in (datatest, cachetest):
            with open(p, 'wb') as file_obj:
                file_obj.write('test!')
        for p in (datatest, cachetest, settings_path):
            assert os.path.exists(p)
        wf.args
        for p in (datatest, cachetest, settings_path):
            assert not os.path.exists(p)
        wf.reset()

def test_delete_settings(infopl):
    if False:
        while True:
            i = 10
    'Magic: delete settings'
    with WorkflowMock(['script', 'workflow:delsettings']):
        wf = Workflow()
        wf.settings['key'] = 'value'
        assert os.path.exists(wf.settings_path)
        wf2 = Workflow()
        assert wf2.settings['key'] == 'value'
        wf.args
        wf3 = Workflow()
        assert 'key' not in wf3.settings
        wf.reset()

def test_folding(infopl):
    if False:
        for i in range(10):
            print('nop')
    'Magic: folding'
    with WorkflowMock(['script', 'workflow:foldingdefault']):
        wf = Workflow()
        wf.args
        assert wf.settings.get('__workflow_diacritic_folding') is None
    with WorkflowMock(['script', 'workflow:foldingon']):
        wf = Workflow()
        wf.args
        assert wf.settings.get('__workflow_diacritic_folding') is True
    with WorkflowMock(['script', 'workflow:foldingdefault']):
        wf = Workflow()
        wf.args
        assert wf.settings.get('__workflow_diacritic_folding') is None
    with WorkflowMock(['script', 'workflow:foldingoff']):
        wf = Workflow()
        wf.args
        assert wf.settings.get('__workflow_diacritic_folding') is False
        wf.reset()

def test_prereleases(infopl):
    if False:
        while True:
            i = 10
    'Magic: prereleases'
    with WorkflowMock(['script', 'workflow:prereleases']):
        wf = Workflow()
        wf.args
        assert wf.settings.get('__workflow_prereleases') is True
        assert wf.prereleases is True
        wf.reset()
    with WorkflowMock(['script', 'workflow:noprereleases']):
        wf = Workflow()
        wf.args
        assert wf.settings.get('__workflow_prereleases') is False
        assert wf.prereleases is False
        wf.reset()

def test_update_settings_override_magic_prereleases(infopl):
    if False:
        while True:
            i = 10
    'Magic: pre-release updates can be overridden by `update_settings`'
    with WorkflowMock(['script', 'workflow:prereleases']):
        d = {'prereleases': True}
        wf = Workflow(update_settings=d)
        wf.args
        assert wf.settings.get('__workflow_prereleases') is True
        assert wf.prereleases is True
        wf.reset()
    with WorkflowMock(['script', 'workflow:noprereleases']):
        wf = Workflow(update_settings=d)
        wf.args
        assert wf.settings.get('__workflow_prereleases') is False
        assert wf.prereleases is True
        wf.reset()
if __name__ == '__main__':
    pytest.main([__file__])