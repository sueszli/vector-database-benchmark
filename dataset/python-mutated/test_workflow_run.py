"""Unit tests for Workflow.run."""
from __future__ import print_function, unicode_literals
from StringIO import StringIO
import sys
import pytest
from workflow.workflow import Workflow
from conftest import env

def test_run_fails(infopl):
    if False:
        return 10
    'Run fails'
    wf = Workflow()

    def cb(wf2):
        if False:
            print('Hello World!')
        assert wf2 is wf
        raise ValueError('Have an error')
    wf.help_url = 'http://www.deanishe.net/alfred-workflow/'
    ret = wf.run(cb)
    assert ret == 1
    with env(alfred_workflow_name=None):
        wf = Workflow()
        wf.name
        ret = wf.run(cb)
        assert ret == 1
        wf = Workflow()
        wf.bundleid
        ret = wf.run(cb)
        assert ret == 1
    wf.reset()

def test_run_fails_with_xml_output(wf):
    if False:
        while True:
            i = 10
    'Run fails with XML output'
    error_text = 'Have an error'
    stdout = sys.stdout
    buf = StringIO()
    sys.stdout = buf

    def cb(wf2):
        if False:
            print('Hello World!')
        assert wf2 is wf
        raise ValueError(error_text)
    ret = wf.run(cb)
    sys.stdout = stdout
    output = buf.getvalue()
    buf.close()
    assert ret == 1
    assert error_text in output
    assert '<?xml' in output

def test_run_fails_with_plain_text_output(wf):
    if False:
        while True:
            i = 10
    'Run fails with plain text output'
    error_text = 'Have an error'
    stdout = sys.stdout
    buf = StringIO()
    sys.stdout = buf

    def cb(wf2):
        if False:
            i = 10
            return i + 15
        assert wf2 is wf
        raise ValueError(error_text)
    ret = wf.run(cb, text_errors=True)
    sys.stdout = stdout
    output = buf.getvalue()
    buf.close()
    assert ret == 1
    assert error_text in output
    assert '<?xml' not in output

def test_run_fails_borked_settings(wf):
    if False:
        while True:
            i = 10
    'Run fails with borked settings.json'
    with open(wf.settings_path, 'wb') as fp:
        fp.write('')

    def fake(wf):
        if False:
            i = 10
            return i + 15
        wf.settings
    ret = wf.run(fake)
    assert ret == 1

def test_run_okay(wf):
    if False:
        for i in range(10):
            print('nop')
    'Run okay'

    def cb(wf2):
        if False:
            print('Hello World!')
        assert wf2 is wf
    ret = wf.run(cb)
    assert ret == 0
if __name__ == '__main__':
    pytest.main([__file__])