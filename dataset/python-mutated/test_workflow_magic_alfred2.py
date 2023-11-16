"""Unit tests for Alfred 2 magic argument handling."""
from __future__ import print_function
import pytest
from workflow import Workflow
from .conftest import env
from .util import VersionFile, WorkflowMock

def test_version_magic(infopl2):
    if False:
        return 10
    'Magic: version magic (Alfred 2)'
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
if __name__ == '__main__':
    pytest.main([__file__])