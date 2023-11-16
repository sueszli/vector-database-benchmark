"""Unit tests for serializers."""
from __future__ import print_function, unicode_literals
import pytest

def test_unicode_paths(wf):
    if False:
        while True:
            i = 10
    'Workflow paths are Unicode'
    s = b'test.txt'
    u = u'über.txt'
    assert isinstance(wf.datadir, unicode)
    assert isinstance(wf.datafile(s), unicode)
    assert isinstance(wf.datafile(u), unicode)
    assert isinstance(wf.cachedir, unicode)
    assert isinstance(wf.cachefile(s), unicode)
    assert isinstance(wf.cachefile(u), unicode)
    assert isinstance(wf.workflowdir, unicode)
    assert isinstance(wf.workflowfile(s), unicode)
    assert isinstance(wf.workflowfile(u), unicode)
if __name__ == '__main__':
    pytest.main([__file__])