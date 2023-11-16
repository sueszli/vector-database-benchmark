"""Unit tests for sys.path manipulation."""
from __future__ import print_function, unicode_literals
import os
import sys
import pytest
from workflow.workflow import Workflow
LIBS = [os.path.join(os.path.dirname(__file__), b'lib')]

def test_additional_libs(alfred4, infopl):
    if False:
        return 10
    'Additional libraries'
    wf = Workflow(libraries=LIBS)
    for path in LIBS:
        assert path in sys.path
    assert sys.path[0:len(LIBS)] == LIBS
    import youcanimportme
    youcanimportme.noop()
    wf.reset()
if __name__ == '__main__':
    pytest.main([__file__])