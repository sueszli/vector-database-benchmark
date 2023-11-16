"""Test sphinx.ext.duration extension."""
import re
import pytest

@pytest.mark.sphinx('dummy', testroot='basic', confoverrides={'extensions': ['sphinx.ext.duration']})
def test_githubpages(app, status, warning):
    if False:
        i = 10
        return i + 15
    app.build()
    assert 'slowest reading durations' in status.getvalue()
    assert re.search('\\d+\\.\\d{3} index\n', status.getvalue())