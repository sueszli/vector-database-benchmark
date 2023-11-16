"""Collection of tests around character encodings."""
import codecs
import locale
import sys
import pytest
PY3 = sys.version_info[0] == 3

@pytest.mark.skipif(not PY3, reason='Only necessary on Python3')
def test_not_ascii():
    if False:
        for i in range(10):
            print('nop')
    'Make sure that the systems preferred encoding is not `ascii`.\n\n    Otherwise `click` is raising a RuntimeError for Python3. For a detailed\n    description of this very problem please consult the following gist:\n    https://gist.github.com/hackebrot/937245251887197ef542\n\n    This test also checks that `tox.ini` explicitly copies the according\n    system environment variables to the test environments.\n    '
    try:
        preferred_encoding = locale.getpreferredencoding()
        fs_enc = codecs.lookup(preferred_encoding).name
    except Exception:
        fs_enc = 'ascii'
    assert fs_enc != 'ascii'