from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import turicreate as _tc
import pytest
pytestmark = [pytest.mark.minimal]

@pytest.mark.skipif(not _tc._deps.is_minimal_pkg(), reason='skip when testing full pkg')
class TestMinimalPackage(object):
    """ test minimal package for toolkits.
    well, other toolkits are too hard to setup
    """

    def test_audio_classifier(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(ImportError, match='.*pip install --force-reinstall turicreate==.*'):
            _tc.load_audio('./dummy/audio')