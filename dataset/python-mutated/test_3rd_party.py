"""
Tests for compatibility against other Python modules.
"""
import pytest
from hypothesis import given
from .strategies import simple_classes
cloudpickle = pytest.importorskip('cloudpickle')

class TestCloudpickleCompat:
    """
    Tests for compatibility with ``cloudpickle``.
    """

    @given(simple_classes())
    def test_repr(self, cls):
        if False:
            while True:
                i = 10
        '\n        attrs instances can be pickled and un-pickled with cloudpickle.\n        '
        inst = cls()
        pkl = cloudpickle.dumps(inst)
        cloudpickle.loads(pkl)