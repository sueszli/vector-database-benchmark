"""Tests for slots."""
from pytype.tests import test_base

class SlotsTest(test_base.BaseTest):
    """Tests for __slots__."""

    def test_slot_with_unicode(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      class Foo:\n        __slots__ = (u"fo\\xf6", u"b\\xe4r", "baz")\n      Foo().baz = 3\n    ')

    def test_slot_with_bytes(self):
        if False:
            i = 10
            return i + 15
        self.CheckWithErrors('\n      class Foo:  # bad-slots\n        __slots__ = (b"x",)\n    ')
if __name__ == '__main__':
    test_base.main()