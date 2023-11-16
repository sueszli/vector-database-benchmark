from nose.tools import assert_equal
from ..type_util import TypeUtil

class TestUtil:

    def test_is_iterable(self):
        if False:
            return 10
        assert_equal(TypeUtil.is_iterable('foo'), True)
        assert_equal(TypeUtil.is_iterable(7), False)

    def test_convert_to_list(self):
        if False:
            while True:
                i = 10
        assert_equal(isinstance(TypeUtil.convert_to_list('foo'), list), True)
        assert_equal(isinstance(TypeUtil.convert_to_list(7), list), False)