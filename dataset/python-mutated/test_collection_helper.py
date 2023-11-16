import pytest
from ding.utils.collection_helper import iter_mapping

@pytest.mark.unittest
class TestCollectionHelper:

    def test_iter_mapping(self):
        if False:
            print('Hello World!')
        _iter = iter_mapping([1, 2, 3, 4, 5], lambda x: x ** 2)
        assert not isinstance(_iter, list)
        assert list(_iter) == [1, 4, 9, 16, 25]