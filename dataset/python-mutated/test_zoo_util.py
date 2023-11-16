import pytest
from bigdl.dllib.utils.common import get_node_and_core_number
from test.bigdl.test_zoo_utils import ZooTestCase
from bigdl.dllib.utils.file_utils import set_core_number

class TestUtil(ZooTestCase):

    def test_set_core_num(self):
        if False:
            return 10
        (_, core_num) = get_node_and_core_number()
        set_core_number(core_num + 1)
        (_, new_core_num) = get_node_and_core_number()
        assert new_core_num == core_num + 1, 'set_core_num failed, set the core number to be {} but got {}'.format(core_num + 1, new_core_num)
        set_core_number(core_num)
if __name__ == '__main__':
    pytest.main([__file__])