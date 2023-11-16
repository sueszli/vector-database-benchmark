import unittest
import paddle
paddle.enable_static()
from paddle.distributed.fleet.runtime.the_one_ps import Table

class TestTable(unittest.TestCase):

    def test_table_tensor(self):
        if False:
            print('Hello World!')
        table = Table()
        table.id = 1001
        table.table_class = 'SPARSE_TABLE'
        table.shard_num = -1
        table.type = None
        table.accessor = None
        table.common = None
        table.tensor = None
        pt = '  downpour_table_param {table_id: 1001 table_class: "SPARSE_TABLE" shard_num: -1 type: None\n\n  }'
        self.assertEqual(table.to_string(0), pt)
if __name__ == '__main__':
    unittest.main()