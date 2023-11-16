import struct
import unittest
import numpy as np
from op_test import OpTest

class TestLookupTableDequantOp(OpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'lookup_table_dequant'
        table = np.random.random((17, 32)).astype('float32')
        ids = np.random.randint(0, 17, 4).astype('int64')
        ids_expand = np.expand_dims(ids, axis=1)
        self.inputs = {'W': table, 'Ids': ids_expand}
        output = []
        for id in ids:
            tmp = []
            (min, max) = (table[id][0], table[id][1])
            for val in table[id][2:]:
                tmp += [int(x) * (max - min) / pow(2, 8) + min for x in bytearray(struct.pack('f', val))]
            output.append(tmp)
        self.outputs = {'Out': np.asarray(output, dtype='float32')}

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output()
if __name__ == '__main__':
    unittest.main()