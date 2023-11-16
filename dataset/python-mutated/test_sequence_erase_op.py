import unittest
import numpy as np
from op_test import OpTest

def sequence_erase(in_seq, lod0, tokens):
    if False:
        while True:
            i = 10
    new_lod0 = []
    out_seq = []
    offset = 0
    for i in range(0, len(lod0)):
        num_out = 0
        for dat in in_seq[offset:offset + lod0[i]]:
            if dat not in tokens:
                out_seq.append(dat)
                num_out += 1
        offset += lod0[i]
        new_lod0.append(num_out)
    return (np.array(out_seq).astype('int32'), new_lod0)

class TestSequenceEraseOpInt32(OpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'sequence_erase'
        in_seq = np.random.randint(0, 10, (30, 1)).astype('int32')
        lod = [[9, 4, 11, 6]]
        tokens = [2, 3, 5]
        (out_seq, new_lod0) = sequence_erase(in_seq, lod[0], tokens)
        self.attrs = {'tokens': tokens}
        self.inputs = {'X': (in_seq, lod)}
        self.outputs = {'Out': (out_seq, [new_lod0])}

    def test_check_output(self):
        if False:
            while True:
                i = 10
        self.check_output()

class TestSequenceEraseOpInt32LoD2(OpTest):

    def setUp(self):
        if False:
            return 10
        self.op_type = 'sequence_erase'
        in_seq = np.random.randint(0, 10, (30, 1)).astype('int32')
        lod = [[1, 3], [9, 4, 11, 6]]
        tokens = [2, 3, 5]
        (out_seq, new_lod0) = sequence_erase(in_seq, lod[-1], tokens)
        self.attrs = {'tokens': tokens}
        self.inputs = {'X': (in_seq, lod)}
        self.outputs = {'Out': (out_seq, lod[:-1] + [new_lod0])}

    def test_check_output(self):
        if False:
            print('Hello World!')
        self.check_output()

class TestSequenceEraseOpInt64(OpTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'sequence_erase'
        in_seq = np.random.randint(0, 10, (30, 1)).astype('int64')
        lod = [[9, 4, 11, 6]]
        tokens = [2, 3, 5]
        (out_seq, new_lod0) = sequence_erase(in_seq, lod[0], tokens)
        self.attrs = {'tokens': tokens}
        self.inputs = {'X': (in_seq, lod)}
        self.outputs = {'Out': (out_seq, [new_lod0])}

    def test_check_output(self):
        if False:
            while True:
                i = 10
        self.check_output()

class TestSequenceEraseOpInt64SeqLen0(OpTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'sequence_erase'
        in_seq = np.random.randint(0, 10, (30, 1)).astype('int64')
        lod = [[0, 9, 0, 0, 10, 11, 0]]
        tokens = [2, 3, 5]
        (out_seq, new_lod0) = sequence_erase(in_seq, lod[0], tokens)
        self.attrs = {'tokens': tokens}
        self.inputs = {'X': (in_seq, lod)}
        self.outputs = {'Out': (out_seq, [new_lod0])}

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output()

class TestSequenceEraseOpEmpty(OpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'sequence_erase'
        in_seq = np.random.randint(0, 10, (30, 1)).astype('int32')
        lod = [[9, 4, 11, 6]]
        tokens = []
        (out_seq, new_lod0) = sequence_erase(in_seq, lod[0], tokens)
        self.attrs = {'tokens': tokens}
        self.inputs = {'X': (in_seq, lod)}
        self.outputs = {'Out': (out_seq, [new_lod0])}

    def test_check_output(self):
        if False:
            while True:
                i = 10
        self.check_output()
if __name__ == '__main__':
    unittest.main()