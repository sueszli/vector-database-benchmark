import unittest
import numpy as np
from op import Operator
from paddle.base import core

class TestBeamSearchDecodeOp(unittest.TestCase):
    """unittest of beam_search_decode_op"""

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.scope = core.Scope()
        self.place = core.CPUPlace()

    def append_lod_tensor(self, tensor_array, lod, data):
        if False:
            return 10
        lod_tensor = core.LoDTensor()
        lod_tensor.set_lod(lod)
        lod_tensor.set(data, self.place)
        tensor_array.append(lod_tensor)

    def test_get_set(self):
        if False:
            while True:
                i = 10
        ids = self.scope.var('ids').get_lod_tensor_array()
        scores = self.scope.var('scores').get_lod_tensor_array()
        [self.append_lod_tensor(array, [[0, 1, 2], [0, 1, 2]], np.array([0, 0], dtype=dtype)) for (array, dtype) in ((ids, 'int64'), (scores, 'float32'))]
        [self.append_lod_tensor(array, [[0, 1, 2], [0, 2, 4]], np.array([2, 3, 4, 5], dtype=dtype)) for (array, dtype) in ((ids, 'int64'), (scores, 'float32'))]
        [self.append_lod_tensor(array, [[0, 2, 4], [0, 2, 2, 4, 4]], np.array([3, 1, 5, 4], dtype=dtype)) for (array, dtype) in ((ids, 'int64'), (scores, 'float32'))]
        [self.append_lod_tensor(array, [[0, 2, 4], [0, 1, 2, 3, 4]], np.array([1, 1, 3, 5], dtype=dtype)) for (array, dtype) in ((ids, 'int64'), (scores, 'float32'))]
        [self.append_lod_tensor(array, [[0, 2, 4], [0, 0, 0, 2, 2]], np.array([5, 1], dtype=dtype)) for (array, dtype) in ((ids, 'int64'), (scores, 'float32'))]
        sentence_ids = self.scope.var('sentence_ids').get_tensor()
        sentence_scores = self.scope.var('sentence_scores').get_tensor()
        beam_search_decode_op = Operator('beam_search_decode', Ids='ids', Scores='scores', SentenceIds='sentence_ids', SentenceScores='sentence_scores', beam_size=2, end_id=1)
        beam_search_decode_op.run(self.scope, self.place)
        expected_lod = [[0, 2, 4], [0, 4, 7, 12, 17]]
        self.assertEqual(sentence_ids.lod(), expected_lod)
        self.assertEqual(sentence_scores.lod(), expected_lod)
        expected_data = np.array([0, 2, 3, 1, 0, 2, 1, 0, 4, 5, 3, 5, 0, 4, 5, 3, 1], 'int64')
        np.testing.assert_array_equal(np.array(sentence_ids), expected_data)
        np.testing.assert_array_equal(np.array(sentence_scores), expected_data)

@unittest.skipIf(not core.is_compiled_with_cuda(), 'core is not compiled with CUDA')
class TestBeamSearchDecodeOpGPU(TestBeamSearchDecodeOp):

    def setUp(self):
        if False:
            return 10
        self.scope = core.Scope()
        self.place = core.CUDAPlace(0)
if __name__ == '__main__':
    unittest.main()