import torch
from numpy.testing import assert_almost_equal
import numpy
from allennlp.common import Params
from allennlp.common.testing.test_case import AllenNlpTestCase
from allennlp.modules.matrix_attention import CosineMatrixAttention
from allennlp.modules.matrix_attention.matrix_attention import MatrixAttention

class TestCosineMatrixAttention(AllenNlpTestCase):

    def test_can_init_cosine(self):
        if False:
            return 10
        legacy_attention = MatrixAttention.from_params(Params({'type': 'cosine'}))
        isinstance(legacy_attention, CosineMatrixAttention)

    def test_cosine_similarity(self):
        if False:
            for i in range(10):
                print('nop')
        output = CosineMatrixAttention()(torch.FloatTensor([[[0, 0, 0], [4, 5, 6]], [[-7, -8, -9], [10, 11, 12]]]), torch.FloatTensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]))
        assert_almost_equal(output.numpy(), numpy.array([[[0, 0], [0.97, 1]], [[-1, -0.99], [0.99, 1]]]), decimal=2)