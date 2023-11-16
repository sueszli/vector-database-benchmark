import math
import torch
from numpy.testing import assert_almost_equal
import numpy
from allennlp.common import Params
from allennlp.common.testing.test_case import AllenNlpTestCase
from allennlp.modules.matrix_attention import ScaledDotProductMatrixAttention
from allennlp.modules.matrix_attention.matrix_attention import MatrixAttention

class TestScaledDotProductMatrixAttention(AllenNlpTestCase):

    def test_can_init_dot(self):
        if False:
            for i in range(10):
                print('nop')
        legacy_attention = MatrixAttention.from_params(Params({'type': 'scaled_dot_product'}))
        isinstance(legacy_attention, ScaledDotProductMatrixAttention)

    def test_dot_product_similarity(self):
        if False:
            for i in range(10):
                print('nop')
        output = ScaledDotProductMatrixAttention()(torch.FloatTensor([[[0, 0, 0], [4, 5, 6]], [[-7, -8, -9], [10, 11, 12]]]), torch.FloatTensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]))
        assert_almost_equal(output.numpy(), numpy.array([[[0, 0], [32, 77]], [[-194, -266], [266, 365]]]) / math.sqrt(3), decimal=2)