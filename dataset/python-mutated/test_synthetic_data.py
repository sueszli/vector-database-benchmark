import unittest
from typing import Optional
import numpy as np
from snorkel.labeling import LFAnalysis
from snorkel.synthetic.synthetic_data import generate_simple_label_matrix

class TestGenerateSimpleLabelMatrix(unittest.TestCase):
    """Testing the generate_simple_label_matrix function."""

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        'Set constants for the tests.'
        self.m = 10
        self.n = 1000

    def _test_generate_L(self, k: int, decimal: Optional[int]=2) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test generated label matrix L for consistency with P, Y.\n\n        This tests for consistency between the true conditional LF probabilities, P,\n        and the empirical ones computed from L and Y, where P, L, and Y are generated\n        by the generate_simple_label_matrix function.\n\n        Parameters\n        ----------\n        k\n            Cardinality\n        decimal\n            Number of decimals to check element-wise error, err < 1.5 * 10**(-decimal)\n        '
        np.random.seed(123)
        (P, Y, L) = generate_simple_label_matrix(self.n, self.m, k)
        P_emp = LFAnalysis(L).lf_empirical_probs(Y, k=k)
        np.testing.assert_array_almost_equal(P, P_emp, decimal=decimal)

    def test_generate_L(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test the generated dataset for consistency.'
        self._test_generate_L(2, decimal=1)

    def test_generate_L_multiclass(self) -> None:
        if False:
            return 10
        'Test the generated dataset for consistency with cardinality=3.'
        self._test_generate_L(3, decimal=1)
if __name__ == '__main__':
    unittest.main()