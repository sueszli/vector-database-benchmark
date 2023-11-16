import pytest
from yellowbrick.target.binning import *
from yellowbrick.datasets import load_occupancy
from tests.base import VisualTestCase
try:
    import pandas as pd
except ImportError:
    pd = None

class TestBalancedBinningReference(VisualTestCase):
    """
    Test the BalancedBinningReference visualizer
    """

    def test_numpy_bins(self):
        if False:
            return 10
        '\n        Test Histogram on a NumPy array\n        '
        data = load_occupancy(return_dataset=True)
        (X, y) = data.to_numpy()
        visualizer = BalancedBinningReference()
        visualizer.fit(y)
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=0.5)

    @pytest.mark.skipif(pd is None, reason='pandas is required')
    def test_pandas_bins(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test Histogram on a Pandas Dataframe\n        '
        data = load_occupancy(return_dataset=True)
        (X, y) = data.to_pandas()
        visualizer = BalancedBinningReference()
        visualizer.fit(y)
        visualizer.finalize()
        self.assert_images_similar(visualizer, tol=0.5)

    def test_quick_method(self):
        if False:
            while True:
                i = 10
        '\n        Test the quick method with producing a valid visualization\n        '
        data = load_occupancy(return_dataset=True)
        (_, y) = data.to_numpy()
        visualizer = balanced_binning_reference(y, show=False)
        assert isinstance(visualizer, BalancedBinningReference)
        self.assert_images_similar(visualizer, tol=0.5)