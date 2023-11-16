"""
Tests for the ClassBalance visualizer
"""
import pytest
from yellowbrick.target.class_balance import *
from yellowbrick.datasets import load_occupancy
from yellowbrick.exceptions import YellowbrickValueError
from tests.fixtures import Dataset, Split
from tests.base import VisualTestCase
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split as tts
try:
    import pandas as pd
except ImportError:
    pd = None

def make_fixture(binary=False, balanced=False, split=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Make a dataset for testing ClassBalance based on the specified params.\n    '
    kwargs = {'n_samples': 100, 'n_features': 20, 'n_informative': 8, 'n_redundant': 2, 'n_clusters_per_class': 1, 'random_state': 89092}
    if binary:
        kwargs['n_classes'] = 2
        kwargs['weights'] = None if balanced else [0.3, 0.7]
    else:
        kwargs['n_classes'] = 5
        kwargs['weights'] = None if balanced else [0.1, 0.2, 0.4, 0.2, 0.01]
    (X, y) = make_classification(**kwargs)
    if split:
        (X_train, X_test, y_train, y_test) = tts(X, y, test_size=0.2, random_state=101)
        return Dataset(Split(X_train, X_test), Split(y_train, y_test))
    return Dataset(X, y)

class TestClassBalance(VisualTestCase):
    """
    Test ClassBalance visualizer
    """

    def test_signature_exception(self):
        if False:
            while True:
                i = 10
        '\n        An exception is raised if X and y are put into the visualizer\n        '
        oz = ClassBalance()
        dataset = make_fixture(split=False)
        message = 'fit has changed to only require a 1D array, y'
        with pytest.raises(YellowbrickValueError, match=message):
            oz.fit(dataset.X, dataset.y)

    def test_invalid_target(self):
        if False:
            print('Hello World!')
        '\n        A value error should be raised on invalid train or test target\n        '
        y_valid = np.random.randint(2, size=100)
        y_invalid = np.random.uniform(size=100)
        oz = ClassBalance()
        with pytest.raises(YellowbrickValueError):
            oz.fit(y_invalid)
        with pytest.raises(YellowbrickValueError):
            oz.fit(y_valid, y_invalid)

    def test_class_names_must_match(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Assert error raised when more classes are in data than specified\n        '
        oz = ClassBalance(labels=['a', 'b', 'c'])
        dataset = make_fixture(binary=False, split=False)
        with pytest.raises(YellowbrickValueError):
            oz.fit(dataset.y)

    def test_binary_balance(self):
        if False:
            while True:
                i = 10
        '\n        Test binary classification in balance mode\n        '
        dataset = make_fixture(binary=True, split=False)
        oz = ClassBalance()
        assert oz.fit(dataset.y) is oz
        assert oz._mode == BALANCE
        self.assert_images_similar(oz)

    def test_binary_compare(self):
        if False:
            while True:
                i = 10
        '\n        Test binary classification in compare mode\n        '
        dataset = make_fixture(binary=True, split=True)
        oz = ClassBalance()
        assert oz.fit(dataset.y.train, dataset.y.test) is oz
        assert oz._mode == COMPARE
        self.assert_images_similar(oz)

    def test_multiclass_balance(self):
        if False:
            i = 10
            return i + 15
        '\n        Test multiclass classification in balance mode\n        '
        dataset = make_fixture(binary=False, split=False)
        oz = ClassBalance()
        assert oz.fit(dataset.y) is oz
        assert oz._mode == BALANCE
        self.assert_images_similar(oz)

    def test_multiclass_compare(self):
        if False:
            while True:
                i = 10
        '\n        Test multiclass classification in compare mode\n        '
        dataset = make_fixture(binary=False, split=True)
        oz = ClassBalance()
        assert oz.fit(dataset.y.train, dataset.y.test) is oz
        assert oz._mode == COMPARE
        self.assert_images_similar(oz)

    @pytest.mark.skipif(pd is None, reason='test requires pandas')
    def test_pandas_occupancy_balance(self):
        if False:
            print('Hello World!')
        '\n        Test pandas data frame with string target in balance mode\n        '
        data = load_occupancy(return_dataset=True)
        (X, y) = data.to_pandas()
        oz = ClassBalance()
        assert oz.fit(y) is oz
        self.assert_images_similar(oz)

    def test_numpy_occupancy_balance(self):
        if False:
            return 10
        '\n        Test NumPy arrays with string target in balance mode\n        '
        data = load_occupancy(return_dataset=True)
        (X, y) = data.to_numpy()
        oz = ClassBalance()
        assert oz.fit(y) is oz
        self.assert_images_similar(oz)

    @pytest.mark.skipif(pd is None, reason='test requires pandas')
    def test_pandas_occupancy_compare(self):
        if False:
            print('Hello World!')
        '\n        Test pandas data frame with string target in compare mode\n        '
        data = load_occupancy(return_dataset=True)
        (X, y) = data.to_pandas()
        (_, _, y_train, y_test) = tts(X, y, test_size=0.4, random_state=2242)
        oz = ClassBalance()
        assert oz.fit(y_train, y_test) is oz
        self.assert_images_similar(oz, tol=0.5)

    def test_numpy_occupancy_compare(self):
        if False:
            print('Hello World!')
        '\n        Test NumPy arrays with string target in compare mode\n        '
        data = load_occupancy(return_dataset=True)
        (X, y) = data.to_numpy()
        (_, _, y_train, y_test) = tts(X, y, test_size=0.4, random_state=2242)
        oz = ClassBalance()
        assert oz.fit(y_train, y_test) is oz
        self.assert_images_similar(oz, tol=0.5)

    def test_quick_method(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test the quick method producing a valid visualization\n        '
        dataset = make_fixture(binary=False, split=False)
        viz = class_balance(dataset.y, show=False)
        assert isinstance(viz, ClassBalance)
        self.assert_images_similar(viz, tol=0.5)

    def test_quick_method_with_splits(self):
        if False:
            return 10
        '\n        Test the quick method works with train and test splits\n        '
        dataset = make_fixture(binary=False, split=True)
        viz = class_balance(dataset.y.train, dataset.y.test, show=False)
        assert isinstance(viz, ClassBalance)
        self.assert_images_similar(viz)