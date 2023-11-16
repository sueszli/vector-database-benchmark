import unittest
import numpy as np
import pandas as pd
import os
from coremltools._deps import _HAS_SKLEARN
from coremltools.models.utils import evaluate_regressor, _macos_version, _is_macos
import pytest
if _HAS_SKLEARN:
    from sklearn.ensemble import RandomForestRegressor
    from coremltools.converters import sklearn as skl_converter

@unittest.skipIf(not _HAS_SKLEARN, 'Missing sklearn. Skipping tests.')
class RandomForestRegressorBostonHousingScikitNumericTest(unittest.TestCase):
    """
    Unit test class for testing scikit-learn converter and running both models
    """

    @classmethod
    def setUpClass(self):
        if False:
            return 10
        '\n        Set up the unit test by loading the dataset and training a model.\n        '
        from sklearn.datasets import load_boston
        scikit_data = load_boston()
        self.scikit_data = scikit_data
        self.X = scikit_data.data.astype('f').astype('d')
        self.target = scikit_data.target
        self.feature_names = scikit_data.feature_names
        self.output_name = 'target'

    def _check_metrics(self, metrics, params={}):
        if False:
            i = 10
            return i + 15
        '\n        Check the metrics\n        '
        self.assertAlmostEquals(metrics['rmse'], 0.0, delta=1e-05, msg='Failed case %s. Results %s' % (params, metrics))
        self.assertAlmostEquals(metrics['max_error'], 0.0, delta=1e-05, msg='Failed case %s. Results %s' % (params, metrics))

    def _train_convert_evaluate_assert(self, **scikit_params):
        if False:
            print('Hello World!')
        '\n        Train a scikit-learn model, convert it and then evaluate it with CoreML\n        '
        scikit_model = RandomForestRegressor(random_state=1, **scikit_params)
        scikit_model.fit(self.X, self.target)
        spec = skl_converter.convert(scikit_model, self.feature_names, self.output_name)
        if _is_macos() and _macos_version() >= (10, 13):
            df = pd.DataFrame(self.X, columns=self.feature_names)
            df['prediction'] = scikit_model.predict(self.X)
            metrics = evaluate_regressor(spec, df, verbose=False)
            self._check_metrics(metrics, scikit_params)

    def test_boston_housing_simple_regression(self):
        if False:
            i = 10
            return i + 15
        self._train_convert_evaluate_assert()

    def test_boston_housing_float_double_corner_case(self):
        if False:
            while True:
                i = 10
        self._train_convert_evaluate_assert(max_depth=13)

    @pytest.mark.slow
    def test_boston_housing_parameter_stress_test(self):
        if False:
            print('Hello World!')
        options = dict(criterion=['mse'], n_estimators=[1, 5, 10], max_depth=[1, 5], min_samples_split=[2, 10, 0.5], min_samples_leaf=[1, 5], min_weight_fraction_leaf=[0.0, 0.5], max_leaf_nodes=[None, 20], min_impurity_decrease=[1e-07, 0.1, 0.0])
        import itertools
        product = itertools.product(*options.values())
        args = [dict(zip(options.keys(), p)) for p in product]
        print('Testing a total of %s cases. This could take a while' % len(args))
        for (it, arg) in enumerate(args):
            self._train_convert_evaluate_assert(**arg)