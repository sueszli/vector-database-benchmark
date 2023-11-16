import unittest
import pandas as pd
import itertools
import pytest
from coremltools._deps import _HAS_SKLEARN, _HAS_XGBOOST
from coremltools.models.utils import evaluate_regressor, _macos_version, _is_macos
if _HAS_XGBOOST:
    import xgboost
    from coremltools.converters import xgboost as xgb_converter
if _HAS_SKLEARN:
    from sklearn.datasets import load_boston
    from sklearn.ensemble import GradientBoostingRegressor
    from coremltools.converters import sklearn as skl_converter
    from sklearn.tree import DecisionTreeRegressor

@unittest.skipIf(not _HAS_SKLEARN, 'Missing sklearn. Skipping tests.')
class GradientBoostingRegressorBostonHousingScikitNumericTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        if False:
            i = 10
            return i + 15
        scikit_data = load_boston()
        self.scikit_data = scikit_data
        self.X = scikit_data['data']
        self.target = scikit_data['target']
        self.feature_names = scikit_data.feature_names
        self.output_name = 'target'

    def _check_metrics(self, metrics, params={}):
        if False:
            i = 10
            return i + 15
        self.assertAlmostEquals(metrics['rmse'], 0, delta=1e-05, msg='Failed case %s. Results %s' % (params, metrics))
        self.assertAlmostEquals(metrics['max_error'], 0, delta=1e-05, msg='Failed case %s. Results %s' % (params, metrics))

    def _train_convert_evaluate_assert(self, **scikit_params):
        if False:
            print('Hello World!')
        scikit_model = GradientBoostingRegressor(random_state=1, **scikit_params)
        scikit_model.fit(self.X, self.target)
        spec = skl_converter.convert(scikit_model, self.feature_names, self.output_name)
        if _is_macos() and _macos_version() >= (10, 13):
            df = pd.DataFrame(self.X, columns=self.feature_names)
            df['prediction'] = scikit_model.predict(self.X)
            metrics = evaluate_regressor(spec, df, 'target', verbose=False)
            self._check_metrics(metrics, scikit_params)

    def test_boston_housing_simple_regression(self):
        if False:
            while True:
                i = 10
        self._train_convert_evaluate_assert()

    @pytest.mark.slow
    def test_boston_housing_parameter_stress_test(self):
        if False:
            i = 10
            return i + 15
        options = dict(max_depth=[1, 10, None], min_samples_split=[2, 0.5], min_samples_leaf=[1, 5], min_weight_fraction_leaf=[0.0, 0.5], max_features=[None, 1], max_leaf_nodes=[None, 20])
        product = itertools.product(*options.values())
        args = [dict(zip(options.keys(), p)) for p in product]
        print('Testing a total of %s cases. This could take a while' % len(args))
        for (it, arg) in enumerate(args):
            self._train_convert_evaluate_assert(**arg)

@unittest.skipIf(not _HAS_XGBOOST, 'Missing xgboost. Skipping')
@unittest.skipIf(not _HAS_SKLEARN, 'Missing scikit-learn. Skipping tests.')
class XgboostBoosterBostonHousingNumericTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        if False:
            return 10
        if not _HAS_XGBOOST:
            return
        if not _HAS_SKLEARN:
            return
        scikit_data = load_boston()
        self.X = scikit_data.data.astype('f').astype('d')
        self.dtrain = xgboost.DMatrix(scikit_data.data, label=scikit_data.target, feature_names=scikit_data.feature_names)
        self.feature_names = scikit_data.feature_names
        self.output_name = 'target'

    def _check_metrics(self, metrics, params={}):
        if False:
            i = 10
            return i + 15
        '\n        Check the metrics\n        '
        self.assertAlmostEquals(metrics['rmse'], 0, delta=0.0001, msg='Failed case %s. Results %s' % (params, metrics))
        self.assertAlmostEquals(metrics['max_error'], 0, delta=0.0001, msg='Failed case %s. Results %s' % (params, metrics))

    def _train_convert_evaluate_assert(self, bt_params={}, **params):
        if False:
            return 10
        '\n        Set up the unit test by loading the dataset and training a model.\n        '
        xgb_model = xgboost.train(bt_params, self.dtrain, **params)
        spec = xgb_converter.convert(xgb_model, self.feature_names, self.output_name, force_32bit_float=False)
        if _is_macos() and _macos_version() >= (10, 13):
            df = pd.DataFrame(self.X, columns=self.feature_names)
            df['prediction'] = xgb_model.predict(self.dtrain)
            metrics = evaluate_regressor(spec, df, target='target', verbose=False)
            self._check_metrics(metrics, bt_params)

    def test_boston_housing_simple_decision_tree_regression(self):
        if False:
            i = 10
            return i + 15
        self._train_convert_evaluate_assert(num_boost_round=1)

    def test_boston_housing_simple_boosted_tree_regression(self):
        if False:
            print('Hello World!')
        self._train_convert_evaluate_assert(num_boost_round=10)

    @pytest.mark.skip('fails: <rdar://63141759>')
    def test_boston_housing_simple_random_forest_regression(self):
        if False:
            return 10
        self._train_convert_evaluate_assert({'subsample': 0.5})

    def test_boston_housing_float_double_corner_case(self):
        if False:
            for i in range(10):
                print('nop')
        self._train_convert_evaluate_assert({'colsample_bytree': 1, 'colsample_bylevel': 1, 'scale_pos_weight': 1, 'learning_rate': 0.5, 'max_delta_step': 0, 'min_child_weight': 1, 'n_estimators': 1, 'subsample': 0.5, 'objective': 'reg:linear', 'max_depth': 5}, num_boost_round=2)

    @pytest.mark.slow
    def test_boston_housing_parameter_stress_test(self):
        if False:
            print('Hello World!')
        options = dict(max_depth=[1, 5], learning_rate=[0.1, 0.5], n_estimators=[1, 10], min_child_weight=[1, 2], max_delta_step=[0, 0.1], colsample_bytree=[1, 0.5], colsample_bylevel=[1, 0.5], scale_pos_weight=[1], objective=['reg:linear'])
        product = itertools.product(*options.values())
        args = [dict(zip(options.keys(), p)) for p in product]
        print('Testing a total of %s cases. This could take a while' % len(args))
        for (it, arg) in enumerate(args):
            self._train_convert_evaluate_assert(arg)

@unittest.skipIf(not _HAS_XGBOOST, 'Missing xgboost. Skipping')
@unittest.skipIf(not _HAS_SKLEARN, 'Missing sklearn. Skipping tests.')
class XGboostRegressorBostonHousingNumericTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        if False:
            i = 10
            return i + 15
        '\n        Set up the unit test by loading the dataset and training a model.\n        '
        scikit_data = load_boston()
        self.X = scikit_data.data
        self.scikit_data = self.X
        self.target = scikit_data.target
        self.feature_names = scikit_data.feature_names
        self.output_name = 'target'

    def _check_metrics(self, metrics, params={}, allowed_error={}):
        if False:
            return 10
        self.assertAlmostEquals(metrics['rmse'], allowed_error['rmse'] if 'rmse' in allowed_error else 0, delta=0.01, msg='Failed case %s. Results %s' % (params, metrics))
        self.assertAlmostEquals(metrics['max_error'], allowed_error['max_error'] if 'max_error' in allowed_error else 0, delta=0.01, msg='Failed case %s. Results %s' % (params, metrics))

    def _train_convert_evaluate_assert(self, bt_params={}, allowed_error={}, **params):
        if False:
            while True:
                i = 10
        '\n        Set up the unit test by loading the dataset and training a model.\n        '
        xgb_model = xgboost.XGBRegressor(**params)
        xgb_model.fit(self.X, self.target)
        spec = xgb_converter.convert(xgb_model, self.feature_names, self.output_name, force_32bit_float=False)
        if _is_macos() and _macos_version() >= (10, 13):
            df = pd.DataFrame(self.X, columns=self.feature_names)
            df['prediction'] = xgb_model.predict(self.X)
            metrics = evaluate_regressor(spec, df, target='target', verbose=False)
            self._check_metrics(metrics, bt_params, allowed_error)

    def test_boston_housing_simple_boosted_tree_regression(self):
        if False:
            return 10
        self._train_convert_evaluate_assert()

    @pytest.mark.skip('fails: <rdar://63141759>')
    def test_boston_housing_simple_random_forest_regression(self):
        if False:
            print('Hello World!')
        self._train_convert_evaluate_assert(allowed_error={'rmse': 0.0162, 'max_error': 0.2886}, subsample=0.5)

    def test_boston_housing_simple_decision_tree_regression(self):
        if False:
            return 10
        self._train_convert_evaluate_assert(n_estimators=1)

    def test_boston_housing_float_double_corner_case(self):
        if False:
            i = 10
            return i + 15
        self._train_convert_evaluate_assert({'colsample_bytree': 1, 'colsample_bylevel': 1, 'scale_pos_weight': 1, 'learning_rate': 0.1, 'max_delta_step': 0, 'min_child_weight': 1, 'n_estimators': 10, 'subsample': 0.3, 'objective': 'reg:linear', 'max_depth': 1})

    @pytest.mark.slow
    def test_boston_housing_parameter_stress_test(self):
        if False:
            return 10
        options = dict(max_depth=[1, 5], learning_rate=[0.1, 0.5], n_estimators=[1, 10], objective=['reg:linear'], min_child_weight=[1, 2], max_delta_step=[0, 0.1], subsample=[1, 0.5, 0.3], colsample_bytree=[1, 0.5], colsample_bylevel=[1, 0.5], scale_pos_weight=[1])
        product = itertools.product(*options.values())
        args = [dict(zip(options.keys(), p)) for p in product]
        print('Testing a total of %s cases. This could take a while' % len(args))
        for (it, arg) in enumerate(args):
            self._train_convert_evaluate_assert(arg)