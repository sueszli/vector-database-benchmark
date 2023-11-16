import itertools
import pytest
import pandas as pd
import unittest
from coremltools._deps import _HAS_SKLEARN, _HAS_XGBOOST
from coremltools.models.utils import evaluate_classifier, evaluate_classifier_with_probabilities, _macos_version, _is_macos
if _HAS_SKLEARN:
    from sklearn.datasets import load_boston
    from sklearn.ensemble import GradientBoostingClassifier
    from coremltools.converters import sklearn as skl_converter
if _HAS_XGBOOST:
    import xgboost
    from coremltools.converters import xgboost as xgb_converter

@unittest.skipIf(not _HAS_SKLEARN, 'Missing sklearn. Skipping tests.')
class BoostedTreeClassificationBostonHousingScikitNumericTest(unittest.TestCase):
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
        self.target = 1 * (scikit_data['target'] > scikit_data['target'].mean())
        self.feature_names = scikit_data.feature_names
        self.output_name = 'target'

    def _check_metrics(self, metrics, params={}):
        if False:
            for i in range(10):
                print('nop')
        self.assertEquals(metrics['num_errors'], 0, msg='Failed case %s. Results %s' % (params, metrics))

    def _train_convert_evaluate_assert(self, **scikit_params):
        if False:
            for i in range(10):
                print('nop')
        '\n        Train a scikit-learn model, convert it and then evaluate it with CoreML\n        '
        scikit_model = GradientBoostingClassifier(random_state=1, **scikit_params)
        scikit_model.fit(self.X, self.target)
        spec = skl_converter.convert(scikit_model, self.feature_names, self.output_name)
        if _is_macos() and _macos_version() >= (10, 13):
            df = pd.DataFrame(self.X, columns=self.feature_names)
            df['prediction'] = scikit_model.predict(self.X)
            metrics = evaluate_classifier(spec, df)
            self._check_metrics(metrics)

@unittest.skipIf(not _HAS_SKLEARN, 'Missing sklearn. Skipping tests.')
class BoostedTreeBinaryClassificationBostonHousingScikitNumericTest(BoostedTreeClassificationBostonHousingScikitNumericTest):

    def test_simple_binary_classifier(self):
        if False:
            return 10
        self._train_convert_evaluate_assert()

    @pytest.mark.slow
    def test_binary_classifier_stress_test(self):
        if False:
            for i in range(10):
                print('nop')
        options = dict(max_depth=[1, 10, None], min_samples_split=[2, 0.5], min_samples_leaf=[1, 5], min_weight_fraction_leaf=[0.0, 0.5], max_features=[None, 1], max_leaf_nodes=[None, 20])
        product = itertools.product(*options.values())
        args = [dict(zip(options.keys(), p)) for p in product]
        print('Testing a total of %s cases. This could take a while' % len(args))
        for (it, arg) in enumerate(args):
            self._train_convert_evaluate_assert(**arg)

@unittest.skipIf(not _HAS_SKLEARN, 'Missing sklearn. Skipping tests.')
class BoostedTreeMultiClassClassificationBostonHousingScikitNumericTest(BoostedTreeClassificationBostonHousingScikitNumericTest):

    @classmethod
    def setUpClass(self):
        if False:
            while True:
                i = 10
        from sklearn.datasets import load_boston
        import numpy as np
        scikit_data = load_boston()
        num_classes = 3
        self.X = scikit_data.data.astype('f').astype('d')
        t = scikit_data.target
        target = np.digitize(t, np.histogram(t, bins=num_classes - 1)[1]) - 1
        self.scikit_data = scikit_data
        self.target = target
        self.feature_names = scikit_data.feature_names
        self.output_name = 'target'

    def test_simple_multiclass(self):
        if False:
            for i in range(10):
                print('nop')
        self._train_convert_evaluate_assert()

    @pytest.mark.slow
    def test_multiclass_stress_test(self):
        if False:
            return 10
        options = dict(max_depth=[1, 10, None], min_samples_split=[2, 0.5], min_samples_leaf=[1, 5], min_weight_fraction_leaf=[0.0, 0.5], max_features=[None, 1], max_leaf_nodes=[None, 20])
        product = itertools.product(*options.values())
        args = [dict(zip(options.keys(), p)) for p in product]
        print('Testing a total of %s cases. This could take a while' % len(args))
        for (it, arg) in enumerate(args):
            self._train_convert_evaluate_assert(**arg)

@unittest.skipIf(not _HAS_SKLEARN, 'Missing sklearn. Skipping tests.')
@unittest.skipIf(not _HAS_XGBOOST, 'Skipping, no xgboost')
class BoostedTreeClassificationBostonHousingXGboostNumericTest(unittest.TestCase):
    """
    Unit test class for testing xgboost converter and running both models
    """

    def _check_metrics(self, metrics, params={}):
        if False:
            while True:
                i = 10
        self.assertEquals(metrics['num_errors'], 0, msg='Failed case %s. Results %s' % (params, metrics))

    def _train_convert_evaluate_assert(self, **xgboost_params):
        if False:
            for i in range(10):
                print('nop')
        '\n        Train a scikit-learn model, convert it and then evaluate it with CoreML\n        '
        xgb_model = xgboost.XGBClassifier(**xgboost_params)
        xgb_model.fit(self.X, self.target)
        spec = xgb_converter.convert(xgb_model, self.feature_names, self.output_name, mode='classifier')
        if _is_macos() and _macos_version() >= (10, 13):
            df = pd.DataFrame(self.X, columns=self.feature_names)
            probabilities = xgb_model.predict_proba(self.X)
            df['classProbability'] = [dict(zip(xgb_model.classes_, cur_vals)) for cur_vals in probabilities]
            metrics = evaluate_classifier_with_probabilities(spec, df, probabilities='classProbability', verbose=False)
            self.assertEquals(metrics['num_key_mismatch'], 0)
            self.assertLess(metrics['max_probability_error'], 0.001)

    def _classifier_stress_test(self):
        if False:
            return 10
        options = dict(max_depth=[1, 10], min_child_weight=[2, 0.5], max_delta_step=[1, 5])
        product = itertools.product(*options.values())
        args = [dict(zip(options.keys(), p)) for p in product]
        print('Testing a total of %s cases. This could take a while' % len(args))
        for (it, arg) in enumerate(args):
            self._train_convert_evaluate_assert(**arg)

@unittest.skipIf(not _HAS_SKLEARN, 'Missing sklearn. Skipping tests.')
@unittest.skipIf(not _HAS_XGBOOST, 'Skipping, no xgboost')
class BoostedTreeBinaryClassificationBostonHousingXGboostNumericTest(BoostedTreeClassificationBostonHousingXGboostNumericTest):

    @classmethod
    def setUpClass(self):
        if False:
            return 10
        '\n        Set up the unit test by loading the dataset and training a model.\n        '
        from sklearn.datasets import load_boston
        scikit_data = load_boston()
        self.scikit_data = scikit_data
        self.X = scikit_data.data.astype('f').astype('d')
        self.target = 1 * (scikit_data['target'] > scikit_data['target'].mean())
        self.feature_names = scikit_data.feature_names
        self.output_name = 'target'

    def test_simple_binary_classifier(self):
        if False:
            return 10
        self._train_convert_evaluate_assert()

    @pytest.mark.slow
    def test_binary_classifier_stress_test(self):
        if False:
            while True:
                i = 10
        self._classifier_stress_test()

@unittest.skipIf(not _HAS_SKLEARN, 'Missing sklearn. Skipping tests.')
@unittest.skipIf(not _HAS_XGBOOST, 'Skipping, no xgboost')
class BoostedTreeMultiClassClassificationBostonHousingXGboostNumericTest(BoostedTreeClassificationBostonHousingXGboostNumericTest):

    @classmethod
    def setUpClass(self):
        if False:
            return 10
        from sklearn.datasets import load_boston
        import numpy as np
        scikit_data = load_boston()
        num_classes = 3
        self.X = scikit_data.data.astype('f').astype('d')
        t = scikit_data.target
        target = np.digitize(t, np.histogram(t, bins=num_classes - 1)[1]) - 1
        self.scikit_data = scikit_data
        self.target = target
        self.feature_names = scikit_data.feature_names
        self.output_name = 'target'

    def test_simple_multiclass(self):
        if False:
            print('Hello World!')
        self._train_convert_evaluate_assert()

    @pytest.mark.slow
    def test_multiclass_stress_test(self):
        if False:
            i = 10
            return i + 15
        self._classifier_stress_test()