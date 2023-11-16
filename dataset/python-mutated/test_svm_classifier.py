from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import unittest
import turicreate as tc
import sys
import operator as op
import uuid
import numpy as np
import array
from turicreate.toolkits._main import ToolkitError
from turicreate.toolkits.classifier.svm_classifier import _DEFAULT_SOLVER_OPTIONS
import shutil
import os as _os
try:
    from sklearn.metrics import *
    from sklearn import svm
except ImportError as e:
    if not tc._deps.is_minimal_pkg():
        raise e

class SVMClassifierTest(unittest.TestCase):
    """
    Unit test class for a LogisticRegressionModel that has already been created.
    """

    @classmethod
    def setUpClass(self):
        if False:
            i = 10
            return i + 15
        '\n        Set up (run only once).\n        '
        np.random.seed(10)
        (n, d) = (100, 10)
        self.sf = tc.SFrame()
        for i in range(d):
            self.sf.add_column(tc.SArray(np.random.randn(n)), inplace=True)
        target = np.random.randint(2, size=n)
        target[0] = 0
        target[1] = 1
        self.sf['target'] = target
        self.def_kwargs = _DEFAULT_SOLVER_OPTIONS
        self.def_opts = dict(list(self.def_kwargs.items()) + list({'solver': 'auto', 'feature_rescaling': 1, 'class_weights': None, 'penalty': 1.0}.items()))
        self.opts = self.def_opts.copy()
        self.opts['max_iterations'] = 500
        self.opts['solver'] = 'lbfgs'
        self.features = ['X{}'.format(i) for i in range(1, d + 1)]
        features = self.features
        self.unpacked_features = ['X{}'.format(i) for i in range(1, d + 1)]
        self.target = 'target'
        self.model = tc.svm_classifier.create(self.sf, target='target', features=None, feature_rescaling=True, validation_set=None, max_iterations=self.opts['max_iterations'])
        target_name = self.target
        feature_names = self.features
        X_train = list(self.sf.apply(lambda row: [row[k] for k in features]))
        y_train = list(self.sf[self.target])
        sm_model = svm.LinearSVC(C=1.0, loss='hinge')
        sm_model.fit(X_train, y_train)
        self.coef = list(sm_model.intercept_) + list(sm_model.coef_[0])
        predictions = list(sm_model.predict(X_train))
        classes = predictions
        margins = [np.concatenate(([1], x)).dot(np.array(self.coef)) for x in X_train]
        self.yhat_class = tc.SArray(predictions)
        self.yhat_margins = tc.SArray(margins)
        self.sm_metrics = {'accuracy': accuracy_score(target, list(self.yhat_class)), 'confusion_matrix': tc.toolkits.evaluation.confusion_matrix(tc.SArray(target), tc.SArray(self.yhat_class)), 'f1_score': f1_score(target, list(self.yhat_class)), 'precision': precision_score(target, list(self.yhat_class)), 'recall': recall_score(target, list(self.yhat_class))}
        self.get_ans = {'coefficients': lambda x: isinstance(x, tc.SFrame), 'convergence_threshold': lambda x: x == self.opts['convergence_threshold'], 'unpacked_features': lambda x: x == self.unpacked_features, 'feature_rescaling': lambda x: x == True, 'features': lambda x: x == self.features, 'lbfgs_memory_level': lambda x: x == 11, 'max_iterations': lambda x: x == self.opts['max_iterations'], 'num_classes': lambda x: x == 2, 'num_coefficients': lambda x: x == 11, 'num_examples': lambda x: x == 100, 'classes': lambda x: set(x) == set([0, 1]), 'class_weights': lambda x: x == {0: 1, 1: 1}, 'num_examples_per_class': lambda x: {0: (tc.SArray(target) == 0).sum(), 1: (tc.SArray(target) == 1).sum()}, 'num_features': lambda x: x == 10, 'num_unpacked_features': lambda x: x == 10, 'penalty': lambda x: x == self.opts['penalty'], 'progress': lambda x: isinstance(x, tc.SFrame), 'solver': lambda x: x == self.opts['solver'], 'target': lambda x: x == self.target, 'training_accuracy': lambda x: x >= 0 and x <= 1, 'training_iterations': lambda x: x > 0, 'training_loss': lambda x: x > 0, 'training_solver_status': lambda x: x == 'SUCCESS: Optimal solution found.', 'training_time': lambda x: x >= 0, 'training_confusion_matrix': lambda x: len(x) > 0, 'training_f1_score': lambda x: x > 0, 'training_precision': lambda x: x > 0, 'training_recall': lambda x: x > 0, 'training_report_by_class': lambda x: len(x) > 0, 'validation_data': lambda x: isinstance(x, tc.SFrame) and len(x) == 0, 'disable_posttrain_evaluation': lambda x: x == False}
        self.fields_ans = self.get_ans.keys()

    def test__list_fields(self):
        if False:
            while True:
                i = 10
        '\n        Check the list fields function.\n        '
        model = self.model
        fields = model._list_fields()
        self.assertEqual(set(fields), set(self.fields_ans))

    def test_get(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Check the get function. Compare with the answer supplied as a lambda\n        function for each field.\n        '
        model = self.model
        for field in self.fields_ans:
            ans = model._get(field)
            self.assertTrue(self.get_ans[field](ans), 'Get failed in field {}. Output was {}.'.format(field, ans))

    def test_coefficients(self):
        if False:
            while True:
                i = 10
        '\n        Check that the coefficient values are very close to the correct values.\n        '
        model = self.model
        coefs = model.coefficients
        coef_list = list(coefs['value'])

    def test_summary(self):
        if False:
            while True:
                i = 10
        '\n        Check the summary function.\n        '
        model = self.model
        model.summary()

    def test_repr(self):
        if False:
            print('Hello World!')
        '\n        Check the repr function.\n        '
        model = self.model
        ans = str(model)
        self.assertTrue(type(ans) == str)

    def test_predict(self):
        if False:
            i = 10
            return i + 15
        '\n        Check the prediction function.\n        '
        model = self.model
        ans = model.predict(self.sf)
        ans = model.predict(self.sf, output_type='class')
        self.assertEqual(ans.dtype, int)
        ans = model.predict(self.sf, output_type='margin')

    def test_classify(self):
        if False:
            print('Hello World!')
        '\n        Check the classify function.\n        '
        model = self.model
        ans = model.classify(self.sf)
        self.assertEqual(len(ans), len(self.sf))

    def test_evaluate(self):
        if False:
            print('Hello World!')
        '\n        Make sure that evaluate works.\n        '
        model = self.model

        def check_cf_matrix(ans):
            if False:
                for i in range(10):
                    print('nop')
            self.assertTrue(ans is not None)
            self.assertTrue('confusion_matrix' in ans)
            cf = ans['confusion_matrix'].sort(['target_label', 'predicted_label'])
            sm = self.sf_margin['confusion_matrix'].sort(['target_label', 'predicted_label'])
            self.assertTrue(np.allclose(cf['count'], sm['count']))

        def check_metric(ans, metric):
            if False:
                return 10
            if metric == 'confusion_matrix':
                check_cf_matrix(ans)
            else:
                self.assertTrue(ans is not None)
                self.assertTrue(metric in ans)
                self.assertAlmostEqual(ans[metric], self.sm_metrics[metric], places=4, msg='%s = (%s,%s)' % (metric, ans[metric], self.sm_metrics[metric]))

    def test_save_and_load(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Make sure saving and loading retains everything.\n        '
        filename = 'save_file{}'.format(uuid.uuid4())
        self.model.save(filename)
        self.model = tc.load_model(filename)
        self.test_get()
        print('Get passed')
        self.test_coefficients()
        print('Coefficients passed')
        self.test_summary()
        print('Summary passed')
        self.test_repr()
        print('Repr passed')
        self.test_predict()
        print('Predict passed')
        self.test_classify()
        print('Classify passed')
        self.test_evaluate()
        print('Evaluate passed')
        self.test__list_fields()
        print('List fields passed')
        shutil.rmtree(filename)

class SVMCreateTest(unittest.TestCase):
    """
    Unit test class for testing a svm model that is not trained.
    """

    @classmethod
    def setUpClass(self):
        if False:
            i = 10
            return i + 15
        "\n        Setup required for all tests that don't require an trained model.\n        "
        np.random.seed(8)
        (n, d) = (100, 10)
        self.sf = tc.SFrame()
        for i in range(d):
            self.sf.add_column(tc.SArray(np.random.randn(n)), inplace=True)
        target = np.random.randint(2, size=n)
        target[0] = 0
        target[1] = 1
        self.sf['target'] = target
        self.def_kwargs = _DEFAULT_SOLVER_OPTIONS
        self.solver = 'auto'
        self.features = ', '.join(['X{}'.format(i) for i in range(1, d + 1)])
        self.target = 'target'
        self.sf['target'] = target
        self.features = ['X{}'.format(i) for i in range(1, d + 1)]
        target_name = self.target
        feature_names = self.features
        X_train = list(self.sf.apply(lambda row: [row[k] for k in feature_names]))
        y_train = list(self.sf[self.target])
        sm_model = svm.LinearSVC(C=1.0, loss='hinge')
        sm_model.fit(X_train, y_train)
        self.coef = list(sm_model.intercept_) + list(sm_model.coef_[0])

    def _test_create(self, sf, target, features, solver, kwargs):
        if False:
            print('Hello World!')
        '\n        Test svm create.\n        '
        model = tc.svm_classifier.create(sf, target, features, solver=solver, verbose=False, validation_set=None, feature_rescaling=False, **kwargs)
        test_case = 'solver = {}, kwargs = {}'.format(solver, kwargs)
        self.assertTrue(model is not None, 'Model is None')
        coefs = list(model.coefficients['value'])
        print(coefs, self.coef)
        self.assertTrue(np.allclose(coefs, self.coef, rtol=0.2, atol=0.2))

    def test_class_weights(self):
        if False:
            while True:
                i = 10
        '\n        Test svm create.\n        '
        model = tc.svm_classifier.create(self.sf, self.target, self.features, class_weights='auto', validation_set=None)
        model = tc.svm_classifier.create(self.sf, self.target, self.features, class_weights={0: 1, 1: 2})
        try:
            model = tc.svm_classifier.create(self.sf, self.target, self.features, class_weights=1.0, validation_set=None)
        except ToolkitError:
            pass
        try:
            model = tc.svm_classifier.create(self.sf, self.target, self.features, class_weights={2: 10}, validation_set=None)
        except ToolkitError:
            pass
        try:
            model = tc.svm_classifier.create(self.sf, self.target, self.features, class_weights=[1, 1], validation_set=None)
        except ToolkitError:
            pass

    def test_create_default_features(self):
        if False:
            print('Hello World!')
        '\n        Test svm create.\n        '
        kwargs = self.def_kwargs.copy()
        kwargs['max_iterations'] = 100
        for solver in ['lbfgs', 'auto']:
            args = (self.sf, self.target, None, solver, kwargs)
            self._test_create(*args)

    def test_create(self):
        if False:
            print('Hello World!')
        '\n        Test svm create.\n        '
        kwargs = self.def_kwargs.copy()
        kwargs['max_iterations'] = 100
        for solver in ['lbfgs', 'auto']:
            args = (self.sf, self.target, self.features, solver, kwargs)
            self._test_create(*args)

class ListCategoricalSVMTest(unittest.TestCase):
    """
    Unit test class for testing svm with a categorical feature.
    """

    @classmethod
    def setUpClass(self):
        if False:
            return 10
        '\n        Set up (run once).\n        '
        np.random.seed(15)
        (n, d) = (100, 3)
        self.sf = tc.SFrame()
        for i in range(d):
            self.sf.add_column(tc.SArray(np.random.rand(n)), inplace=True)
        species = np.array(['cat', 'dog', 'foosa'])
        idx = np.random.randint(3, size=n)
        idx[0] = 0
        idx[1] = 1
        idx[2] = 2
        self.sf['species'] = list(species[idx])
        y = np.random.randint(2, size=n)
        y[0] = 0
        y[1] = 1
        self.sf['target'] = y
        self.target = 'target'
        self.features = ['species', 'X1', 'X2', 'X3']
        self.def_kwargs = _DEFAULT_SOLVER_OPTIONS
        target_name = self.target
        order = ['cat', 'dog', 'foosa']
        self.sf['species_0'] = self.sf['species'] == order[1]
        self.sf['species_1'] = self.sf['species'] == order[2]
        feature_names = ['species_0', 'species_1', 'X1', 'X2', 'X3']
        X_train = list(self.sf.apply(lambda row: [row[k] for k in feature_names]))
        y_train = list(self.sf[self.target])
        sm_model = svm.LinearSVC(C=1.0, loss='hinge')
        sm_model.fit(X_train, y_train)
        self.coef = list(sm_model.intercept_) + list(sm_model.coef_[0])
        self.sf['species'] = self.sf['species'].apply(lambda x: [x])

    def _test_coefficients(self, model):
        if False:
            while True:
                i = 10
        '\n        Check that the coefficient values are very close to the correct values.\n        '
        coefs = model.coefficients
        coef_list = list(coefs['value'])

    def _test_create(self, sf, target, features, solver, kwargs):
        if False:
            while True:
                i = 10
        '\n        Test svm create function for a particular set of inputs.\n        '
        test_label = 'solver: {}\tkwargs: {}'.format(solver, kwargs)
        model = tc.svm_classifier.create(sf, target, features, solver=solver, feature_rescaling=False, **kwargs)
        self.assertTrue(model is not None, 'Model is None')
        self._test_coefficients(model)

    def test_create(self):
        if False:
            i = 10
            return i + 15
        '\n        Driver for testing create function under various inputs.\n        '
        kwargs = self.def_kwargs.copy()
        kwargs['max_iterations'] = 100
        for solver in ['auto', 'lbfgs']:
            self._test_create(self.sf, self.target, self.features, solver, kwargs)

class CategoricalSVMTest(unittest.TestCase):
    """
    Unit test class for testing svm with a categorical feature.
    """

    @classmethod
    def setUpClass(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set up (run once).\n        '
        np.random.seed(15)
        (n, d) = (100, 3)
        self.sf = tc.SFrame()
        for i in range(d):
            self.sf.add_column(tc.SArray(np.random.rand(n)), inplace=True)
        species = np.array(['cat', 'dog', 'foosa'])
        idx = np.random.randint(3, size=n)
        idx[0] = 0
        idx[1] = 1
        idx[2] = 2
        self.sf['species'] = list(species[idx])
        y = np.random.randint(2, size=n)
        y[0] = 0
        y[1] = 1
        self.sf['target'] = y
        self.target = 'target'
        self.features = ['species', 'X1', 'X2', 'X3']
        self.def_kwargs = _DEFAULT_SOLVER_OPTIONS
        target_name = self.target
        order = ['cat', 'dog', 'foosa']
        self.sf['species_0'] = self.sf['species'] == order[1]
        self.sf['species_1'] = self.sf['species'] == order[2]
        feature_names = ['species_0', 'species_1', 'X1', 'X2', 'X3']
        X_train = list(self.sf.apply(lambda row: [row[k] for k in feature_names]))
        y_train = list(self.sf[self.target])
        sm_model = svm.LinearSVC(C=1.0, loss='hinge')
        sm_model.fit(X_train, y_train)
        self.coef = list(sm_model.intercept_) + list(sm_model.coef_[0])

    def _test_coefficients(self, model):
        if False:
            return 10
        '\n        Check that the coefficient values are very close to the correct values.\n        '
        coefs = model.coefficients
        coef_list = list(coefs['value'])

    def _test_create(self, sf, target, features, solver, kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Test svm create function for a particular set of inputs.\n        '
        test_label = 'solver: {}\tkwargs: {}'.format(solver, kwargs)
        model = tc.svm_classifier.create(sf, target, features, solver=solver, feature_rescaling=False, **kwargs)
        self.assertTrue(model is not None, 'Model is None')
        self._test_coefficients(model)

    def test_create(self):
        if False:
            while True:
                i = 10
        '\n        Driver for testing create function under various inputs.\n        '
        kwargs = self.def_kwargs.copy()
        kwargs['max_iterations'] = 100
        for solver in ['auto', 'lbfgs']:
            self._test_create(self.sf, self.target, self.features, solver, kwargs)

    def test_predict_new_categories(self):
        if False:
            for i in range(10):
                print('nop')
        model = tc.svm_classifier.create(self.sf, self.target, self.features, feature_rescaling=False, validation_set=None)
        pred = model.predict(self.sf)
        self.sf['species'] = self.sf['species'].apply(lambda x: 'rat' if x == 'foosa' else x)
        pred = model.evaluate(self.sf)
        self.sf['species'] = self.sf['species'].apply(lambda x: 'foosa' if x == 'rat' else x)

    def test_evaluate_new_categories(self):
        if False:
            for i in range(10):
                print('nop')
        model = tc.svm_classifier.create(self.sf, self.target, self.features, feature_rescaling=False, validation_set=None)
        pred = model.predict(self.sf)
        self.sf['species'] = self.sf['species'].apply(lambda x: 'rat' if x == 'foosa' else x)
        pred = model.evaluate(self.sf)
        self.sf['species'] = self.sf['species'].apply(lambda x: 'foosa' if x == 'rat' else x)
    '\n       Test detection of columns that are almost the same.\n    '

    def test_zero_variance_detection(self):
        if False:
            while True:
                i = 10
        sf = self.sf
        try:
            sf['error-column'] = 1
            model = tc.svm_classifier.create(sf, self.target)
        except ToolkitError:
            pass
        try:
            sf['error-column'] = '1'
            model = tc.svm_classifier.create(sf, self.target)
        except ToolkitError:
            pass
        try:
            sf['error-column'] = [[1] for i in sf]
            model = tc.svm_classifier.create(sf, self.target)
        except ToolkitError:
            pass
        try:
            sf['error-column'] = [{1: 1} for i in sf]
            model = tc.svm_classifier.create(sf, self.target)
        except ToolkitError:
            pass
        del sf['error-column']
    '\n       Test detection of columns have nan\n    '

    def test_nan_detection(self):
        if False:
            for i in range(10):
                print('nop')
        sf = self.sf
        try:
            sf['error-column'] = np.nan
            model = tc.svm_classifier.create(sf, self.target)
        except ToolkitError:
            pass
        try:
            sf['error-column'] = [[np.nan] for i in sf]
            model = tc.svm_classifier.create(sf, self.target)
        except ToolkitError:
            pass
        try:
            sf['error-column'] = [{1: np.nan} for i in sf]
            model = tc.svm_classifier.create(sf, self.target)
        except ToolkitError:
            pass
        del sf['error-column']

class VectorSVMTest(unittest.TestCase):
    """
    Unit test class for testing a svm create function.
  """

    @classmethod
    def setUpClass(self):
        if False:
            while True:
                i = 10
        '\n        Set up (Run only once)\n    '
        np.random.seed(15)
        (n, d) = (100, 3)
        self.sf = tc.SFrame()
        for i in range(d):
            self.sf.add_column(tc.SArray(np.random.rand(n)), inplace=True)
        self.sf['target'] = np.random.randint(2, size=n)
        self.target = 'target'
        self.sf['vec'] = self.sf.apply(lambda row: [row['X{}'.format(i + 1)] for i in range(d)])
        self.sf['vec'] = self.sf['vec'].apply(lambda x: x, array.array)
        self.features = ['vec']
        self.unpacked_features = ['vec[%s]' % i for i in range(d)]
        self.def_kwargs = _DEFAULT_SOLVER_OPTIONS
        target_name = self.target
        feature_names = self.features
        X_train = list(self.sf['vec'])
        y_train = list(self.sf[self.target])
        sm_model = svm.LinearSVC(C=1.0, loss='hinge')
        sm_model.fit(X_train, y_train)
        self.coef = list(sm_model.intercept_) + list(sm_model.coef_[0])

    def _test_coefficients(self, model):
        if False:
            print('Hello World!')
        '\n      Check that the coefficient values are very close to the correct values.\n      '
        coefs = model.coefficients
        coef_list = list(coefs['value'])

    def _test_create(self, sf, target, features, solver, kwargs):
        if False:
            while True:
                i = 10
        model = tc.svm_classifier.create(sf, target, features, solver=solver, feature_rescaling=False, **kwargs)
        test_case = 'solver = {solver}, kwargs = {kwargs}'.format(solver=solver, kwargs=kwargs)
        self.assertTrue(model is not None, 'Model is None')
        self._test_coefficients(model)

    def test_create(self):
        if False:
            return 10
        kwargs = self.def_kwargs.copy()
        kwargs['max_iterations'] = 1000
        for solver in ['auto', 'lbfgs']:
            args = (self.sf, self.target, self.features, solver, self.def_kwargs)
            self._test_create(*args)

    def test_features(self):
        if False:
            print('Hello World!')
        model = tc.svm_classifier.create(self.sf, self.target, self.features, feature_rescaling=False)
        self.assertEqual(model.num_features, len(self.features))
        self.assertEqual(model.features, self.features)
        self.assertEqual(model.num_unpacked_features, len(self.unpacked_features))
        self.assertEqual(model.unpacked_features, self.unpacked_features)

class DictSVMTest(unittest.TestCase):
    """
    Unit test class for testing a svm create function.
  """

    @classmethod
    def setUpClass(self):
        if False:
            while True:
                i = 10
        '\n        Set up (Run only once)\n    '
        np.random.seed(15)
        (n, d) = (100, 3)
        self.sf = tc.SFrame()
        for i in range(d):
            self.sf.add_column(tc.SArray(np.random.rand(n)), inplace=True)
        self.sf['target'] = np.random.randint(2, size=n)
        self.target = 'target'
        self.sf['dict'] = self.sf.apply(lambda row: {i: row['X{}'.format(i + 1)] for i in range(d)})
        self.features = ['dict']
        self.unpacked_features = ['dict[%s]' % i for i in range(d)]
        self.def_kwargs = _DEFAULT_SOLVER_OPTIONS
        target_name = self.target
        feature_names = self.features
        X_train = list(self.sf['dict'].apply(lambda x: [x[k] for k in sorted(x.keys())]))
        y_train = list(self.sf[self.target])
        sm_model = svm.LinearSVC(C=1.0, loss='hinge')
        sm_model.fit(X_train, y_train)
        self.coef = list(sm_model.intercept_) + list(sm_model.coef_[0])

    def _test_coefficients(self, model):
        if False:
            print('Hello World!')
        coefs = model.coefficients
        coef_list = list(coefs['value'])

    def _test_create(self, sf, target, features, solver, kwargs):
        if False:
            return 10
        model = tc.svm_classifier.create(sf, target, features, solver=solver, feature_rescaling=False, **kwargs)
        test_case = 'solver = {solver}, kwargs = {kwargs}'.format(solver=solver, kwargs=kwargs)
        self.assertTrue(model is not None, 'Model is None')
        self._test_coefficients(model)

    def test_create(self):
        if False:
            for i in range(10):
                print('nop')
        kwargs = self.def_kwargs.copy()
        kwargs['max_iterations'] = 100
        for solver in ['auto']:
            args = (self.sf, self.target, self.features, solver, self.def_kwargs)
            self._test_create(*args)

    def test_predict_extra_cols(self):
        if False:
            print('Hello World!')
        sf = self.sf[:]
        model = tc.svm_classifier.create(sf, self.target, self.features, feature_rescaling=False)
        pred = model.predict(sf)
        sf['dict'] = sf['dict'].apply(lambda x: dict(list(x.items()) + list({'extra_col': 0}.items())))
        pred2 = model.predict(sf)
        self.assertTrue((pred == pred2).all())

    def test_evaluate_extra_cols(self):
        if False:
            return 10
        sf = self.sf[:]
        model = tc.svm_classifier.create(sf, self.target, self.features, feature_rescaling=False)
        eval1 = model.predict(sf)
        sf['dict'] = sf['dict'].apply(lambda x: dict(list(x.items()) + list({'extra_col': 0}.items())))
        eval2 = model.predict(sf)
        self.assertTrue((eval1 == eval2).all())

    def test_features(self):
        if False:
            return 10
        model = tc.svm_classifier.create(self.sf, self.target, self.features, feature_rescaling=False)
        self.assertEqual(model.num_features, len(self.features))
        self.assertEqual(model.features, self.features)
        self.assertEqual(model.num_unpacked_features, len(self.unpacked_features))
        self.assertEqual(model.unpacked_features, self.unpacked_features)

class SVMStringTargetTest(unittest.TestCase):
    """
    Check that the model works with String target types.
    This is not a correctness of the model training.
    """

    @classmethod
    def setUpClass(self):
        if False:
            while True:
                i = 10
        '\n        Set up (run only once).\n        '
        np.random.seed(8)
        (n, d) = (100, 10)
        self.sf = tc.SFrame()
        for i in range(d):
            self.sf.add_column(tc.SArray(np.random.rand(n)), inplace=True)
        target = np.random.randint(2, size=n)
        self.sf['target'] = target
        self.sf['target'] = self.sf['target'].astype(str)

    def test_create(self):
        if False:
            for i in range(10):
                print('nop')
        model = tc.svm_classifier.create(self.sf, target='target')
        predictions = model.predict(self.sf)
        results = model.classify(self.sf)
        results = model.evaluate(self.sf)

class ValidationSetSVMTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        if False:
            while True:
                i = 10
        np.random.seed(10)
        (n, d) = (100, 10)
        self.sf = tc.SFrame()
        for i in range(d):
            self.sf.add_column(tc.SArray(np.random.randn(n)), inplace=True)
        target = np.random.randint(2, size=n)
        self.sf['target'] = target
        self.def_kwargs = _DEFAULT_SOLVER_OPTIONS
        self.def_opts = dict(list(self.def_kwargs.items()) + list({'solver': 'auto', 'feature_rescaling': True, 'class_weights': None, 'penalty': 1.0}.items()))
        self.solver = 'auto'
        self.opts = self.def_opts.copy()
        self.opts['max_iterations'] = 500
        self.features = ['X{}'.format(i) for i in range(1, d + 1)]
        self.unpacked_features = ['X{}'.format(i) for i in range(1, d + 1)]
        self.target = 'target'

    def test_valid_set(self):
        if False:
            while True:
                i = 10
        model = tc.svm_classifier.create(self.sf, target='target', validation_set='auto')
        self.assertTrue(model is not None)
        self.assertTrue(isinstance(model.progress, tc.SFrame))
        model = tc.svm_classifier.create(self.sf, target='target', validation_set=self.sf)
        self.assertTrue(model is not None)
        self.assertTrue(isinstance(model.progress, tc.SFrame))
        model = tc.svm_classifier.create(self.sf, target='target', validation_set=None)
        self.assertTrue(model is not None)
        self.assertTrue(isinstance(model.progress, tc.SFrame))

class TestStringTarget(unittest.TestCase):

    def test_cat(self):
        if False:
            print('Hello World!')
        import numpy as np
        np.random.seed(8)
        (n, d) = (1000, 100)
        sf = tc.SFrame()
        for i in range(d):
            sf.add_column(tc.SArray(np.random.rand(n)), inplace=True)
            target = np.random.randint(2, size=n)
            sf['target'] = target
        sf['target'] = sf['target'].astype(str)
        sf['target'] = 'cat-' + sf['target']
        model = tc.logistic_classifier.create(sf, 'target')
        evaluation = model.evaluate(sf)
        self.assertEqual(['cat-0', 'cat-1'], sorted(list(evaluation['confusion_matrix']['target_label'].unique())))