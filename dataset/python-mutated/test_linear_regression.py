from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import unittest
import turicreate as tc
import sys
import operator as op
import uuid
import os
import array
import shutil
import numpy as np
from turicreate.toolkits._main import ToolkitError
from turicreate.toolkits.regression.linear_regression import _DEFAULT_SOLVER_OPTIONS
if sys.version_info.major == 3:
    from functools import reduce
try:
    import statsmodels.formula.api as sm
    from sklearn import linear_model
except ImportError as e:
    if not tc._deps.is_minimal_pkg():
        raise e

class LinearRegressionTest(unittest.TestCase):
    """
    Unit test class for testing a Linear Regression model that has already
    been created.
    """

    @classmethod
    def setUpClass(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set up (Run only once)\n        '
        np.random.seed(42)
        (n, d) = (100, 10)
        self.sf = tc.SFrame()
        for i in range(d):
            self.sf.add_column(tc.SArray(np.random.rand(n)), inplace=True)
        target = np.random.rand(n)
        self.sf['target'] = target
        formula = 'target ~ ' + ' + '.join(['X{}'.format(i) for i in range(1, d + 1)])
        df = self.sf.to_dataframe()
        sm_model = sm.ols(formula, data=df).fit()
        self.loss = sm_model.ssr
        self.coef = list(sm_model.params)
        self.stderr = list(sm_model.bse)
        self.yhat = list(sm_model.fittedvalues)
        self.rmse = np.sqrt(sm_model.ssr / float(n))
        self.maxerr = abs(target - np.array(self.yhat)).max()
        self.def_kwargs = _DEFAULT_SOLVER_OPTIONS
        self.solver = 'auto'
        self.unpacked_features = ['X{}'.format(i) for i in range(1, d + 1)]
        self.features = ['X{}'.format(i) for i in range(1, d + 1)]
        self.target = 'target'
        self.def_opts = dict(list(self.def_kwargs.items()) + list({'solver': 'auto', 'feature_rescaling': True, 'l1_penalty': 0, 'l2_penalty': 0.01}.items()))
        self.opts = self.def_opts.copy()
        self.opts['l2_penalty'] = 0.0
        self.opts['solver'] = 'newton'
        self.model = tc.linear_regression.create(self.sf, target=self.target, features=None, l2_penalty=0.0, l1_penalty=0.0, feature_rescaling=True, validation_set=None, solver=self.solver)
        self.evaluate_ans = {'max_error': lambda x: abs(x - self.maxerr) < 0.001, 'evaluate_time': lambda x: x > 0, 'rmse': lambda x: abs(x - self.rmse) < 0.001}
        self.get_ans = {'coefficients': lambda x: isinstance(x, tc.SFrame), 'convergence_threshold': lambda x: x == self.opts['convergence_threshold'], 'features': lambda x: x == self.features, 'unpacked_features': lambda x: x == self.unpacked_features, 'feature_rescaling': lambda x: x == self.opts['feature_rescaling'], 'l1_penalty': lambda x: x == 0.0, 'l2_penalty': lambda x: x == 0.0, 'lbfgs_memory_level': lambda x: x == self.opts['lbfgs_memory_level'], 'max_iterations': lambda x: x == self.opts['max_iterations'], 'num_coefficients': lambda x: x == 11, 'num_examples': lambda x: x == 100, 'num_features': lambda x: x == 10, 'num_unpacked_features': lambda x: x == 10, 'progress': lambda x: isinstance(x, tc.SFrame), 'solver': lambda x: x == self.opts['solver'], 'training_solver_status': lambda x: x == 'SUCCESS: Optimal solution found.', 'step_size': lambda x: x == self.opts['step_size'], 'target': lambda x: x == self.target, 'training_iterations': lambda x: x > 0, 'training_loss': lambda x: abs(x - self.loss) < 1e-05, 'training_rmse': lambda x: abs(x - self.rmse) < 1e-05, 'training_time': lambda x: x >= 0, 'training_max_error': lambda x: x > 0, 'validation_data': lambda x: isinstance(x, tc.SFrame) and len(x) == 0, 'disable_posttrain_evaluation': lambda x: x == False}
        self.fields_ans = self.get_ans.keys()

    def test__list_fields(self):
        if False:
            print('Hello World!')
        '\n        Check the _list_fields function. Compare with the answer.\n        '
        model = self.model
        fields = model._list_fields()
        self.assertEqual(set(fields), set(self.fields_ans))

    def test_get(self):
        if False:
            return 10
        '\n        Check the get function. Compare with the answer supplied as a lambda\n        function for each field.\n        '
        model = self.model
        for field in self.fields_ans:
            ans = model._get(field)
            self.assertTrue(self.get_ans[field](ans), 'Get failed in field {}. Output was {}.'.format(field, ans))

    def test_coefficients(self, test_stderr=True):
        if False:
            print('Hello World!')
        '\n        Check that the coefficient values are very close to the correct values.\n        '
        model = self.model
        coefs = model.coefficients
        coef_list = list(coefs['value'])
        self.assertTrue(np.allclose(coef_list, self.coef, rtol=0.001, atol=0.001))
        if test_stderr:
            stderr_list = list(coefs['stderr'])
            self.assertTrue(np.allclose(stderr_list, self.stderr, rtol=0.001, atol=0.001))
        else:
            self.assertTrue('stderr' in coefs.column_names())
            self.assertEqual(list(coefs['stderr']), [None for v in coef_list])

    def test_summary(self):
        if False:
            print('Hello World!')
        '\n        Check the summary function. Compare with the answer supplied as\n        a lambda function for each field. Uses the same answers as test_get.\n        '
        model = self.model
        model.summary()

    def test_repr(self):
        if False:
            while True:
                i = 10
        '\n        Check the repr function.\n        '
        model = self.model
        ans = str(model)
        self.assertEqual(type(ans), str)

    def test_predict(self):
        if False:
            print('Hello World!')
        '\n        Check the prediction function with precomputed answers.  Check that all\n        predictions are atmost 1e-5 away from the true answers.\n        '
        model = self.model
        ans = model.predict(self.sf)
        reduce(op.and_, map(lambda x, y: abs(x - y) < 1e-05, ans, self.yhat))
        self.sf['extra_col'] = 1
        ans = model.predict(self.sf)
        reduce(op.and_, map(lambda x, y: abs(x - y) < 1e-05, ans, self.yhat))
        del self.sf['extra_col']

    def test_evaluate(self):
        if False:
            i = 10
            return i + 15
        '\n        Check the evaluation function with precomputed answers.\n        '
        model = self.model
        ans = model.evaluate(self.sf)

        def check_ans():
            if False:
                for i in range(10):
                    print('nop')
            for field in ans:
                self.assertTrue(self.evaluate_ans[field](ans[field]), 'Evaluation failed in field {}.  Output was {}'.format(field, ans[field]))
        check_ans()
        rmse = model.evaluate(self.sf, metric='rmse')
        check_ans()
        max_error = model.evaluate(self.sf, metric='max_error')
        check_ans()

    def test_save_and_load(self):
        if False:
            return 10
        '\n        Make sure saving and loading retains things.\n        '
        filename = 'save_file%s' % str(uuid.uuid4())
        self.model.save(filename)
        self.model = tc.load_model(filename)
        try:
            self.test_coefficients()
            print('Coefs passed')
            self.test_summary()
            print('Summary passed')
            self.test_repr()
            print('Repr passed')
            self.test_predict()
            print('Predict passed')
            self.test_evaluate()
            print('Evaluate passed')
            self.test__list_fields()
            print('List field passed')
            self.test_get()
            print('Get passed')
            shutil.rmtree(filename)
        except:
            self.assertTrue(False, 'Failed during save & load diagnostics')

class LinearRegressionCreateTest(unittest.TestCase):
    """
    Unit test class for testing a Linear Regression create function.
    """

    @classmethod
    def setUpClass(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set up (Run only once)\n        '
        np.random.seed(42)
        (n, d) = (100, 10)
        self.sf = tc.SFrame()
        for i in range(d):
            self.sf.add_column(tc.SArray(np.random.rand(n)), inplace=True)
        target = np.random.rand(n)
        self.sf['target'] = target
        formula = 'target ~ ' + ' + '.join(['X{}'.format(i) for i in range(1, d + 1)])
        df = self.sf.to_dataframe()
        sm_model = sm.ols(formula, data=df).fit()
        self.loss = sm_model.ssr
        self.coef = list(sm_model.params)
        self.stderr = list(sm_model.bse)
        self.yhat = list(sm_model.fittedvalues)
        self.rmse = np.sqrt(sm_model.ssr / float(n))
        self.maxerr = abs(target - np.array(self.yhat)).max()
        self.def_kwargs = _DEFAULT_SOLVER_OPTIONS
        self.solver = 'newton'
        self.features = ', '.join(['X{}'.format(i) for i in range(1, d + 1)])
        self.target = 'target'

    def _test_coefficients(self, model, test_case, test_stderr):
        if False:
            return 10
        '\n        Check that the coefficient values are very close to the correct values.\n        '
        coefs = model.coefficients
        coef_list = list(coefs['value'])
        self.assertTrue(np.allclose(coef_list, self.coef, rtol=0.1, atol=0.1))
        if test_stderr:
            stderr_list = list(coefs['stderr'])
            self.assertTrue(np.allclose(stderr_list, self.stderr, rtol=0.001, atol=0.001))
        else:
            self.assertTrue('stderr' in coefs.column_names())
            self.assertEqual(list(coefs['stderr']), [None for v in coef_list])

    def _test_create_no_rescaling(self, sf, target, solver, kwargs):
        if False:
            i = 10
            return i + 15
        model = tc.linear_regression.create(self.sf, target=self.target, features=None, l2_penalty=0.0, l1_penalty=0.0, solver=solver, feature_rescaling=False, validation_set=None, **kwargs)
        test_case = 'solver = {solver}, kwargs = {kwargs}'.format(solver=solver, kwargs=kwargs)
        self.assertTrue(model is not None)
        self.assertTrue(abs(model.training_rmse - self.rmse) < 0.1, 'rmse failed: %s' % test_case)
        self._test_coefficients(model, test_case, solver == 'newton')

    def _test_create(self, sf, target, solver, kwargs):
        if False:
            for i in range(10):
                print('nop')
        model = tc.linear_regression.create(self.sf, target=self.target, features=None, l2_penalty=0.0, l1_penalty=0.0, solver=solver, feature_rescaling=True, validation_set=None, **kwargs)
        test_case = 'solver = {solver}, kwargs= {kwargs}'.format(solver=solver, kwargs=kwargs)
        self.assertTrue(model is not None)
        self.assertTrue(abs(model.training_rmse - self.rmse) < 0.1, 'rmse failed: %s' % test_case)
        self.assertTrue(abs(model.training_loss - self.loss) < 0.1, 'loss failed: %s' % test_case)
        self._test_coefficients(model, test_case, solver == 'newton')

    def test_create(self):
        if False:
            return 10
        '\n        Test linear regression create.\n        '
        kwargs = self.def_kwargs.copy()
        kwargs['convergence_threshold'] = 1e-06
        kwargs['max_iterations'] = 100
        for solver in ['newton', 'fista', 'lbfgs']:
            args = (self.sf, self.target, solver, kwargs)
            self._test_create(*args)
            self._test_create_no_rescaling(*args)

    def test_lbfgs(self):
        if False:
            i = 10
            return i + 15
        for m in [5, 21]:
            kwargs = self.def_kwargs.copy()
            kwargs.update({'lbfgs_memory_level': m})
            kwargs['max_iterations'] = 100
            args = (self.sf, self.target, 'lbfgs', kwargs)
            self._test_create(*args)
            self._test_create_no_rescaling(*args)
    '\n     Test detection of columns that are almost the same.\n  '

    def test_zero_variance_detection(self):
        if False:
            i = 10
            return i + 15
        sf = self.sf
        try:
            sf['error-column'] = 1
            model = tc.linear_regression.create(sf, self.target)
        except ToolkitError:
            pass
        try:
            sf['error-column'] = '1'
            model = tc.linear_regression.create(sf, self.target)
        except ToolkitError:
            pass
        try:
            sf['error-column'] = [[1] for i in sf]
            model = tc.linear_regression.create(sf, self.target)
        except ToolkitError:
            pass
        try:
            sf['error-column'] = [{1: 1} for i in sf]
            model = tc.linear_regression.create(sf, self.target)
        except ToolkitError:
            pass
        del sf['error-column']
    '\n     Test detection of columns with nan values\n  '

    def test_nan_detection(self):
        if False:
            for i in range(10):
                print('nop')
        sf = self.sf
        try:
            sf['error-column'] = np.nan
            model = tc.linear_regression.create(sf, self.target)
        except ToolkitError:
            pass
        try:
            sf['error-column'] = [[np.nan] for i in sf]
            model = tc.linear_regression.create(sf, self.target)
        except ToolkitError:
            pass
        try:
            sf['error-column'] = [{1: np.nan} for i in sf]
            model = tc.linear_regression.create(sf, self.target)
        except ToolkitError:
            pass
        del sf['error-column']

class VectorLinearRegressionTest(unittest.TestCase):
    """
    Unit test class for testing a Linear Regression create function.
    """

    @classmethod
    def setUpClass(self):
        if False:
            while True:
                i = 10
        '\n        Set up (Run only once)\n        '
        np.random.seed(15)
        (n, d) = (100, 3)
        self.sf = tc.SFrame()
        for i in range(d):
            self.sf.add_column(tc.SArray(np.random.rand(n)), inplace=True)
        self.sf['target'] = np.random.randint(2, size=n)
        df = self.sf.to_dataframe()
        formula = 'target ~ ' + ' + '.join(['X{}'.format(i + 1) for i in range(d)])
        sm_model = sm.ols(formula, data=df).fit()
        self.loss = sm_model.ssr
        self.coef = list(sm_model.params)
        self.stderr = list(sm_model.bse)
        self.yhat = list(sm_model.fittedvalues)
        self.rmse = np.sqrt(sm_model.ssr / float(n))
        self.target = 'target'
        self.sf['vec'] = self.sf.apply(lambda row: [row['X{}'.format(i + 1)] for i in range(d)])
        self.sf['vec'] = self.sf['vec'].apply(lambda x: x, array.array)
        self.features = ['vec']
        self.unpacked_features = ['vec[%s]' % i for i in range(d)]
        self.def_kwargs = _DEFAULT_SOLVER_OPTIONS

    def _test_coefficients(self, model):
        if False:
            print('Hello World!')
        '\n      Check that the coefficient values are very close to the correct values.\n      '
        coefs = model.coefficients
        coef_list = list(coefs['value'])
        stderr_list = list(coefs['stderr'])
        self.assertTrue(np.allclose(coef_list, self.coef, rtol=0.1, atol=0.1))
        self.assertTrue(np.allclose(stderr_list, self.stderr, rtol=0.001, atol=0.001))

    def _test_create(self, sf, target, features, solver, opts, rescaling):
        if False:
            i = 10
            return i + 15
        model = tc.linear_regression.create(sf, target, features, solver=solver, l2_penalty=0.0, feature_rescaling=rescaling, validation_set=None, **opts)
        test_case = 'solver = {solver}, opts = {opts}'.format(solver=solver, opts=opts)
        self.assertTrue(model is not None)
        self.assertTrue(abs(model.training_rmse - self.rmse) < 0.1, 'rmse failed: %s' % test_case)
        self.assertTrue(abs(model.training_loss - self.loss) < 0.1, 'loss failed: %s' % test_case)
        self._test_coefficients(model)
    '\n     Test linear regression create.\n  '

    def test_create(self):
        if False:
            return 10
        for solver in ['newton']:
            args = (self.sf, self.target, self.features, solver, self.def_kwargs, True)
            self._test_create(*args)
            args = (self.sf, self.target, self.features, solver, self.def_kwargs, False)
            self._test_create(*args)

    def test_features(self):
        if False:
            i = 10
            return i + 15
        model = tc.linear_regression.create(self.sf, self.target, self.features, feature_rescaling=False, validation_set=None)
        self.assertEqual(model.num_features, len(self.features))
        self.assertEqual(model.features, self.features)
        self.assertEqual(model.num_unpacked_features, len(self.unpacked_features))
        self.assertEqual(model.unpacked_features, self.unpacked_features)

class NDArrayLinearRegressionTest(unittest.TestCase):
    """
    Unit test class for testing a Linear Regression create function.
  """

    @classmethod
    def setUpClass(self):
        if False:
            while True:
                i = 10
        '\n        Set up (Run only once)\n    '
        np.random.seed(15)
        (n, d) = (100, 6)
        self.sf = tc.SFrame()
        for i in range(d):
            self.sf.add_column(tc.SArray(np.random.rand(n)), inplace=True)
        self.sf['target'] = np.random.randint(2, size=n)
        df = self.sf.to_dataframe()
        print(df)
        formula = 'target ~ ' + ' + '.join(['X{}'.format(i + 1) for i in range(d)])
        sm_model = sm.ols(formula, data=df).fit()
        self.loss = sm_model.ssr
        self.coef = list(sm_model.params)
        self.stderr = list(sm_model.bse)
        self.yhat = list(sm_model.fittedvalues)
        self.rmse = np.sqrt(sm_model.ssr / float(n))
        self.target = 'target'
        self.sf['nd_vec'] = self.sf.apply(lambda row: np.array([row['X{}'.format(i + 1)] for i in range(d)]).reshape((3, 2)))
        self.features = ['nd_vec']
        self.unpacked_features = ['nd_vec[%d,%d]' % (i, j) for i in range(3) for j in range(2)]
        self.def_kwargs = _DEFAULT_SOLVER_OPTIONS

    def _test_coefficients(self, model):
        if False:
            while True:
                i = 10
        '\n      Check that the coefficient values are very close to the correct values.\n      '
        coefs = model.coefficients
        coef_list = list(coefs['value'])
        stderr_list = list(coefs['stderr'])
        self.assertTrue(np.allclose(coef_list, self.coef, rtol=0.1, atol=0.1))
        self.assertTrue(np.allclose(stderr_list, self.stderr, rtol=0.001, atol=0.001))

    def _test_create(self, sf, target, features, solver, opts, rescaling):
        if False:
            for i in range(10):
                print('nop')
        model = tc.linear_regression.create(sf, target, features, solver=solver, l2_penalty=0.0, feature_rescaling=rescaling, validation_set=None, **opts)
        test_case = 'solver = {solver}, opts = {opts}'.format(solver=solver, opts=opts)
        self.assertTrue(model is not None)
        self.assertTrue(abs(model.training_rmse - self.rmse) < 0.1, 'rmse failed: %s' % test_case)
        self.assertTrue(abs(model.training_loss - self.loss) < 0.1, 'loss failed: %s' % test_case)
        self._test_coefficients(model)
    '\n     Test linear regression create.\n  '

    def test_create(self):
        if False:
            for i in range(10):
                print('nop')
        for solver in ['newton']:
            args = (self.sf, self.target, self.features, solver, self.def_kwargs, True)
            self._test_create(*args)
            args = (self.sf, self.target, self.features, solver, self.def_kwargs, False)
            self._test_create(*args)

    def test_features(self):
        if False:
            for i in range(10):
                print('nop')
        model = tc.linear_regression.create(self.sf, self.target, self.features, feature_rescaling=False, validation_set=None)
        self.assertEqual(model.num_features, len(self.features))
        self.assertEqual(model.features, self.features)
        self.assertEqual(model.num_unpacked_features, len(self.unpacked_features))
        self.assertEqual(model.unpacked_features, self.unpacked_features)

class DictLinearRegressionTest(unittest.TestCase):
    """
    Unit test class for testing a Linear Regression create function.
    """

    @classmethod
    def setUpClass(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set up (Run only once)\n        '
        np.random.seed(15)
        (n, d) = (100, 3)
        self.d = d
        self.sf = tc.SFrame()
        for i in range(d):
            self.sf.add_column(tc.SArray(np.random.rand(n)), inplace=True)
        self.sf['target'] = np.random.randint(2, size=n)
        df = self.sf.to_dataframe()
        formula = 'target ~ ' + ' + '.join(['X{}'.format(i + 1) for i in range(d)])
        sm_model = sm.ols(formula, data=df).fit()
        self.loss = sm_model.ssr
        self.coef = list(sm_model.params)
        self.stderr = list(sm_model.bse)
        self.yhat = list(sm_model.fittedvalues)
        self.rmse = np.sqrt(sm_model.ssr / float(n))
        self.target = 'target'
        self.sf['dict'] = self.sf.apply(lambda row: {i: row['X{}'.format(i + 1)] for i in range(d)})
        self.features = ['dict']
        self.unpacked_features = ['dict[%s]' % i for i in range(d)]
        self.def_kwargs = {'convergence_threshold': 1e-05, 'step_size': 1.0, 'max_iterations': 100}

    def _test_coefficients(self, model):
        if False:
            while True:
                i = 10
        '\n      Check that the coefficient values are very close to the correct values.\n      '
        coefs = model.coefficients
        coef_list = list(coefs['value'])
        stderr_list = list(coefs['stderr'])
        self.assertTrue(np.allclose(coef_list, self.coef, rtol=0.1, atol=0.1))
        self.assertTrue(np.allclose(stderr_list, self.stderr, rtol=0.001, atol=0.001))

    def _test_create(self, sf, target, features, solver, opts, rescaling):
        if False:
            i = 10
            return i + 15
        model = tc.linear_regression.create(sf, target, features, solver=solver, l2_penalty=0.0, feature_rescaling=rescaling, validation_set=None, **opts)
        test_case = 'solver = {solver}, opts = {opts}'.format(solver=solver, opts=opts)
        self.assertTrue(model is not None)
        self.assertTrue(abs(model.training_rmse - self.rmse) < 0.1, 'rmse failed: %s' % test_case)
        self.assertTrue(abs(model.training_loss - self.loss) < 0.1, 'loss failed: %s' % test_case)
        self._test_coefficients(model)

    def test_create(self):
        if False:
            while True:
                i = 10
        '\n        Test linear regression create.\n        '
        for solver in ['newton']:
            args = (self.sf, self.target, self.features, solver, self.def_kwargs, True)
            self._test_create(*args)
            args = (self.sf, self.target, self.features, solver, self.def_kwargs, False)
            self._test_create(*args)

    def test_features(self):
        if False:
            while True:
                i = 10
        d = self.d
        self.sf['dict'] = self.sf.apply(lambda row: {i: row['X{}'.format(i + 1)] for i in range(d)})
        model = tc.linear_regression.create(self.sf, self.target, self.features, feature_rescaling=False, validation_set=None)
        self.assertEqual(model.num_features, len(self.features))
        self.assertEqual(model.features, self.features)
        self.assertEqual(model.num_unpacked_features, len(self.unpacked_features))
        self.assertEqual(model.unpacked_features, self.unpacked_features)

    def test_predict_extra_cols(self):
        if False:
            print('Hello World!')
        sf = self.sf[:]
        model = tc.linear_regression.create(sf, self.target, self.features, feature_rescaling=False, validation_set=None)
        pred = model.predict(sf)
        sf['dict'] = sf['dict'].apply(lambda x: dict(list(x.items()) + list({'extra_col': 0, 'extra_col_2': 1}.items())))
        pred2 = model.predict(sf)
        self.assertEqual(list(pred), list(pred2))

    def test_evaluate_extra_cols(self):
        if False:
            while True:
                i = 10
        sf = self.sf[:]
        model = tc.linear_regression.create(sf, self.target, self.features, feature_rescaling=False, validation_set=None)
        eval1 = model.evaluate(sf)
        sf['dict'] = sf['dict'].apply(lambda x: dict(list(x.items()) + list({'extra_col': 0, 'extra_col_2': 1}.items())))
        eval2 = model.evaluate(sf)
        self.assertEqual(eval1, eval2)

class ListCategoricalLinearRegressionTest(unittest.TestCase):
    """
    Unit test class for testing a Linear Regression create function.
    """

    @classmethod
    def setUpClass(self):
        if False:
            i = 10
            return i + 15
        '\n        Set up (Run only once)\n        '
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
        self.sf['target'] = np.random.randint(2, size=n)
        df = self.sf.to_dataframe()
        formula = 'target ~ species + ' + ' + '.join(['X{}'.format(i + 1) for i in range(d)])
        sm_model = sm.ols(formula, data=df).fit()
        self.loss = sm_model.ssr
        self.stderr = list(sm_model.bse)
        self.coef = list(sm_model.params)
        self.yhat = list(sm_model.fittedvalues)
        self.rmse = np.sqrt(sm_model.ssr / float(n))
        self.target = 'target'
        self.features = ['species', 'X1', 'X2', 'X3']
        self.unpacked_features = ['species[dog]', 'species[foosa]', 'X1', 'X2', 'X3']
        self.sf['species'] = self.sf['species'].apply(lambda x: [x])
        self.def_kwargs = {'convergence_threshold': 1e-05, 'step_size': 1.0, 'max_iterations': 100}

    def _test_coefficients(self, model, test_stderr):
        if False:
            return 10
        '\n        Check that the coefficient values are very close to the correct values.\n        '
        coefs = model.coefficients
        coef_list = list(coefs['value'])
        self.assertTrue(np.allclose(coef_list, self.coef, rtol=0.1, atol=0.1))
        if test_stderr:
            stderr_list = list(coefs['stderr'])
            self.assertTrue(np.allclose(stderr_list, self.stderr, rtol=0.001, atol=0.001))
        else:
            self.assertTrue('stderr' in coefs.column_names())
            self.assertEqual(list(coefs['stderr']), [None for v in coef_list])

    def _test_create(self, sf, target, features, solver, opts, rescaling):
        if False:
            while True:
                i = 10
        model = tc.linear_regression.create(sf, target, features, solver=solver, l2_penalty=0.0, feature_rescaling=rescaling, validation_set=None, **opts)
        test_case = 'solver = {solver}, opts = {opts}'.format(solver=solver, opts=opts)
        self.assertTrue(model is not None)
        self.assertTrue(abs(model.training_loss - self.loss) < 0.1, 'loss failed: %s' % test_case)
        self.assertTrue(abs(model.training_rmse - self.rmse) < 0.1, 'rmse failed: %s' % test_case)
        self._test_coefficients(model, solver == 'newton')

    def test_create(self):
        if False:
            return 10
        '\n        Test linear regression create.\n        '
        for solver in ['newton', 'lbfgs', 'fista']:
            args = (self.sf, self.target, self.features, solver, self.def_kwargs, True)
            self._test_create(*args)
            args = (self.sf, self.target, self.features, solver, self.def_kwargs, False)
            self._test_create(*args)

class CategoricalLinearRegressionTest(unittest.TestCase):
    """
    Unit test class for testing a Linear Regression create function.
    """

    @classmethod
    def setUpClass(self):
        if False:
            return 10
        '\n        Set up (Run only once)\n        '
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
        self.sf['target'] = np.random.randint(2, size=n)
        df = self.sf.to_dataframe()
        formula = 'target ~ species + ' + ' + '.join(['X{}'.format(i + 1) for i in range(d)])
        sm_model = sm.ols(formula, data=df).fit()
        self.loss = sm_model.ssr
        self.stderr = list(sm_model.bse)
        self.coef = list(sm_model.params)
        self.yhat = list(sm_model.fittedvalues)
        self.rmse = np.sqrt(sm_model.ssr / float(n))
        self.target = 'target'
        self.features = ['species', 'X1', 'X2', 'X3']
        self.unpacked_features = ['species', 'X1', 'X2', 'X3']
        self.def_kwargs = {'convergence_threshold': 1e-05, 'step_size': 1.0, 'max_iterations': 100}

    def _test_coefficients(self, model, test_stderr):
        if False:
            print('Hello World!')
        '\n        Check that the coefficient values are very close to the correct values.\n        '
        coefs = model.coefficients
        coef_list = list(coefs['value'])
        self.assertTrue(np.allclose(coef_list, self.coef, rtol=0.1, atol=0.1))
        if test_stderr:
            stderr_list = list(coefs['stderr'])
            self.assertTrue(np.allclose(stderr_list, self.stderr, rtol=0.001, atol=0.001))
        else:
            self.assertTrue('stderr' in coefs.column_names())
            self.assertEqual(list(coefs['stderr']), [None for v in coef_list])

    def _test_create(self, sf, target, features, solver, opts, rescaling):
        if False:
            print('Hello World!')
        model = tc.linear_regression.create(sf, target, features, solver=solver, l2_penalty=0.0, feature_rescaling=rescaling, validation_set=None, **opts)
        test_case = 'solver = {solver}, opts = {opts}'.format(solver=solver, opts=opts)
        self.assertTrue(model is not None)
        self.assertTrue(abs(model.training_loss - self.loss) < 0.1, 'loss failed: %s' % test_case)
        self.assertTrue(abs(model.training_rmse - self.rmse) < 0.1, 'rmse failed: %s' % test_case)
        self._test_coefficients(model, solver == 'newton')

    def test_create(self):
        if False:
            return 10
        '\n        Test linear regression create.\n        '
        for solver in ['newton', 'lbfgs', 'fista']:
            args = (self.sf, self.target, self.features, solver, self.def_kwargs, True)
            self._test_create(*args)
            args = (self.sf, self.target, self.features, solver, self.def_kwargs, False)
            self._test_create(*args)

    def test_predict_extra_cols(self):
        if False:
            while True:
                i = 10
        model = tc.linear_regression.create(self.sf, self.target, self.features, feature_rescaling=False, validation_set=None)
        X_sf = self.sf.copy()
        X_sf['species'] = X_sf['species'].apply(lambda x: x if x != 'foosa' else 'rat')
        pred = model.predict(X_sf)

    def test_evaluate_extra_cols(self):
        if False:
            return 10
        model = tc.linear_regression.create(self.sf, self.target, self.features, feature_rescaling=False, validation_set=None)
        X_sf = self.sf.copy()
        X_sf['species'] = X_sf['species'].apply(lambda x: x if x != 'foosa' else 'rat')
        pred = model.evaluate(X_sf)

    def test_features(self):
        if False:
            return 10
        model = tc.linear_regression.create(self.sf, self.target, self.features, feature_rescaling=False, validation_set=None)
        self.assertEqual(model.num_features, len(self.features))
        self.assertEqual(model.features, self.features)
        self.assertEqual(model.num_unpacked_features, len(self.unpacked_features))
        self.assertEqual(model.unpacked_features, self.unpacked_features)

class L1LinearRegressionTest(unittest.TestCase):
    """
    Unit test class for testing a Linear Regression create function.
    """

    @classmethod
    def setUpClass(self):
        if False:
            i = 10
            return i + 15
        '\n        Set up (Run only once)\n        '
        test_data = 'y,0,1,2,3,4\n      38,0,3,1,0,1.47\n      58,1,2,2,8,4.38\n      30,1,1,1,0,1.64\n      50,1,1,3,0,2.54\n      49,1,1,3,1,2.06\n      45,0,3,1,4,4.76\n      42,1,1,2,0,3.05\n      59,0,3,3,3,2.73\n      47,1,2,1,0,3.14\n      34,0,1,1,3,4.42\n      53,0,2,3,0,2.36\n      35,1,1,1,1,4.29\n      42,0,1,2,2,3.81\n      42,0,1,2,2,3.84\n      51,0,3,2,7,3.15\n      51,1,2,1,8,5.07\n      40,0,1,2,3,2.73\n      48,1,2,1,1,3.56\n      34,1,1,1,7,3.54\n      46,1,2,1,2,2.71\n      45,0,1,2,6,5.18\n      50,1,1,3,2,2.66\n      61,0,3,3,3,3.7\n      62,1,3,1,2,3.75\n      51,0,1,3,8,3.96\n      59,0,3,3,0,2.88\n      65,1,2,3,5,3.37\n      49,0,1,3,0,2.84\n      37,1,1,1,9,5.12'
        dataset = 'data_file%s.csv' % str(uuid.uuid4())
        self.dataset = dataset
        f = open(dataset, 'w')
        f.write(test_data)
        f.close()
        self.def_kwargs = {'convergence_threshold': 1e-05, 'max_iterations': 1000}
        self.features = ['0', '1', '2', '3', '4']
        self.target = 'y'
        type_dict = {n: float for n in self.features + [self.target]}
        self.sf = tc.SFrame.read_csv(dataset, header=True, delimiter=',', column_type_hints=type_dict)
        feature_matrix = np.genfromtxt(dataset, delimiter=',', skip_header=1)
        X = feature_matrix[:, 1:]
        y = feature_matrix[:, 0]
        self.examples = X.shape[0]
        self.l1_penalty = 10.0
        clf = linear_model.ElasticNet(alpha=self.l1_penalty / (2 * self.examples), l1_ratio=1)
        clf.fit(X, y)
        self.coef = np.append(clf.intercept_, clf.coef_)
        self.predictions = clf.predict(X)
        self.loss = np.dot(self.predictions - y, self.predictions - y)
        self.rmse = np.sqrt(self.loss / self.examples)

    @classmethod
    def tearDownClass(self):
        if False:
            for i in range(10):
                print('nop')
        os.remove(self.dataset)

    def _test_coefficients(self, model):
        if False:
            print('Hello World!')
        '\n      Check that the coefficient values are very close to the correct values.\n      '
        coefs = model.coefficients
        coef_list = list(coefs['value'])
        self.assertTrue(np.allclose(coef_list, self.coef, rtol=0.01, atol=0.01), '%s vs %s' % (coef_list, self.coef))

    def _test_create(self, sf, target, features, solver, opts):
        if False:
            for i in range(10):
                print('nop')
        model = tc.linear_regression.create(sf, target, features, solver=solver, l1_penalty=self.l1_penalty, l2_penalty=0.0, feature_rescaling=False, validation_set=None, **opts)
        test_case = 'solver = {solver}, opts = {opts}'.format(solver=solver, opts=opts)
        self.assertTrue(model is not None)
        self._test_coefficients(model)
        self.assertTrue(abs(model.training_rmse - self.rmse) < 0.1, 'rmse failed: %s' % test_case)

    def test_create(self):
        if False:
            return 10
        for solver in ['fista']:
            args = (self.sf, self.target, self.features, solver, self.def_kwargs)
            self._test_create(*args)

class L2LinearRegressionTest(unittest.TestCase):
    """
    Unit test class for testing a Linear Regression create function.
    """

    @classmethod
    def setUpClass(self):
        if False:
            print('Hello World!')
        '\n        Set up (Run only once)\n        '
        test_data = 'y,0,1,2,3,4\n      38,0,3,1,0,1.47\n      58,1,2,2,8,4.38\n      30,1,1,1,0,1.64\n      50,1,1,3,0,2.54\n      49,1,1,3,1,2.06\n      45,0,3,1,4,4.76\n      42,1,1,2,0,3.05\n      59,0,3,3,3,2.73\n      47,1,2,1,0,3.14\n      34,0,1,1,3,4.42\n      53,0,2,3,0,2.36\n      35,1,1,1,1,4.29\n      42,0,1,2,2,3.81\n      42,0,1,2,2,3.84\n      51,0,3,2,7,3.15\n      51,1,2,1,8,5.07\n      40,0,1,2,3,2.73\n      48,1,2,1,1,3.56\n      34,1,1,1,7,3.54\n      46,1,2,1,2,2.71\n      45,0,1,2,6,5.18\n      50,1,1,3,2,2.66\n      61,0,3,3,3,3.7\n      62,1,3,1,2,3.75\n      51,0,1,3,8,3.96\n      59,0,3,3,0,2.88\n      65,1,2,3,5,3.37\n      49,0,1,3,0,2.84\n      37,1,1,1,9,5.12'
        dataset = 'data_file%s.csv' % str(uuid.uuid4())
        self.dataset = dataset
        f = open(dataset, 'w')
        f.write(test_data)
        f.close()
        self.def_kwargs = {'convergence_threshold': 1e-05, 'step_size': 1.0, 'lbfgs_memory_level': 11, 'max_iterations': 1000}
        self.features = ['0', '1', '2', '3', '4']
        self.target = 'y'
        type_dict = {n: float for n in self.features + [self.target]}
        self.sf = tc.SFrame.read_csv(dataset, header=True, delimiter=',', column_type_hints=type_dict)
        feature_matrix = np.genfromtxt(dataset, delimiter=',', skip_header=1)
        X = feature_matrix[:, 1:]
        y = feature_matrix[:, 0]
        self.examples = X.shape[0]
        self.variables = X.shape[1] + 1
        self.l2_penalty = 10.0
        clf = linear_model.ElasticNet(alpha=self.l2_penalty / self.examples, l1_ratio=0)
        clf.fit(X, y)
        self.coef = np.append(clf.intercept_, clf.coef_)
        self.predictions = clf.predict(X)
        self.loss = np.dot(self.predictions - y, self.predictions - y)
        self.rmse = np.sqrt(self.loss / self.examples)

    @classmethod
    def tearDownClass(self):
        if False:
            while True:
                i = 10
        os.remove(self.dataset)

    def _test_coefficients(self, model):
        if False:
            return 10
        '\n        Check that the coefficient values are very close to the correct values.\n        '
        coefs = model.coefficients
        coef_list = list(coefs['value'])
        self.assertTrue(np.allclose(coef_list, self.coef, rtol=0.1, atol=0.1))

    def _test_create(self, sf, target, features, solver, opts):
        if False:
            for i in range(10):
                print('nop')
        model = tc.linear_regression.create(sf, target, features, solver=solver, l2_penalty=self.l2_penalty, feature_rescaling=False, validation_set=None, **opts)
        test_case = 'solver = {solver}, opts = {opts}'.format(solver=solver, opts=opts)
        self.assertTrue(model is not None)
        self.assertTrue(abs(model.training_rmse - self.rmse) < 0.1, 'rmse failed: %s' % test_case)
        self._test_coefficients(model)

    def test_create(self):
        if False:
            print('Hello World!')
        '\n        Test linear regression create.\n        '
        for solver in ['newton', 'lbfgs', 'fista']:
            args = (self.sf, self.target, self.features, solver, self.def_kwargs)
            self._test_create(*args)

class ElasticNetLinearRegressionTest(unittest.TestCase):
    """
    Unit test class for testing a Linear Regression create function.
    """

    @classmethod
    def setUpClass(self):
        if False:
            return 10
        '\n        Set up (Run only once)\n        '
        test_data = 'y,0,1,2,3,4\n      38,0,3,1,0,1.47\n      58,1,2,2,8,4.38\n      30,1,1,1,0,1.64\n      50,1,1,3,0,2.54\n      49,1,1,3,1,2.06\n      45,0,3,1,4,4.76\n      42,1,1,2,0,3.05\n      59,0,3,3,3,2.73\n      47,1,2,1,0,3.14\n      34,0,1,1,3,4.42\n      53,0,2,3,0,2.36\n      35,1,1,1,1,4.29\n      42,0,1,2,2,3.81\n      42,0,1,2,2,3.84\n      51,0,3,2,7,3.15\n      51,1,2,1,8,5.07\n      40,0,1,2,3,2.73\n      48,1,2,1,1,3.56\n      34,1,1,1,7,3.54\n      46,1,2,1,2,2.71\n      45,0,1,2,6,5.18\n      50,1,1,3,2,2.66\n      61,0,3,3,3,3.7\n      62,1,3,1,2,3.75\n      51,0,1,3,8,3.96\n      59,0,3,3,0,2.88\n      65,1,2,3,5,3.37\n      49,0,1,3,0,2.84\n      37,1,1,1,9,5.12'
        dataset = 'data_file%s.csv' % str(uuid.uuid4())
        self.dataset = dataset
        f = open(dataset, 'w')
        f.write(test_data)
        f.close()
        self.def_kwargs = {'convergence_threshold': 1e-05, 'step_size': 1.0, 'lbfgs_memory_level': 3, 'max_iterations': 1000}
        self.features = ['0', '1', '2', '3', '4']
        self.target = 'y'
        type_dict = {n: float for n in self.features + [self.target]}
        self.sf = tc.SFrame.read_csv(dataset, header=True, delimiter=',', column_type_hints=type_dict)
        feature_matrix = np.genfromtxt(dataset, delimiter=',', skip_header=1)
        X = feature_matrix[:, 1:]
        y = feature_matrix[:, 0]
        self.examples = X.shape[0]
        self.penalty = 10.0
        self.ratio = 0.5
        clf = linear_model.ElasticNet(alpha=self.penalty / self.examples, l1_ratio=0.5)
        clf.fit(X, y)
        self.coef = np.append(clf.intercept_, clf.coef_)
        self.predictions = clf.predict(X)
        self.loss = np.dot(self.predictions - y, self.predictions - y)
        self.rmse = np.sqrt(self.loss / self.examples)

    @classmethod
    def tearDownClass(self):
        if False:
            return 10
        os.remove(self.dataset)

    def _test_coefficients(self, model):
        if False:
            while True:
                i = 10
        '\n        Check that the coefficient values are very close to the correct values.\n        '
        coefs = model.coefficients
        coef_list = list(coefs['value'])
        self.assertTrue(np.allclose(coef_list, self.coef, rtol=0.1, atol=0.1), '%s vs %s' % (coef_list, self.coef))

    def _test_create(self, sf, target, features, solver, opts):
        if False:
            return 10
        model = tc.linear_regression.create(sf, target, features, solver=solver, l1_penalty=self.penalty, l2_penalty=0.5 * self.penalty, feature_rescaling=False, validation_set=None, **opts)
        test_case = 'solver = {solver}, opts = {opts}'.format(solver=solver, opts=opts)
        self.assertTrue(model is not None)
        self.assertTrue(abs(model.training_rmse - self.rmse) < 0.1, 'rmse failed: %s' % test_case)
        self._test_coefficients(model)

    def test_create(self):
        if False:
            i = 10
            return i + 15
        '\n        Test linear regression create.\n        '
        for solver in ['fista']:
            args = (self.sf, self.target, self.features, solver, self.def_kwargs)
            self._test_create(*args)

class ValidationDataLinearRegressionTest(unittest.TestCase):
    """
    Unit test class for testing create with validation data.
    """

    @classmethod
    def setUpClass(self):
        if False:
            while True:
                i = 10
        '\n        Set up (Run only once)\n        '
        test_data = 'y,0,1,2,3,4\n      38,0,3,1,0,1.47\n      58,1,2,2,8,4.38\n      30,1,1,1,0,1.64\n      50,1,1,3,0,2.54\n      49,1,1,3,1,2.06\n      45,0,3,1,4,4.76\n      42,1,1,2,0,3.05\n      59,0,3,3,3,2.73\n      47,1,2,1,0,3.14\n      34,0,1,1,3,4.42\n      53,0,2,3,0,2.36\n      35,1,1,1,1,4.29\n      42,0,1,2,2,3.81\n      42,0,1,2,2,3.84\n      51,0,3,2,7,3.15\n      51,1,2,1,8,5.07\n      40,0,1,2,3,2.73\n      48,1,2,1,1,3.56\n      34,1,1,1,7,3.54\n      46,1,2,1,2,2.71\n      45,0,1,2,6,5.18\n      50,1,1,3,2,2.66\n      61,0,3,3,3,3.7\n      62,1,3,1,2,3.75\n      51,0,1,3,8,3.96\n      59,0,3,3,0,2.88\n      65,1,2,3,5,3.37\n      49,0,1,3,0,2.84\n      37,1,1,1,9,5.12'
        dataset = 'data_file%s.csv' % str(uuid.uuid4())
        self.dataset = dataset
        f = open(dataset, 'w')
        f.write(test_data)
        f.close()
        self.def_kwargs = {'convergence_threshold': 0.0001, 'step_size': 1.0, 'lbfgs_memory_level': 11, 'max_iterations': 200}
        self.features = ['0', '1', '2', '3', '4']
        self.target = 'y'
        type_dict = {n: float for n in self.features + [self.target]}
        self.sf = tc.SFrame.read_csv(dataset, header=True, delimiter=',', column_type_hints=type_dict)

    @classmethod
    def tearDownClass(self):
        if False:
            i = 10
            return i + 15
        os.remove(self.dataset)

    def test_valid_set(self):
        if False:
            return 10
        m = tc.linear_regression.create(self.sf, target=self.target, validation_set=self.sf)
        self.assertTrue(m is not None)
        self.assertTrue(isinstance(m.progress, tc.SFrame))
        m = tc.linear_regression.create(self.sf, target=self.target, validation_set='auto')
        self.assertTrue(m is not None)
        self.assertTrue(isinstance(m.progress, tc.SFrame))
        m = tc.linear_regression.create(self.sf, target=self.target, validation_set=None)
        self.assertTrue(m is not None)
        self.assertTrue(isinstance(m.progress, tc.SFrame))