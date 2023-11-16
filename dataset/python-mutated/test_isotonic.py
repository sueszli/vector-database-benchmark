import copy
import pickle
import warnings
import numpy as np
import pytest
from scipy.special import expit
import sklearn
from sklearn.datasets import make_regression
from sklearn.isotonic import IsotonicRegression, _make_unique, check_increasing, isotonic_regression
from sklearn.utils import shuffle
from sklearn.utils._testing import assert_allclose, assert_array_almost_equal, assert_array_equal
from sklearn.utils.validation import check_array

def test_permutation_invariance():
    if False:
        i = 10
        return i + 15
    ir = IsotonicRegression()
    x = [1, 2, 3, 4, 5, 6, 7]
    y = [1, 41, 51, 1, 2, 5, 24]
    sample_weight = [1, 2, 3, 4, 5, 6, 7]
    (x_s, y_s, sample_weight_s) = shuffle(x, y, sample_weight, random_state=0)
    y_transformed = ir.fit_transform(x, y, sample_weight=sample_weight)
    y_transformed_s = ir.fit(x_s, y_s, sample_weight=sample_weight_s).transform(x)
    assert_array_equal(y_transformed, y_transformed_s)

def test_check_increasing_small_number_of_samples():
    if False:
        for i in range(10):
            print('nop')
    x = [0, 1, 2]
    y = [1, 1.1, 1.05]
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        is_increasing = check_increasing(x, y)
    assert is_increasing

def test_check_increasing_up():
    if False:
        while True:
            i = 10
    x = [0, 1, 2, 3, 4, 5]
    y = [0, 1.5, 2.77, 8.99, 8.99, 50]
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        is_increasing = check_increasing(x, y)
    assert is_increasing

def test_check_increasing_up_extreme():
    if False:
        return 10
    x = [0, 1, 2, 3, 4, 5]
    y = [0, 1, 2, 3, 4, 5]
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        is_increasing = check_increasing(x, y)
    assert is_increasing

def test_check_increasing_down():
    if False:
        print('Hello World!')
    x = [0, 1, 2, 3, 4, 5]
    y = [0, -1.5, -2.77, -8.99, -8.99, -50]
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        is_increasing = check_increasing(x, y)
    assert not is_increasing

def test_check_increasing_down_extreme():
    if False:
        while True:
            i = 10
    x = [0, 1, 2, 3, 4, 5]
    y = [0, -1, -2, -3, -4, -5]
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        is_increasing = check_increasing(x, y)
    assert not is_increasing

def test_check_ci_warn():
    if False:
        i = 10
        return i + 15
    x = [0, 1, 2, 3, 4, 5]
    y = [0, -1, 2, -3, 4, -5]
    msg = 'interval'
    with pytest.warns(UserWarning, match=msg):
        is_increasing = check_increasing(x, y)
    assert not is_increasing

def test_isotonic_regression():
    if False:
        i = 10
        return i + 15
    y = np.array([3, 7, 5, 9, 8, 7, 10])
    y_ = np.array([3, 6, 6, 8, 8, 8, 10])
    assert_array_equal(y_, isotonic_regression(y))
    y = np.array([10, 0, 2])
    y_ = np.array([4, 4, 4])
    assert_array_equal(y_, isotonic_regression(y))
    x = np.arange(len(y))
    ir = IsotonicRegression(y_min=0.0, y_max=1.0)
    ir.fit(x, y)
    assert_array_equal(ir.fit(x, y).transform(x), ir.fit_transform(x, y))
    assert_array_equal(ir.transform(x), ir.predict(x))
    perm = np.random.permutation(len(y))
    ir = IsotonicRegression(y_min=0.0, y_max=1.0)
    assert_array_equal(ir.fit_transform(x[perm], y[perm]), ir.fit_transform(x, y)[perm])
    assert_array_equal(ir.transform(x[perm]), ir.transform(x)[perm])
    ir = IsotonicRegression()
    assert_array_equal(ir.fit_transform(np.ones(len(x)), y), np.mean(y))

def test_isotonic_regression_ties_min():
    if False:
        while True:
            i = 10
    x = [1, 1, 2, 3, 4, 5]
    y = [1, 2, 3, 4, 5, 6]
    y_true = [1.5, 1.5, 3, 4, 5, 6]
    ir = IsotonicRegression()
    ir.fit(x, y)
    assert_array_equal(ir.fit(x, y).transform(x), ir.fit_transform(x, y))
    assert_array_equal(y_true, ir.fit_transform(x, y))

def test_isotonic_regression_ties_max():
    if False:
        return 10
    x = [1, 2, 3, 4, 5, 5]
    y = [1, 2, 3, 4, 5, 6]
    y_true = [1, 2, 3, 4, 5.5, 5.5]
    ir = IsotonicRegression()
    ir.fit(x, y)
    assert_array_equal(ir.fit(x, y).transform(x), ir.fit_transform(x, y))
    assert_array_equal(y_true, ir.fit_transform(x, y))

def test_isotonic_regression_ties_secondary_():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test isotonic regression fit, transform  and fit_transform\n    against the "secondary" ties method and "pituitary" data from R\n     "isotone" package, as detailed in: J. d. Leeuw, K. Hornik, P. Mair,\n     Isotone Optimization in R: Pool-Adjacent-Violators Algorithm\n    (PAVA) and Active Set Methods\n\n    Set values based on pituitary example and\n     the following R command detailed in the paper above:\n    > library("isotone")\n    > data("pituitary")\n    > res1 <- gpava(pituitary$age, pituitary$size, ties="secondary")\n    > res1$x\n\n    `isotone` version: 1.0-2, 2014-09-07\n    R version: R version 3.1.1 (2014-07-10)\n    '
    x = [8, 8, 8, 10, 10, 10, 12, 12, 12, 14, 14]
    y = [21, 23.5, 23, 24, 21, 25, 21.5, 22, 19, 23.5, 25]
    y_true = [22.22222, 22.22222, 22.22222, 22.22222, 22.22222, 22.22222, 22.22222, 22.22222, 22.22222, 24.25, 24.25]
    ir = IsotonicRegression()
    ir.fit(x, y)
    assert_array_almost_equal(ir.transform(x), y_true, 4)
    assert_array_almost_equal(ir.fit_transform(x, y), y_true, 4)

def test_isotonic_regression_with_ties_in_differently_sized_groups():
    if False:
        print('Hello World!')
    '\n    Non-regression test to handle issue 9432:\n    https://github.com/scikit-learn/scikit-learn/issues/9432\n\n    Compare against output in R:\n    > library("isotone")\n    > x <- c(0, 1, 1, 2, 3, 4)\n    > y <- c(0, 0, 1, 0, 0, 1)\n    > res1 <- gpava(x, y, ties="secondary")\n    > res1$x\n\n    `isotone` version: 1.1-0, 2015-07-24\n    R version: R version 3.3.2 (2016-10-31)\n    '
    x = np.array([0, 1, 1, 2, 3, 4])
    y = np.array([0, 0, 1, 0, 0, 1])
    y_true = np.array([0.0, 0.25, 0.25, 0.25, 0.25, 1.0])
    ir = IsotonicRegression()
    ir.fit(x, y)
    assert_array_almost_equal(ir.transform(x), y_true)
    assert_array_almost_equal(ir.fit_transform(x, y), y_true)

def test_isotonic_regression_reversed():
    if False:
        print('Hello World!')
    y = np.array([10, 9, 10, 7, 6, 6.1, 5])
    y_ = IsotonicRegression(increasing=False).fit_transform(np.arange(len(y)), y)
    assert_array_equal(np.ones(y_[:-1].shape), y_[:-1] - y_[1:] >= 0)

def test_isotonic_regression_auto_decreasing():
    if False:
        return 10
    y = np.array([10, 9, 10, 7, 6, 6.1, 5])
    x = np.arange(len(y))
    ir = IsotonicRegression(increasing='auto')
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        y_ = ir.fit_transform(x, y)
        assert all(['invalid value encountered in ' in str(warn.message) for warn in w])
    is_increasing = y_[0] < y_[-1]
    assert not is_increasing

def test_isotonic_regression_auto_increasing():
    if False:
        print('Hello World!')
    y = np.array([5, 6.1, 6, 7, 10, 9, 10])
    x = np.arange(len(y))
    ir = IsotonicRegression(increasing='auto')
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        y_ = ir.fit_transform(x, y)
        assert all(['invalid value encountered in ' in str(warn.message) for warn in w])
    is_increasing = y_[0] < y_[-1]
    assert is_increasing

def test_assert_raises_exceptions():
    if False:
        while True:
            i = 10
    ir = IsotonicRegression()
    rng = np.random.RandomState(42)
    msg = 'Found input variables with inconsistent numbers of samples'
    with pytest.raises(ValueError, match=msg):
        ir.fit([0, 1, 2], [5, 7, 3], [0.1, 0.6])
    with pytest.raises(ValueError, match=msg):
        ir.fit([0, 1, 2], [5, 7])
    msg = 'X should be a 1d array'
    with pytest.raises(ValueError, match=msg):
        ir.fit(rng.randn(3, 10), [0, 1, 2])
    msg = 'Isotonic regression input X should be a 1d array'
    with pytest.raises(ValueError, match=msg):
        ir.transform(rng.randn(3, 10))

def test_isotonic_sample_weight_parameter_default_value():
    if False:
        i = 10
        return i + 15
    ir = IsotonicRegression()
    rng = np.random.RandomState(42)
    n = 100
    x = np.arange(n)
    y = rng.randint(-50, 50, size=(n,)) + 50.0 * np.log(1 + np.arange(n))
    weights = np.ones(n)
    y_set_value = ir.fit_transform(x, y, sample_weight=weights)
    y_default_value = ir.fit_transform(x, y)
    assert_array_equal(y_set_value, y_default_value)

def test_isotonic_min_max_boundaries():
    if False:
        i = 10
        return i + 15
    ir = IsotonicRegression(y_min=2, y_max=4)
    n = 6
    x = np.arange(n)
    y = np.arange(n)
    y_test = [2, 2, 2, 3, 4, 4]
    y_result = np.round(ir.fit_transform(x, y))
    assert_array_equal(y_result, y_test)

def test_isotonic_sample_weight():
    if False:
        print('Hello World!')
    ir = IsotonicRegression()
    x = [1, 2, 3, 4, 5, 6, 7]
    y = [1, 41, 51, 1, 2, 5, 24]
    sample_weight = [1, 2, 3, 4, 5, 6, 7]
    expected_y = [1, 13.95, 13.95, 13.95, 13.95, 13.95, 24]
    received_y = ir.fit_transform(x, y, sample_weight=sample_weight)
    assert_array_equal(expected_y, received_y)

def test_isotonic_regression_oob_raise():
    if False:
        for i in range(10):
            print('nop')
    y = np.array([3, 7, 5, 9, 8, 7, 10])
    x = np.arange(len(y))
    ir = IsotonicRegression(increasing='auto', out_of_bounds='raise')
    ir.fit(x, y)
    msg = 'in x_new is below the interpolation range'
    with pytest.raises(ValueError, match=msg):
        ir.predict([min(x) - 10, max(x) + 10])

def test_isotonic_regression_oob_clip():
    if False:
        print('Hello World!')
    y = np.array([3, 7, 5, 9, 8, 7, 10])
    x = np.arange(len(y))
    ir = IsotonicRegression(increasing='auto', out_of_bounds='clip')
    ir.fit(x, y)
    y1 = ir.predict([min(x) - 10, max(x) + 10])
    y2 = ir.predict(x)
    assert max(y1) == max(y2)
    assert min(y1) == min(y2)

def test_isotonic_regression_oob_nan():
    if False:
        return 10
    y = np.array([3, 7, 5, 9, 8, 7, 10])
    x = np.arange(len(y))
    ir = IsotonicRegression(increasing='auto', out_of_bounds='nan')
    ir.fit(x, y)
    y1 = ir.predict([min(x) - 10, max(x) + 10])
    assert sum(np.isnan(y1)) == 2

def test_isotonic_regression_pickle():
    if False:
        i = 10
        return i + 15
    y = np.array([3, 7, 5, 9, 8, 7, 10])
    x = np.arange(len(y))
    ir = IsotonicRegression(increasing='auto', out_of_bounds='clip')
    ir.fit(x, y)
    ir_ser = pickle.dumps(ir, pickle.HIGHEST_PROTOCOL)
    ir2 = pickle.loads(ir_ser)
    np.testing.assert_array_equal(ir.predict(x), ir2.predict(x))

def test_isotonic_duplicate_min_entry():
    if False:
        print('Hello World!')
    x = [0, 0, 1]
    y = [0, 0, 1]
    ir = IsotonicRegression(increasing=True, out_of_bounds='clip')
    ir.fit(x, y)
    all_predictions_finite = np.all(np.isfinite(ir.predict(x)))
    assert all_predictions_finite

def test_isotonic_ymin_ymax():
    if False:
        for i in range(10):
            print('nop')
    x = np.array([1.263, 1.318, -0.572, 0.307, -0.707, -0.176, -1.599, 1.059, 1.396, 1.906, 0.21, 0.028, -0.081, 0.444, 0.018, -0.377, -0.896, -0.377, -1.327, 0.18])
    y = isotonic_regression(x, y_min=0.0, y_max=0.1)
    assert np.all(y >= 0)
    assert np.all(y <= 0.1)
    y = isotonic_regression(x, y_min=0.0, y_max=0.1, increasing=False)
    assert np.all(y >= 0)
    assert np.all(y <= 0.1)
    y = isotonic_regression(x, y_min=0.0, increasing=False)
    assert np.all(y >= 0)

def test_isotonic_zero_weight_loop():
    if False:
        print('Hello World!')
    rng = np.random.RandomState(42)
    regression = IsotonicRegression()
    n_samples = 50
    x = np.linspace(-3, 3, n_samples)
    y = x + rng.uniform(size=n_samples)
    w = rng.uniform(size=n_samples)
    w[5:8] = 0
    regression.fit(x, y, sample_weight=w)
    regression.fit(x, y, sample_weight=w)

def test_fast_predict():
    if False:
        for i in range(10):
            print('nop')
    rng = np.random.RandomState(123)
    n_samples = 10 ** 3
    X_train = 20.0 * rng.rand(n_samples) - 10
    y_train = np.less(rng.rand(n_samples), expit(X_train)).astype('int64').astype('float64')
    weights = rng.rand(n_samples)
    weights[rng.rand(n_samples) < 0.1] = 0
    slow_model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    fast_model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    (X_train_fit, y_train_fit) = slow_model._build_y(X_train, y_train, sample_weight=weights, trim_duplicates=False)
    slow_model._build_f(X_train_fit, y_train_fit)
    fast_model.fit(X_train, y_train, sample_weight=weights)
    X_test = 20.0 * rng.rand(n_samples) - 10
    y_pred_slow = slow_model.predict(X_test)
    y_pred_fast = fast_model.predict(X_test)
    assert_array_equal(y_pred_slow, y_pred_fast)

def test_isotonic_copy_before_fit():
    if False:
        while True:
            i = 10
    ir = IsotonicRegression()
    copy.copy(ir)

def test_isotonic_dtype():
    if False:
        return 10
    y = [2, 1, 4, 3, 5]
    weights = np.array([0.9, 0.9, 0.9, 0.9, 0.9], dtype=np.float64)
    reg = IsotonicRegression()
    for dtype in (np.int32, np.int64, np.float32, np.float64):
        for sample_weight in (None, weights.astype(np.float32), weights):
            y_np = np.array(y, dtype=dtype)
            expected_dtype = check_array(y_np, dtype=[np.float64, np.float32], ensure_2d=False).dtype
            res = isotonic_regression(y_np, sample_weight=sample_weight)
            assert res.dtype == expected_dtype
            X = np.arange(len(y)).astype(dtype)
            reg.fit(X, y_np, sample_weight=sample_weight)
            res = reg.predict(X)
            assert res.dtype == expected_dtype

@pytest.mark.parametrize('y_dtype', [np.int32, np.int64, np.float32, np.float64])
def test_isotonic_mismatched_dtype(y_dtype):
    if False:
        for i in range(10):
            print('nop')
    reg = IsotonicRegression()
    y = np.array([2, 1, 4, 3, 5], dtype=y_dtype)
    X = np.arange(len(y), dtype=np.float32)
    reg.fit(X, y)
    assert reg.predict(X).dtype == X.dtype

def test_make_unique_dtype():
    if False:
        i = 10
        return i + 15
    x_list = [2, 2, 2, 3, 5]
    for dtype in (np.float32, np.float64):
        x = np.array(x_list, dtype=dtype)
        y = x.copy()
        w = np.ones_like(x)
        (x, y, w) = _make_unique(x, y, w)
        assert_array_equal(x, [2, 3, 5])

@pytest.mark.parametrize('dtype', [np.float64, np.float32])
def test_make_unique_tolerance(dtype):
    if False:
        while True:
            i = 10
    x = np.array([0, 1e-16, 1, 1 + 1e-14], dtype=dtype)
    y = x.copy()
    w = np.ones_like(x)
    (x, y, w) = _make_unique(x, y, w)
    if dtype == np.float64:
        x_out = np.array([0, 1, 1 + 1e-14])
    else:
        x_out = np.array([0, 1])
    assert_array_equal(x, x_out)

def test_isotonic_make_unique_tolerance():
    if False:
        i = 10
        return i + 15
    X = np.array([0, 1, 1 + 1e-16, 2], dtype=np.float64)
    y = np.array([0, 1, 2, 3], dtype=np.float64)
    ireg = IsotonicRegression().fit(X, y)
    y_pred = ireg.predict([0, 0.5, 1, 1.5, 2])
    assert_array_equal(y_pred, np.array([0, 0.75, 1.5, 2.25, 3]))
    assert_array_equal(ireg.X_thresholds_, np.array([0.0, 1.0, 2.0]))
    assert_array_equal(ireg.y_thresholds_, np.array([0.0, 1.5, 3.0]))

def test_isotonic_non_regression_inf_slope():
    if False:
        i = 10
        return i + 15
    X = np.array([0.0, 4.1e-320, 4.4e-314, 1.0])
    y = np.array([0.42, 0.42, 0.44, 0.44])
    ireg = IsotonicRegression().fit(X, y)
    y_pred = ireg.predict(np.array([0, 2.1e-319, 5.4e-316, 1e-10]))
    assert np.all(np.isfinite(y_pred))

@pytest.mark.parametrize('increasing', [True, False])
def test_isotonic_thresholds(increasing):
    if False:
        i = 10
        return i + 15
    rng = np.random.RandomState(42)
    n_samples = 30
    X = rng.normal(size=n_samples)
    y = rng.normal(size=n_samples)
    ireg = IsotonicRegression(increasing=increasing).fit(X, y)
    (X_thresholds, y_thresholds) = (ireg.X_thresholds_, ireg.y_thresholds_)
    assert X_thresholds.shape == y_thresholds.shape
    assert X_thresholds.shape[0] < X.shape[0]
    assert np.isin(X_thresholds, X).all()
    assert y_thresholds.max() <= y.max()
    assert y_thresholds.min() >= y.min()
    assert all(np.diff(X_thresholds) > 0)
    if increasing:
        assert all(np.diff(y_thresholds) >= 0)
    else:
        assert all(np.diff(y_thresholds) <= 0)

def test_input_shape_validation():
    if False:
        return 10
    X = np.arange(10)
    X_2d = X.reshape(-1, 1)
    y = np.arange(10)
    iso_reg = IsotonicRegression().fit(X, y)
    iso_reg_2d = IsotonicRegression().fit(X_2d, y)
    assert iso_reg.X_max_ == iso_reg_2d.X_max_
    assert iso_reg.X_min_ == iso_reg_2d.X_min_
    assert iso_reg.y_max == iso_reg_2d.y_max
    assert iso_reg.y_min == iso_reg_2d.y_min
    assert_array_equal(iso_reg.X_thresholds_, iso_reg_2d.X_thresholds_)
    assert_array_equal(iso_reg.y_thresholds_, iso_reg_2d.y_thresholds_)
    y_pred1 = iso_reg.predict(X)
    y_pred2 = iso_reg_2d.predict(X_2d)
    assert_allclose(y_pred1, y_pred2)

def test_isotonic_2darray_more_than_1_feature():
    if False:
        return 10
    X = np.arange(10)
    X_2d = np.c_[X, X]
    y = np.arange(10)
    msg = 'should be a 1d array or 2d array with 1 feature'
    with pytest.raises(ValueError, match=msg):
        IsotonicRegression().fit(X_2d, y)
    iso_reg = IsotonicRegression().fit(X, y)
    with pytest.raises(ValueError, match=msg):
        iso_reg.predict(X_2d)
    with pytest.raises(ValueError, match=msg):
        iso_reg.transform(X_2d)

def test_isotonic_regression_sample_weight_not_overwritten():
    if False:
        i = 10
        return i + 15
    'Check that calling fitting function of isotonic regression will not\n    overwrite `sample_weight`.\n    Non-regression test for:\n    https://github.com/scikit-learn/scikit-learn/issues/20508\n    '
    (X, y) = make_regression(n_samples=10, n_features=1, random_state=41)
    sample_weight_original = np.ones_like(y)
    sample_weight_original[0] = 10
    sample_weight_fit = sample_weight_original.copy()
    isotonic_regression(y, sample_weight=sample_weight_fit)
    assert_allclose(sample_weight_fit, sample_weight_original)
    IsotonicRegression().fit(X, y, sample_weight=sample_weight_fit)
    assert_allclose(sample_weight_fit, sample_weight_original)

@pytest.mark.parametrize('shape', ['1d', '2d'])
def test_get_feature_names_out(shape):
    if False:
        print('Hello World!')
    'Check `get_feature_names_out` for `IsotonicRegression`.'
    X = np.arange(10)
    if shape == '2d':
        X = X.reshape(-1, 1)
    y = np.arange(10)
    iso = IsotonicRegression().fit(X, y)
    names = iso.get_feature_names_out()
    assert isinstance(names, np.ndarray)
    assert names.dtype == object
    assert_array_equal(['isotonicregression0'], names)

def test_isotonic_regression_output_predict():
    if False:
        while True:
            i = 10
    'Check that `predict` does return the expected output type.\n\n    We need to check that `transform` will output a DataFrame and a NumPy array\n    when we set `transform_output` to `pandas`.\n\n    Non-regression test for:\n    https://github.com/scikit-learn/scikit-learn/issues/25499\n    '
    pd = pytest.importorskip('pandas')
    (X, y) = make_regression(n_samples=10, n_features=1, random_state=42)
    regressor = IsotonicRegression()
    with sklearn.config_context(transform_output='pandas'):
        regressor.fit(X, y)
        X_trans = regressor.transform(X)
        y_pred = regressor.predict(X)
    assert isinstance(X_trans, pd.DataFrame)
    assert isinstance(y_pred, np.ndarray)