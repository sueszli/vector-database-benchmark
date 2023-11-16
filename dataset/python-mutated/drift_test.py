"""Test drift utils"""
import numpy as np
from hamcrest import assert_that, calling, close_to, equal_to, raises
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.utils.distribution.drift import cramers_v, earth_movers_distance, kolmogorov_smirnov

def test_emd():
    if False:
        for i in range(10):
            print('nop')
    dist1 = np.ones(100)
    dist2 = np.zeros(100)
    res = earth_movers_distance(dist1=dist1, dist2=dist2, margin_quantile_filter=0)
    assert_that(res, equal_to(1))

def test_real_input():
    if False:
        return 10
    dist1 = np.array(range(100))
    dist2 = np.array(range(50, 150))
    res = earth_movers_distance(dist1=dist1, dist2=dist2, margin_quantile_filter=0)
    assert_that(res, close_to(0.33, 0.01))

def test_emd_scaling():
    if False:
        while True:
            i = 10
    dist1 = np.ones(100) * 10
    dist2 = np.zeros(100)
    res = earth_movers_distance(dist1=dist1, dist2=dist2, margin_quantile_filter=0)
    assert_that(res, equal_to(1))

def test_emd_margin_filter():
    if False:
        while True:
            i = 10
    dist1 = np.concatenate([np.ones(99) * 10, np.ones(1) * 100])
    dist2 = np.concatenate([np.zeros(99), np.ones(1)])
    res = earth_movers_distance(dist1=dist1, dist2=dist2, margin_quantile_filter=0.01)
    assert_that(res, equal_to(1))

def test_emd_raises_exception():
    if False:
        while True:
            i = 10
    dist1 = np.ones(100)
    dist2 = np.zeros(100)
    assert_that(calling(earth_movers_distance).with_args(dist1, dist2, -1), raises(DeepchecksValueError, 'margin_quantile_filter expected a value in range \\[0, 0.5\\), instead got -1'))

def test_cramers_v_sampling():
    if False:
        return 10
    dist1 = np.array(['a'] * 2000 + ['b'] * 8000)
    dist2 = np.array(['a'] * 4000 + ['b'] * 6000)
    res = cramers_v(dist1=dist1, dist2=dist2)
    dist2 = np.array(['a'] * 400 + ['b'] * 600)
    res_sampled = cramers_v(dist1=dist1, dist2=dist2)
    dist1 = np.array(['a'] * 200 + ['b'] * 800)
    res_double_sampled = cramers_v(dist1=dist1, dist2=dist2)
    assert_that(res, close_to(res_sampled, 0.01))
    assert_that(res_sampled, close_to(res_double_sampled, 0.0001))

def test_cramers_v():
    if False:
        print('Hello World!')
    dist1 = np.array(['a'] * 200 + ['b'] * 800)
    dist2 = np.array(['a'] * 400 + ['b'] * 600)
    res = cramers_v(dist1=dist1, dist2=dist2)
    assert_that(res, close_to(0.21, 0.01))

def test_cramers_v_from_freqs():
    if False:
        for i in range(10):
            print('nop')
    dist1 = np.array([200, 800])
    dist2 = np.array([400, 600])
    res = cramers_v(dist1=dist1, dist2=dist2, from_freqs=True)
    assert_that(res, close_to(0.21, 0.01))

def test_cramers_v_completely_diff_columns():
    if False:
        print('Hello World!')
    dist1 = np.array(['a'] * 1000)
    dist2 = np.array(['b'] * 1000)
    res = cramers_v(dist1=dist1, dist2=dist2)
    assert_that(res, close_to(1, 0.01))

def test_cramers_v_single_value_columns():
    if False:
        print('Hello World!')
    dist1 = np.array(['a'] * 1000)
    dist2 = np.array(['a'] * 1000)
    res = cramers_v(dist1=dist1, dist2=dist2)
    assert_that(res, equal_to(0))

def test_cramers_v_with_nones():
    if False:
        print('Hello World!')
    dist1 = np.array(['a'] * 200 + ['b'] * 800 + [None] * 100)
    dist2 = np.array(['a'] * 400 + ['b'] * 600)
    res = cramers_v(dist1=dist1, dist2=dist2)
    assert_that(res, close_to(0.3, 0.01))

def test_cramers_v_min_category_ratio():
    if False:
        return 10
    dist1 = np.array(['a'] * 200 + ['b'] * 800 + ['c'] * 10 + ['d'] * 10)
    dist2 = np.array(['a'] * 400 + ['b'] * 620)
    res = cramers_v(dist1=dist1, dist2=dist2, min_category_size_ratio=0)
    assert_that(res, close_to(0.228, 0.01))
    res_min_cat_ratio = cramers_v(dist1=dist1, dist2=dist2, min_category_size_ratio=0.1)
    assert_that(res_min_cat_ratio, close_to(0.208, 0.01))

def test_cramers_v_imbalanced_big_goes_to_0():
    if False:
        for i in range(10):
            print('nop')
    dist1 = np.array([0] * 9900 + [1] * 100)
    dist2 = np.array([0] * 10000)
    dist2_small = np.array([0] * 1000)
    dist2_very_small = np.array([0] * 100)
    res = cramers_v(dist1=dist1, dist2=dist2, balance_classes=True)
    res_small = cramers_v(dist1=dist1, dist2=dist2_small, balance_classes=True)
    res_very_small = cramers_v(dist1=dist1, dist2=dist2_very_small, balance_classes=True)
    assert_that(res, close_to(0.56, 0.01))
    assert_that(res_small, close_to(0.45, 0.01))
    assert_that(res_very_small, close_to(0.14, 0.01))

def test_cramers_v_imbalanced_medium_goes_to_0():
    if False:
        print('Hello World!')
    dist1 = np.array([0] * 99900 + [1] * 100)
    dist2 = np.array([0] * 10000)
    dist2_small = np.array([0] * 1000)
    res = cramers_v(dist1=dist1, dist2=dist2, balance_classes=True)
    res_small = cramers_v(dist1=dist1, dist2=dist2_small, balance_classes=True)
    assert_that(res, close_to(0.45, 0.01))
    assert_that(res_small, close_to(0.16, 0.01))

def test_cramers_v_imbalanced_very_small_goes_to_0():
    if False:
        print('Hello World!')
    dist1 = np.array([0] * 9999900 + [1] * 100)
    dist2 = np.array([0] * 10000)
    dist2_small = np.array([0] * 1000)
    res = cramers_v(dist1=dist1, dist2=dist2, balance_classes=True)
    res_small = cramers_v(dist1=dist1, dist2=dist2_small, balance_classes=True)
    assert_that(res, close_to(0.02, 0.01))
    assert_that(res_small, close_to(0.0, 0.01))

def test_cramers_v_imbalanced_medium_goes_to_big():
    if False:
        i = 10
        return i + 15
    dist1 = np.array([0] * 99900 + [1] * 100)
    dist2 = np.array([0] * 99000 + [1] * 1000)
    dist2_small = np.array([0] * 990 + [1] * 10)
    res = cramers_v(dist1=dist1, dist2=dist2, balance_classes=True)
    res_small = cramers_v(dist1=dist1, dist2=dist2_small, balance_classes=True)
    assert_that(res, close_to(0.45, 0.01))
    assert_that(res_small, close_to(0.36, 0.01))

def test_cramers_v_imbalanced_big_goes_to_medium():
    if False:
        return 10
    dist1 = np.array([0] * 99000 + [1] * 1000)
    dist2 = np.array([0] * 99900 + [1] * 100)
    dist2_small = np.array([0] * 999 + [1] * 1)
    res = cramers_v(dist1=dist1, dist2=dist2, balance_classes=True)
    res_small = cramers_v(dist1=dist1, dist2=dist2_small, balance_classes=True)
    assert_that(res, close_to(0.45, 0.01))
    assert_that(res_small, close_to(0.37, 0.01))

def test_cramers_v_imbalanced_three_classes():
    if False:
        for i in range(10):
            print('nop')
    dist1 = np.array([0] * 4900 + [1] * 100 + [2] * 5000)
    dist2 = np.array([0] * 4950 + [1] * 50 + [2] * 5000)
    res = cramers_v(dist1=dist1, dist2=dist2, balance_classes=True)
    assert_that(res, close_to(0.15, 0.01))

def test_cramers_v_imbalanced():
    if False:
        return 10
    dist1 = np.array([0] * 9900 + [1] * 100)
    dist2 = np.array([0] * 9950 + [1] * 50)
    res = cramers_v(dist1=dist1, dist2=dist2, balance_classes=True)
    assert_that(res, close_to(0.17, 0.01))

def test_cramers_v_imbalanced_ignore_min_category_size():
    if False:
        for i in range(10):
            print('nop')
    dist1 = np.array([0] * 9900 + [1] * 100)
    dist2 = np.array([0] * 9950 + [1] * 50)
    res = cramers_v(dist1=dist1, dist2=dist2, balance_classes=True, min_category_size_ratio=0.1)
    assert_that(res, close_to(0.17, 0.01))

def test_ks_no_drift():
    if False:
        for i in range(10):
            print('nop')
    dist1 = np.zeros(100)
    dist2 = np.zeros(100)
    res = kolmogorov_smirnov(dist1=dist1, dist2=dist2)
    assert_that(res, equal_to(0))

def test_ks_max_drift():
    if False:
        while True:
            i = 10
    dist1 = np.ones(100)
    dist2 = np.zeros(100)
    res = kolmogorov_smirnov(dist1=dist1, dist2=dist2)
    assert_that(res, equal_to(1))

def test_ks_regular_drift():
    if False:
        for i in range(10):
            print('nop')
    np.random.seed(42)
    dist1 = np.random.normal(0, 1, 10000)
    dist2 = np.random.normal(1, 1, 10000)
    res = kolmogorov_smirnov(dist1=dist1, dist2=dist2)
    assert_that(res, close_to(0.382, 0.01))

def test_ks_regular_drift_scaled():
    if False:
        while True:
            i = 10
    dist1 = np.random.normal(0, 1, 10000) * 100
    dist2 = np.random.normal(1, 1, 10000) * 100
    res = kolmogorov_smirnov(dist1=dist1, dist2=dist2)
    assert_that(res, close_to(0.382, 0.01))