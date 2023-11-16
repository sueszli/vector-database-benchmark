import sys

sys.path.insert(1, "../../../")
import h2o
from tests import pyunit_utils
from tests.pyunit_utils import CustomOneFuncStr, \
    assert_all_metrics_equal, regression_model
from h2o.estimators.gbm import H2OGradientBoostingEstimator


def test_custom_metric_from_str():
    custom_metric = h2o.upload_custom_metric(CustomOneFuncStr, class_name="CustomOneFunc", func_name="custom_mm")
    (model2, f_test2) = regression_model(H2OGradientBoostingEstimator, custom_metric)
    assert_all_metrics_equal(model2, f_test2, "custom_mm", 1)


__TESTS__ = [
    test_custom_metric_from_str
]

if __name__ == "__main__":
    for func in __TESTS__:
        pyunit_utils.standalone_test(func)
else:
    for func in __TESTS__:
        func()
