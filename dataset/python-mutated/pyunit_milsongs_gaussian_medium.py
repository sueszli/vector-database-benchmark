from h2o.estimators.xgboost import *
from tests import pyunit_utils

def xgboost_milsongs_gaussian_medium():
    if False:
        for i in range(10):
            print('nop')
    assert H2OXGBoostEstimator.available()
    training_frame = h2o.import_file(pyunit_utils.locate('bigdata/laptop/milsongs/milsongs-train.csv.gz'))
    test_frame = h2o.import_file(pyunit_utils.locate('bigdata/laptop/milsongs/milsongs-test.csv.gz'))
    x = list(range(1, training_frame.ncol))
    y = 0
    model_2_trees = H2OXGBoostEstimator(training_frame=training_frame, learn_rate=0.3, booster='gbtree', seed=1, ntrees=2, distribution='gaussian')
    model_2_trees.train(x=x, y=y, training_frame=training_frame)
    prediction_2_trees = model_2_trees.predict(test_frame)
    assert prediction_2_trees.nrows == test_frame.nrows
    model_10_trees = H2OXGBoostEstimator(training_frame=training_frame, learn_rate=0.3, booster='gbtree', seed=1, ntrees=10, distribution='gaussian')
    model_10_trees.train(x=x, y=y, training_frame=training_frame)
    prediction_10_trees = model_10_trees.predict(test_frame)
    assert prediction_10_trees.nrows == test_frame.nrows
    assert model_2_trees.mse() > model_10_trees.mse()
if __name__ == '__main__':
    pyunit_utils.standalone_test(xgboost_milsongs_gaussian_medium)
else:
    xgboost_milsongs_gaussian_medium()