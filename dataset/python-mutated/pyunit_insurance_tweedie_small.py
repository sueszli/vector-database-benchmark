from h2o.estimators.xgboost import *
from tests import pyunit_utils

def xgboost_insurance_tweedie_small():
    if False:
        return 10
    assert H2OXGBoostEstimator.available()
    training_frame = h2o.import_file(pyunit_utils.locate('smalldata/testng/insurance_train1.csv'))
    training_frame['offset'] = training_frame['Holders'].log()
    test_frame = h2o.import_file(pyunit_utils.locate('smalldata/testng/insurance_validation1.csv'))
    x = ['District', 'Group', 'Age', 'Holders']
    y = 'Claims'
    model_2_trees = H2OXGBoostEstimator(training_frame=training_frame, learn_rate=0.7, booster='gbtree', seed=1, ntrees=2, distribution='tweedie')
    model_2_trees.train(x=x, y=y, training_frame=training_frame)
    prediction_2_trees = model_2_trees.predict(test_frame)
    assert prediction_2_trees.nrows == test_frame.nrows
    model_10_trees = H2OXGBoostEstimator(training_frame=training_frame, learn_rate=0.7, booster='gbtree', seed=1, ntrees=10, distribution='tweedie')
    model_10_trees.train(x=x, y=y, training_frame=training_frame)
    prediction_10_trees = model_10_trees.predict(test_frame)
    assert prediction_10_trees.nrows == test_frame.nrows
    assert model_2_trees.mse() > model_10_trees.mse()
if __name__ == '__main__':
    pyunit_utils.standalone_test(xgboost_insurance_tweedie_small)
else:
    xgboost_insurance_tweedie_small()