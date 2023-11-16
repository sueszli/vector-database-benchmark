import h2o
import sys
sys.path.insert(1, '../../../')
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
from tests import pyunit_utils

def stackedensemble_metalearner_seed_test():
    if False:
        while True:
            i = 10
    train = h2o.import_file(path=pyunit_utils.locate('smalldata/testng/higgs_train_5k.csv'), destination_frame='higgs_train_5k')
    test = h2o.import_file(path=pyunit_utils.locate('smalldata/testng/higgs_test_5k.csv'), destination_frame='higgs_test_5k')
    x = train.columns
    y = 'response'
    x.remove(y)
    train[y] = train[y].asfactor()
    test[y] = test[y].asfactor()
    nfolds = 3
    gbm_params = {'sample_rate': 0.3, 'col_sample_rate': 0.3}
    my_gbm = H2OGradientBoostingEstimator(distribution='bernoulli', ntrees=10, nfolds=nfolds, keep_cross_validation_predictions=True, seed=1)
    my_gbm.train(x=x, y=y, training_frame=train)
    my_rf = H2ORandomForestEstimator(ntrees=10, nfolds=nfolds, keep_cross_validation_predictions=True, seed=1)
    my_rf.train(x=x, y=y, training_frame=train)
    stack_gbm1 = H2OStackedEnsembleEstimator(base_models=[my_gbm, my_rf], metalearner_algorithm='gbm', metalearner_params=gbm_params, seed=55555)
    stack_gbm2 = H2OStackedEnsembleEstimator(base_models=[my_gbm, my_rf], metalearner_algorithm='gbm', metalearner_params=gbm_params, seed=55555)
    stack_gbm1.train(x=x, y=y, training_frame=train)
    stack_gbm2.train(x=x, y=y, training_frame=train)
    meta_gbm1 = h2o.get_model(stack_gbm1.metalearner()['name'])
    meta_gbm2 = h2o.get_model(stack_gbm2.metalearner()['name'])
    assert meta_gbm1.rmse(train=True) == meta_gbm2.rmse(train=True), 'RMSE should match if same seed'
    stack_gbm3 = H2OStackedEnsembleEstimator(base_models=[my_gbm, my_rf], metalearner_algorithm='gbm', metalearner_params=gbm_params, seed=55555)
    stack_gbm4 = H2OStackedEnsembleEstimator(base_models=[my_gbm, my_rf], metalearner_algorithm='gbm', metalearner_params=gbm_params, seed=98765)
    stack_gbm3.train(x=x, y=y, training_frame=train)
    stack_gbm4.train(x=x, y=y, training_frame=train)
    meta_gbm3 = h2o.get_model(stack_gbm3.metalearner()['name'])
    meta_gbm4 = h2o.get_model(stack_gbm4.metalearner()['name'])
    assert meta_gbm3.rmse(train=True) != meta_gbm4.rmse(train=True), 'RMSE should NOT match if diff seed'
pyunit_utils.run_tests([stackedensemble_metalearner_seed_test])