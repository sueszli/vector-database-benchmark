import h2o
import sys
sys.path.insert(1, '../../../')
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
from tests import pyunit_utils

def stackedensemble_grid_gaussian():
    if False:
        i = 10
        return i + 15
    'This test check the following (for guassian regression):\n    1) That H2OStackedEnsembleEstimator executes w/o erros on a random-grid-based ensemble.\n    2) That .predict() works on a stack.\n    3) That .model_performance() works on a stack.\n    4) That the training and test performance is better on ensemble vs the base learners.\n    5) That the validation_frame arg on H2OStackedEnsembleEstimator works correctly.\n    '
    dat = h2o.import_file(path=pyunit_utils.locate('smalldata/extdata/australia.csv'), destination_frame='australia_hex')
    (train, test) = dat.split_frame(ratios=[0.75], seed=1)
    print(train.summary())
    x = ['premax', 'salmax', 'minairtemp', 'maxairtemp', 'maxsst', 'maxsoilmoist', 'Max_czcs']
    y = 'runoffnew'
    nfolds = 5
    hyper_params = {'learn_rate': [0.01, 0.03], 'max_depth': [3, 4, 5, 6, 9], 'sample_rate': [0.7, 0.8, 0.9, 1.0], 'col_sample_rate': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}
    search_criteria = {'strategy': 'RandomDiscrete', 'max_models': 3, 'seed': 1}
    grid = H2OGridSearch(model=H2OGradientBoostingEstimator(ntrees=10, seed=1, nfolds=nfolds, fold_assignment='Modulo', keep_cross_validation_predictions=True), hyper_params=hyper_params, search_criteria=search_criteria, grid_id='gbm_grid_guassian')
    grid.train(x=x, y=y, training_frame=train)
    stack = H2OStackedEnsembleEstimator(model_id='my_ensemble_gbm_grid_guassian', base_models=grid.model_ids)
    stack.train(x=x, y=y, training_frame=train, validation_frame=test)
    pred = stack.predict(test_data=test)
    assert pred.nrow == test.nrow, 'expected ' + str(pred.nrow) + ' to be equal to ' + str(test.nrow)
    assert pred.ncol == 1, 'expected ' + str(pred.ncol) + ' to be equal to 1 but it was equal to ' + str(pred.ncol)
    perf_stack_train = stack.model_performance()
    perf_stack_test = stack.model_performance(test_data=test)
    baselearner_best_rmse_train = max([h2o.get_model(model).rmse(train=True) for model in grid.model_ids])
    stack_rmse_train = perf_stack_train.rmse()
    print('Best Base-learner Training RMSE:  {0}'.format(baselearner_best_rmse_train))
    print('Ensemble Training RMSE:  {0}'.format(stack_rmse_train))
    assert stack_rmse_train < baselearner_best_rmse_train, "expected stack_rmse_train would be less than  found it wasn't baselearner_best_rmse_train"
    baselearner_best_rmse_test = max([h2o.get_model(model).model_performance(test_data=test).rmse() for model in grid.model_ids])
    stack_rmse_test = perf_stack_test.rmse()
    print('Best Base-learner Test RMSE:  {0}'.format(baselearner_best_rmse_test))
    print('Ensemble Test RMSE:  {0}'.format(stack_rmse_test))
    assert stack_rmse_test < baselearner_best_rmse_test, "expected stack_rmse_test would be less than baselearner_best_rmse_test, found it wasn't baselearner_best_rmse_test = " + str(baselearner_best_rmse_test) + ',stack_rmse_test = ' + str(stack_rmse_test)
    perf_stack_validation_frame = stack.model_performance(valid=True)
    assert stack_rmse_test == perf_stack_validation_frame.rmse(), 'expected stack_rmse_test to be the same as perf_stack_validation_frame.rmse() found they were not perf_stack_validation_frame.rmse() = ' + str(perf_stack_validation_frame.rmse()) + 'stack_rmse_test was ' + str(stack_rmse_test)
if __name__ == '__main__':
    pyunit_utils.standalone_test(stackedensemble_grid_gaussian)
else:
    stackedensemble_grid_gaussian()