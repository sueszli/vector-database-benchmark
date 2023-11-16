from builtins import range
import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.random_forest import H2ORandomForestEstimator
import random

def cv_carsRF():
    if False:
        return 10
    cars = h2o.import_file(path=pyunit_utils.locate('smalldata/junit/cars_20mpg.csv'))
    problem = random.sample(list(range(3)), 1)[0]
    problem = 2
    predictors = ['displacement', 'power', 'weight', 'acceleration', 'year']
    if problem == 1:
        response_col = 'economy_20mpg'
        cars[response_col] = cars[response_col].asfactor()
    elif problem == 2:
        response_col = 'cylinders'
        cars[response_col] = cars[response_col].asfactor()
    else:
        response_col = 'economy'
    print('Response column: {0}'.format(response_col))
    nfolds = random.randint(3, 10)
    rf1 = H2ORandomForestEstimator(nfolds=nfolds, fold_assignment='Modulo', seed=1234)
    rf1.train(x=predictors, y=response_col, training_frame=cars)
    rf2 = H2ORandomForestEstimator(nfolds=nfolds, fold_assignment='Modulo', seed=1234)
    rf2.train(x=predictors, y=response_col, training_frame=cars)
    pyunit_utils.check_models(rf1, rf2, True)
    nfolds = random.randint(3, 10)
    rf1 = H2ORandomForestEstimator(nfolds=nfolds, fold_assignment='Random')
    rf1.train(x=predictors, y=response_col, training_frame=cars)
    rf2 = H2ORandomForestEstimator(nfolds=nfolds, fold_assignment='Random')
    rf2.train(x=predictors, y=response_col, training_frame=cars)
    try:
        pyunit_utils.check_models(rf1, rf2, True)
        assert False, 'Expected models to be different over repeated Random runs'
    except AssertionError:
        assert True
    num_folds = random.randint(2, 5)
    fold_assignments = h2o.H2OFrame([[random.randint(0, num_folds - 1)] for f in range(cars.nrow)])
    fold_assignments.set_names(['fold_assignments'])
    cars = cars.cbind(fold_assignments)
    rf = H2ORandomForestEstimator(keep_cross_validation_models=True, keep_cross_validation_predictions=True)
    rf.train(y=response_col, x=predictors, training_frame=cars, fold_column='fold_assignments')
    num_cv_models = len(rf._model_json['output']['cross_validation_models'])
    assert num_cv_models == num_folds, 'Expected {0} cross-validation models, but got {1}'.format(num_folds, num_cv_models)
    cv_model1 = h2o.get_model(rf._model_json['output']['cross_validation_models'][0]['name'])
    cv_model2 = h2o.get_model(rf._model_json['output']['cross_validation_models'][1]['name'])
    cv_predictions = rf1._model_json['output']['cross_validation_predictions']
    assert cv_predictions is None, 'Expected cross-validation predictions to be None, but got {0}'.format(cv_predictions)
    cv_predictions = rf._model_json['output']['cross_validation_predictions']
    assert len(cv_predictions) == num_folds, 'Expected the same number of cross-validation predictions as folds, but got {0}'.format(len(cv_predictions))
    rf = H2ORandomForestEstimator(nfolds=cars.nrow, fold_assignment='Modulo')
    rf.train(y=response_col, x=predictors, training_frame=cars)
    rf1 = H2ORandomForestEstimator(nfolds=0, seed=1234)
    rf1.train(y=response_col, x=predictors, training_frame=cars)
    rf2 = H2ORandomForestEstimator(seed=1234)
    rf2.train(y=response_col, x=predictors, training_frame=cars)
    pyunit_utils.check_models(rf1, rf2)
    rf = H2ORandomForestEstimator(nfolds=random.randint(3, 10))
    rf.train(y=response_col, x=predictors, training_frame=cars, validation_frame=cars)
    try:
        rf = H2ORandomForestEstimator(nfolds=random.sample([-1, 1], 1)[0])
        rf.train(y=response_col, x=predictors, training_frame=cars)
        assert False, 'Expected model-build to fail when nfolds is 1 or < 0'
    except EnvironmentError:
        assert True
    try:
        rf = H2ORandomForestEstimator(nfolds=cars.nrow + 1, fold_assignment='Modulo')
        rf.train(y=response_col, x=predictors, training_frame=cars)
        assert False, 'Expected model-build to fail when nfolds > nobs'
    except EnvironmentError:
        assert True
    try:
        rf = H2ORandomForestEstimator(nfolds=3)
        rf.train(y=response_col, x=predictors, fold_column='fold_assignments', training_frame=cars)
        assert False, 'Expected model-build to fail when fold_column and nfolds both specified'
    except EnvironmentError:
        assert True
if __name__ == '__main__':
    pyunit_utils.standalone_test(cv_carsRF)
else:
    cv_carsRF()