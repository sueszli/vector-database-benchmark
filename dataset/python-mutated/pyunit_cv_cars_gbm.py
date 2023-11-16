from builtins import range
import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
import random
from h2o.estimators.gbm import H2OGradientBoostingEstimator

def cv_cars_gbm():
    if False:
        return 10
    cars = h2o.import_file(path=pyunit_utils.locate('smalldata/junit/cars_20mpg.csv'))
    problem = 1
    predictors = ['displacement', 'power', 'weight', 'acceleration', 'year']
    if problem == 1:
        response_col = 'economy_20mpg'
        distribution = 'bernoulli'
        cars[response_col] = cars[response_col].asfactor()
    elif problem == 2:
        response_col = 'cylinders'
        distribution = 'multinomial'
        cars[response_col] = cars[response_col].asfactor()
    else:
        response_col = 'economy'
        distribution = 'gaussian'
    print('Distribution: {0}'.format(distribution))
    print('Response column: {0}'.format(response_col))
    nfolds = random.randint(3, 10)
    gbm1 = H2OGradientBoostingEstimator(nfolds=nfolds, distribution=distribution, ntrees=5, fold_assignment='Modulo')
    gbm1.train(x=predictors, y=response_col, training_frame=cars)
    gbm2 = H2OGradientBoostingEstimator(nfolds=nfolds, distribution=distribution, ntrees=5, fold_assignment='Modulo')
    gbm2.train(x=predictors, y=response_col, training_frame=cars)
    pyunit_utils.check_models(gbm1, gbm2, True)
    nfolds = random.randint(3, 10)
    gbm1 = H2OGradientBoostingEstimator(nfolds=nfolds, distribution=distribution, ntrees=5, fold_assignment='Random')
    gbm1.train(x=predictors, y=response_col, training_frame=cars)
    gbm2 = H2OGradientBoostingEstimator(nfolds=nfolds, distribution=distribution, ntrees=5, fold_assignment='Random')
    gbm2.train(x=predictors, y=response_col, training_frame=cars)
    try:
        pyunit_utils.check_models(gbm1, gbm2, True)
        assert False, 'Expected models to be different over repeated Random runs'
    except AssertionError:
        assert True
    num_folds = random.randint(2, 5)
    fold_assignments = h2o.H2OFrame([[random.randint(0, num_folds - 1)] for f in range(cars.nrow)])
    fold_assignments.set_names(['fold_assignments'])
    cars = cars.cbind(fold_assignments)
    gbm = H2OGradientBoostingEstimator(distribution=distribution, ntrees=5, keep_cross_validation_models=True, keep_cross_validation_predictions=True)
    gbm.train(x=predictors, y=response_col, training_frame=cars, fold_column='fold_assignments')
    num_cv_models = len(gbm._model_json['output']['cross_validation_models'])
    assert num_cv_models == num_folds, 'Expected {0} cross-validation models, but got {1}'.format(num_folds, num_cv_models)
    cv_model1 = h2o.get_model(gbm._model_json['output']['cross_validation_models'][0]['name'])
    cv_model2 = h2o.get_model(gbm._model_json['output']['cross_validation_models'][1]['name'])
    cv_predictions = gbm1._model_json['output']['cross_validation_predictions']
    assert cv_predictions is None, 'Expected cross-validation predictions to be None, but got {0}'.format(cv_predictions)
    cv_predictions = gbm._model_json['output']['cross_validation_predictions']
    assert len(cv_predictions) == num_folds, 'Expected the same number of cross-validation predictions as folds, but got {0}'.format(len(cv_predictions))
    gbm = H2OGradientBoostingEstimator(nfolds=cars.nrow, distribution=distribution, ntrees=5, fold_assignment='Modulo')
    gbm.train(x=predictors, y=response_col, training_frame=cars)
    gbm1 = H2OGradientBoostingEstimator(nfolds=0, distribution=distribution, ntrees=5)
    gbm1.train(x=predictors, y=response_col, training_frame=cars)
    gbm2 = H2OGradientBoostingEstimator(distribution=distribution, ntrees=5)
    gbm2.train(x=predictors, y=response_col, training_frame=cars)
    pyunit_utils.check_models(gbm1, gbm2)
    gbm = H2OGradientBoostingEstimator(nfolds=random.randint(3, 10), ntrees=5, distribution=distribution)
    gbm.train(x=predictors, y=response_col, training_frame=cars, validation_frame=cars)
    try:
        gbm = H2OGradientBoostingEstimator(nfolds=random.sample([-1, 1], 1)[0], ntrees=5, distribution=distribution)
        gbm.train(x=predictors, y=response_col, training_frame=cars)
        assert False, 'Expected model-build to fail when nfolds is 1 or < 0'
    except EnvironmentError:
        assert True
    try:
        gbm = H2OGradientBoostingEstimator(nfolds=cars.nrow + 1, distribution=distribution, ntrees=5, fold_assignment='Modulo')
        gbm.train(x=predictors, y=response_col, training_frame=cars)
        assert False, 'Expected model-build to fail when nfolds > nobs'
    except EnvironmentError:
        assert True
    try:
        gbm = H2OGradientBoostingEstimator(nfolds=3, ntrees=5, distribution=distribution)
        gbm.train(x=predictors, y=response_col, training_frame=cars, fold_column='fold_assignments')
        assert False, 'Expected model-build to fail when fold_column and nfolds both specified'
    except EnvironmentError:
        assert True
    try:
        gbm = H2OGradientBoostingEstimator(ntrees=5, fold_assignment='Random', distribution=distribution)
        gbm.train(x=predictors, y=response_col, training_frame=cars, fold_column='fold_assignments')
        assert False, 'Expected model-build to fail when fold_column and fold_assignment both specified'
    except EnvironmentError:
        assert True
if __name__ == '__main__':
    pyunit_utils.standalone_test(cv_cars_gbm)
else:
    cv_cars_gbm()