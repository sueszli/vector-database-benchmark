from builtins import range
import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
import random
from h2o.estimators.glm import H2OGeneralizedLinearEstimator

def cv_cars_glm():
    if False:
        while True:
            i = 10
    cars = h2o.import_file(path=pyunit_utils.locate('smalldata/junit/cars_20mpg.csv'))
    problem = random.sample(list(range(3)), 1)[0]
    predictors = ['displacement', 'power', 'weight', 'acceleration', 'year']
    if problem == 1:
        response_col = 'economy_20mpg'
        family = 'binomial'
        cars[response_col] = cars[response_col].asfactor()
    elif problem == 2:
        family = 'poisson'
        response_col = 'cylinders'
    else:
        family = 'gaussian'
        response_col = 'economy'
    print('Distribution: {0}'.format(family))
    print('Response column: {0}'.format(response_col))
    nfolds = random.randint(3, 10)
    glm1 = H2OGeneralizedLinearEstimator(nfolds=nfolds, family=family, fold_assignment='Modulo')
    glm1.train(x=predictors, y=response_col, training_frame=cars)
    glm2 = H2OGeneralizedLinearEstimator(nfolds=nfolds, family=family, fold_assignment='Modulo')
    glm2.train(x=predictors, y=response_col, training_frame=cars)
    pyunit_utils.check_models(glm1, glm2, True)
    nfolds = random.randint(3, 10)
    glm1 = H2OGeneralizedLinearEstimator(nfolds=nfolds, family=family, fold_assignment='Random')
    glm1.train(x=predictors, y=response_col, training_frame=cars)
    glm2 = H2OGeneralizedLinearEstimator(nfolds=nfolds, family=family, fold_assignment='Random')
    glm2.train(x=predictors, y=response_col, training_frame=cars)
    try:
        pyunit_utils.check_models(glm1, glm2, True)
        assert False, 'Expected models to be different over repeated Random runs'
    except AssertionError:
        assert True
    num_folds = random.randint(2, 5)
    fold_assignments = h2o.H2OFrame([[random.randint(0, num_folds - 1)] for f in range(cars.nrow)])
    fold_assignments.set_names(['fold_assignments'])
    cars = cars.cbind(fold_assignments)
    glm = H2OGeneralizedLinearEstimator(family=family, keep_cross_validation_models=True, keep_cross_validation_predictions=True)
    glm.train(x=predictors, y=response_col, training_frame=cars, fold_column='fold_assignments')
    num_cv_models = len(glm._model_json['output']['cross_validation_models'])
    assert num_cv_models == num_folds, 'Expected {0} cross-validation models, but got {1}'.format(num_folds, num_cv_models)
    cv_model1 = h2o.get_model(glm._model_json['output']['cross_validation_models'][0]['name'])
    cv_model2 = h2o.get_model(glm._model_json['output']['cross_validation_models'][1]['name'])
    cv_predictions = glm1._model_json['output']['cross_validation_predictions']
    assert cv_predictions is None, 'Expected cross-validation predictions to be None, but got {0}'.format(cv_predictions)
    cv_predictions = glm._model_json['output']['cross_validation_predictions']
    assert len(cv_predictions) == num_folds, 'Expected the same number of cross-validation predictions as folds, but got {0}'.format(len(cv_predictions))
    glm1 = H2OGeneralizedLinearEstimator(nfolds=0, family=family)
    glm1.train(x=predictors, y=response_col, training_frame=cars)
    glm2 = H2OGeneralizedLinearEstimator(family=family)
    glm2.train(x=predictors, y=response_col, training_frame=cars)
    pyunit_utils.check_models(glm1, glm2)
    glm = H2OGeneralizedLinearEstimator(nfolds=random.randint(3, 10), family=family)
    glm.train(x=predictors, y=response_col, training_frame=cars, validation_frame=cars)
    try:
        glm = H2OGeneralizedLinearEstimator(nfolds=random.sample([-1, 1], 1)[0], family=family)
        glm.train(x=predictors, y=response_col, training_frame=cars)
        assert False, 'Expected model-build to fail when nfolds is 1 or < 0'
    except EnvironmentError:
        assert True
    try:
        glm = H2OGeneralizedLinearEstimator(nfolds=cars.nrow + 1, family=family, fold_assignment='Modulo')
        glm.train(x=predictors, y=response_col, training_frame=cars)
        assert False, 'Expected model-build to fail when nfolds > nobs'
    except EnvironmentError:
        assert True
    try:
        glm = H2OGeneralizedLinearEstimator(nfolds=3, family=family)
        glm.train(x=predictors, y=response_col, training_frame=cars, fold_column='fold_assignments')
        assert False, 'Expected model-build to fail when fold_column and nfolds both specified'
    except EnvironmentError:
        assert True
if __name__ == '__main__':
    pyunit_utils.standalone_test(cv_cars_glm)
else:
    cv_cars_glm()