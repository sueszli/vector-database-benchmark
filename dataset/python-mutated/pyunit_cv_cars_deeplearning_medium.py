from builtins import range
import sys, os
sys.path.insert(1, os.path.join('..', '..', '..'))
import h2o
from tests import pyunit_utils
import random
from h2o.estimators.deeplearning import H2ODeepLearningEstimator

def cv_cars_dl():
    if False:
        return 10
    cars = h2o.import_file(path=pyunit_utils.locate('smalldata/junit/cars_20mpg.csv'))
    problem = random.sample(list(range(3)), 1)[0]
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
    dl = H2ODeepLearningEstimator(nfolds=random.randint(3, 10), fold_assignment='Modulo', hidden=[20, 20], epochs=10)
    dl.train(x=predictors, y=response_col, training_frame=cars)
    nfolds = random.randint(3, 10)
    dl1 = H2ODeepLearningEstimator(nfolds=nfolds, fold_assignment='Random', hidden=[20, 20], epochs=10)
    dl1.train(x=predictors, y=response_col, training_frame=cars)
    dl2 = H2ODeepLearningEstimator(nfolds=nfolds, fold_assignment='Random', hidden=[20, 20], epochs=10)
    dl2.train(x=predictors, y=response_col, training_frame=cars)
    try:
        pyunit_utils.check_models(dl1, dl2, True)
        assert False, 'Expected models to be different over repeated Random runs'
    except AssertionError:
        assert True
    num_folds = random.randint(2, 5)
    fold_assignments = h2o.H2OFrame([[random.randint(0, num_folds - 1)] for _ in range(cars.nrow)])
    fold_assignments.set_names(['fold_assignments'])
    cars = cars.cbind(fold_assignments)
    dl = H2ODeepLearningEstimator(keep_cross_validation_models=True, keep_cross_validation_predictions=True, hidden=[20, 20], epochs=10)
    dl.train(x=predictors, y=response_col, training_frame=cars, fold_column='fold_assignments')
    num_cv_models = len(dl._model_json['output']['cross_validation_models'])
    assert num_cv_models == num_folds, 'Expected {0} cross-validation models, but got {1}'.format(num_folds, num_cv_models)
    cv_model1 = h2o.get_model(dl._model_json['output']['cross_validation_models'][0]['name'])
    cv_model2 = h2o.get_model(dl._model_json['output']['cross_validation_models'][1]['name'])
    cv_predictions = dl1._model_json['output']['cross_validation_predictions']
    dl = H2ODeepLearningEstimator(nfolds=cars.nrow, fold_assignment='Modulo', hidden=[20, 20], epochs=10)
    dl.train(x=predictors, y=response_col, training_frame=cars)
    dl = H2ODeepLearningEstimator(nfolds=0, hidden=[20, 20], epochs=10)
    dl.train(x=predictors, y=response_col, training_frame=cars)
    dl = H2ODeepLearningEstimator(nfolds=random.randint(3, 10), hidden=[20, 20], epochs=10)
    dl.train(x=predictors, y=response_col, training_frame=cars, validation_frame=cars)
    try:
        dl = H2ODeepLearningEstimator(nfolds=random.sample([-1, 1], 1)[0], hidden=[20, 20], epochs=10)
        dl.train(x=predictors, y=response_col, training_frame=cars)
        assert False, 'Expected model-build to fail when nfolds is 1 or < 0'
    except EnvironmentError:
        assert True
    try:
        dl = H2ODeepLearningEstimator(nfolds=cars.nrow + 1, fold_assignment='Modulo', hidden=[20, 20], epochs=10)
        dl.train(x=predictors, y=response_col, training_frame=cars)
        assert False, 'Expected model-build to fail when nfolds > nobs'
    except EnvironmentError:
        assert True
    try:
        dl = H2ODeepLearningEstimator(nfolds=3, hidden=[20, 20], epochs=10)
        dl.train(x=predictors, y=response_col, fold_column='fold_assignments', training_frame=cars)
        assert False, 'Expected model-build to fail when fold_column and nfolds both specified'
    except EnvironmentError:
        assert True
if __name__ == '__main__':
    pyunit_utils.standalone_test(cv_cars_dl)
else:
    cv_cars_dl()