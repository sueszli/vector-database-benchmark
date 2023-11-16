from builtins import range
import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
import random

def grid_lambda_search():
    if False:
        print('Hello World!')
    prostate = h2o.import_file(path=pyunit_utils.locate('smalldata/logreg/prostate.csv'))
    model = H2OGeneralizedLinearEstimator(family='binomial', nlambdas=5, lambda_search=True, n_folds=2)
    model.train(x=list(range(2, 9)), y=1, training_frame=prostate)
    if random.random() < 0.5:
        model_idx = 0
    else:
        model_idx = 1
    model_bestlambda = model.models(model_idx)
    params_bestlambda = model.params()
    assert len(params_bestlambda.lambdas()) <= 5, 'expected 5 or less lambdas'
    random_lambda = random.choice(params_bestlambda.lambdas())
    print('RANDOM LAMBDA')
    print(random_lambda)
    random_model = model.getGLMLambdaModel(model_bestlambda, random_lambda)
    print(random_model.Lambda())
    print(random_lambda)
    assert random_model.Lambda() == random_lambda, 'expected lambdas to be equal'
    best_model = h2o.getGLMLambdaModel(model_bestlambda, params_bestlambda.lambda_best())
    assert best_model.model() == model_bestlambda.model(), 'expected models to be equal'
    prostate_search = H2OGeneralizedLinearEstimator(family='binomial', alpha=[0.25, 0.5], nlambdas=5, lambda_search=True, n_folds=2)
    prostate_search.train(x=list(range(2, 9)), y=1, training_frame=prostate)
    model_search = prostate_search.models(model_idx)
    models_best = model_search.models(model_search.best_model())
    params_best = models_best.params()
    assert params_bestlambda.lambda_best() == params_best.lambda_best(), 'expected lambdas to be equal'
    assert len(params_best.lambda_all()) <= 20, 'expected 20 or fewer lambdas'
if __name__ == '__main__':
    pyunit_utils.standalone_test(grid_lambda_search)
else:
    grid_lambda_search()