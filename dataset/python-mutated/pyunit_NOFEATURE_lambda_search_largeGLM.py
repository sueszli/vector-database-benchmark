from builtins import range
import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
import random

def lambda_search():
    if False:
        i = 10
        return i + 15
    prostate = h2o.import_file(pyunit_utils.locate('smalldata/logreg/prostate.csv'))
    prostate_nosearch = H2OGeneralizedLinearEstimator(family='binomial', nlambdas=5, lambda_search=False, n_folds=2)
    prostate_nosearch.train(x=list(range(2, 9)), y=1, training_frame=prostate.hex)
    params_nosearch = prostate_nosearch.params()
    try:
        prostate_nosearch.getGLMLambdaModel(0.5)
        assert False, 'expected an error'
    except EnvironmentError:
        assert True
    prostate_search = H2OGeneralizedLinearEstimator(family='binomial', nlambdas=5, lambda_search=True, n_folds=2)
    prostate_search.train(x=list(range(2, 9)), y=1, training_frame=prostate.hex)
    params_search = prostate_search.params()
    random_lambda = random.choice(prostate_search.lambda_all())
    random_model = prostate_search.getGLMLambdaModel(random_lambda)
    assert random_model.getLambda() == random_lambda, 'expected equal lambdas'
    best_model = prostate_search.getGLMLambdaModel(params_search.bestlambda())
    assert best_model.model() == prostate_search.model(), 'expected models to be equal'
if __name__ == '__main__':
    pyunit_utils.standalone_test(lambda_search)
else:
    lambda_search()