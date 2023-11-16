import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from tests import pyunit_utils

def test_binomial_alpha():
    if False:
        print('Hello World!')
    training_data = h2o.import_file(pyunit_utils.locate('smalldata/logreg/benign.csv'))
    Y = 3
    X = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]
    model1 = H2OGeneralizedLinearEstimator(family='binomial', alpha=[0, 0.2, 1], lambda_search=True, generate_scoring_history=True, nlambdas=5)
    model1.train(x=X, y=Y, training_frame=training_data)
    model2 = H2OGeneralizedLinearEstimator(family='binomial', alpha=[0, 0.2, 1], lambda_search=True, generate_scoring_history=True, nlambdas=5)
    model2.train(x=X, y=Y, training_frame=training_data)
    pyunit_utils.assertCoefDictEqual(model1.coef(), model2.coef())
    model1 = H2OGeneralizedLinearEstimator(family='binomial', alpha=[0, 0.8, 1], lambda_search=False, generate_scoring_history=True, Lambda=[0, 0.1])
    model1.train(x=X, y=Y, training_frame=training_data)
    model2 = H2OGeneralizedLinearEstimator(family='binomial', alpha=[0, 0.8, 1], lambda_search=False, generate_scoring_history=True, Lambda=[0, 0.1])
    model2.train(x=X, y=Y, training_frame=training_data)
    pyunit_utils.assertCoefDictEqual(model1.coef(), model2.coef())
    model1 = H2OGeneralizedLinearEstimator(family='binomial', alpha=[0, 0.8, 1], lambda_search=True, generate_scoring_history=True, nfolds=2, seed=12345, nlambdas=5)
    model1.train(x=X, y=Y, training_frame=training_data)
    model2 = H2OGeneralizedLinearEstimator(family='binomial', alpha=[0, 0.8, 1], lambda_search=True, generate_scoring_history=True, nfolds=2, seed=12345, nlambdas=5)
    model2.train(x=X, y=Y, training_frame=training_data)
    pyunit_utils.assertCoefDictEqual(model1.coef(), model2.coef())
    model1 = H2OGeneralizedLinearEstimator(family='binomial', alpha=[0, 0.2, 1], lambda_search=False, generate_scoring_history=True, nfolds=2, seed=12345, Lambda=[0, 0.001])
    model1.train(x=X, y=Y, training_frame=training_data)
    model2 = H2OGeneralizedLinearEstimator(family='binomial', alpha=[0, 0.2, 1], lambda_search=False, generate_scoring_history=True, nfolds=2, seed=12345, Lambda=[0, 0.001])
    model2.train(x=X, y=Y, training_frame=training_data)
    pyunit_utils.assertCoefDictEqual(model1.coef(), model2.coef())
if __name__ == '__main__':
    pyunit_utils.standalone_test(test_binomial_alpha)
else:
    test_binomial_alpha()