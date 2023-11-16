import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.gam import H2OGeneralizedAdditiveEstimator

def test_gam_model_predict():
    if False:
        i = 10
        return i + 15
    covtype_df = h2o.import_file(pyunit_utils.locate('bigdata/laptop/covtype/covtype.full.csv'))
    (train, valid) = covtype_df.split_frame([0.9], seed=1234)
    covtype_X = covtype_df.col_names[:-1]
    covtype_y = covtype_df.col_names[-1]
    gam_multi_valid = H2OGeneralizedAdditiveEstimator(family='multinomial', solver='IRLSM', bs=[0, 0, 0], gam_columns=['Elevation', 'Aspect', 'Slope'], standardize=True, nfolds=2, fold_assignment='modulo', alpha=[0.9, 0.5, 0.1], lambda_search=True, nlambdas=5, max_iterations=3, seed=1234)
    gam_multi_valid.train(covtype_X, covtype_y, training_frame=train, validation_frame=valid)
    gam_multi = H2OGeneralizedAdditiveEstimator(family='multinomial', solver='IRLSM', bs=[0, 0, 0], gam_columns=['Elevation', 'Aspect', 'Slope'], standardize=True, nfolds=2, fold_assignment='modulo', alpha=[0.9, 0.5, 0.1], lambda_search=True, nlambdas=5, max_iterations=3, seed=1234)
    gam_multi.train(covtype_X, covtype_y, training_frame=train)
    gam_multi_coef = gam_multi.coef()
    gam_multi_valid_coef = gam_multi_valid.coef()
    pyunit_utils.assertEqualCoeffDicts(gam_multi_coef['coefficients'], gam_multi_valid_coef['coefficients'])
if __name__ == '__main__':
    pyunit_utils.standalone_test(test_gam_model_predict)
else:
    test_gam_model_predict()