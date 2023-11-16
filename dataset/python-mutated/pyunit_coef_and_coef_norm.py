import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.glm import H2OGeneralizedLinearEstimator

def test_coef():
    if False:
        print('Hello World!')
    prostate = h2o.import_file(path=pyunit_utils.locate('smalldata/prostate/prostate.csv'))
    prostate['CAPSULE'] = prostate['CAPSULE'].asfactor()
    prostate['RACE'] = prostate['RACE'].asfactor()
    prostate['DCAPS'] = prostate['DCAPS'].asfactor()
    prostate['DPROS'] = prostate['DPROS'].asfactor()
    predictors = ['AGE', 'RACE', 'VOL', 'GLEASON']
    response_col = 'CAPSULE'
    glm_model = H2OGeneralizedLinearEstimator(family='binomial', lambda_=0, seed=1234)
    glm_model.train(predictors, response_col, training_frame=prostate)
    tbl = glm_model._model_json['output']['coefficients_table']
    column_name = 'coefficients'
    assert column_name in tbl.as_data_frame().columns, "expected {0} column in coefficients table, but didn't find it".format(column_name)
    item_list = tbl[column_name]
    assert all((isinstance(item, float) for item in item_list)), '{0} column expected float values, got {1} '.format(column_name, [type(item) for item in item_list])
    assert glm_model.coef() == {name: coef for (name, coef) in zip(tbl['names'], tbl['coefficients'])}, "coefficients don't match coefficients in glm coefficients table"
if __name__ == '__main__':
    pyunit_utils.standalone_test(test_coef)
else:
    test_coef()

def test_coef_norm():
    if False:
        i = 10
        return i + 15
    prostate = h2o.import_file(path=pyunit_utils.locate('smalldata/prostate/prostate.csv'))
    prostate['CAPSULE'] = prostate['CAPSULE'].asfactor()
    prostate['RACE'] = prostate['RACE'].asfactor()
    prostate['DCAPS'] = prostate['DCAPS'].asfactor()
    prostate['DPROS'] = prostate['DPROS'].asfactor()
    predictors = ['AGE', 'RACE', 'VOL', 'GLEASON']
    response_col = 'CAPSULE'
    glm_model = H2OGeneralizedLinearEstimator(family='binomial', lambda_=0, seed=1234)
    glm_model.train(predictors, response_col, training_frame=prostate)
    tbl = glm_model._model_json['output']['coefficients_table']
    column_name = 'standardized_coefficients'
    assert column_name in tbl.as_data_frame().columns, "expected {0} column in coefficients table, but didn't find it".format(column_name)
    item_list = tbl[column_name]
    assert all((isinstance(item, float) for item in item_list)), '{0} column expected float values, got {1} '.format(column_name, [type(item) for item in item_list])
    assert glm_model.coef_norm() == {name: coef for (name, coef) in zip(tbl['names'], tbl['standardized_coefficients'])}, "std coef don't match std coef in glm coefficients table"
if __name__ == '__main__':
    pyunit_utils.standalone_test(test_coef_norm)
else:
    test_coef_norm()