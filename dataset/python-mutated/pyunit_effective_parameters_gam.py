import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.gam import H2OGeneralizedAdditiveEstimator

def test_gam_effective_parameters():
    if False:
        for i in range(10):
            print('nop')
    h2o_data = h2o.import_file(pyunit_utils.locate('smalldata/glm_test/binomial_20_cols_10KRows.csv'))
    h2o_data['C1'] = h2o_data['C1'].asfactor()
    h2o_data['C2'] = h2o_data['C2'].asfactor()
    h2o_data['C21'] = h2o_data['C21'].asfactor()
    gam = H2OGeneralizedAdditiveEstimator(family='binomial', gam_columns=['C11', 'C12', 'C13'], scale=[1, 1, 1], num_knots=[5, 6, 7], standardize=True, Lambda=[0], alpha=[0], max_iterations=3)
    gam.train(x=['C1', 'C2'], y='C21', training_frame=h2o_data)
    assert gam.parms['solver']['input_value'] == 'AUTO'
    assert gam.parms['solver']['actual_value'] == 'IRLSM'
    assert gam.parms['fold_assignment']['input_value'] == 'AUTO'
    assert gam.parms['fold_assignment']['actual_value'] is None
    try:
        h2o.rapids('(setproperty "{}" "{}")'.format('sys.ai.h2o.algos.evaluate_auto_model_parameters', 'false'))
        gam = H2OGeneralizedAdditiveEstimator(family='binomial', gam_columns=['C11', 'C12', 'C13'], scale=[1, 1, 1], num_knots=[5, 6, 7], standardize=True, Lambda=[0], alpha=[0], max_iterations=3)
        gam.train(x=['C1', 'C2'], y='C21', training_frame=h2o_data)
        assert gam.parms['solver']['input_value'] == 'AUTO'
        assert gam.parms['solver']['actual_value'] == 'AUTO'
        assert gam.parms['fold_assignment']['input_value'] == 'AUTO'
        assert gam.parms['fold_assignment']['actual_value'] == 'AUTO'
    finally:
        h2o.rapids('(setproperty "{}" "{}")'.format('sys.ai.h2o.algos.evaluate_auto_model_parameters', 'true'))
if __name__ == '__main__':
    pyunit_utils.standalone_test(test_gam_effective_parameters)
else:
    test_gam_effective_parameters()