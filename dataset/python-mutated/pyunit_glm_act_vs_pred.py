import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from tests import pyunit_utils
from pandas.testing import assert_frame_equal

def test_predicted_vs_actual_by_variable():
    if False:
        while True:
            i = 10
    training_data = h2o.import_file(pyunit_utils.locate('smalldata/prostate/prostate_cat.csv'))
    model = H2OGeneralizedLinearEstimator(family='binomial', alpha=0, lambda_=1e-05)
    model.train(y='CAPSULE', training_frame=training_data)
    predicted = model.predict(training_data)
    print(model.predicted_vs_actual_by_variable(training_data, predicted, variable='DPROS'))
    pva_pd = model.predicted_vs_actual_by_variable(training_data, predicted, variable='DPROS', use_pandas=True)
    fr = training_data['DPROS']
    fr['predict'] = predicted['predict']
    fr['CAPSULE'] = training_data['CAPSULE']
    manual = fr.group_by('DPROS').mean().get_frame()
    print(manual)
    expected = manual['DPROS']
    expected['predict'] = manual['mean_predict']
    expected['actual'] = manual['mean_CAPSULE']
    expected_pd = expected.as_data_frame().set_index('DPROS')
    assert_frame_equal(expected_pd, pva_pd)
if __name__ == '__main__':
    pyunit_utils.standalone_test(test_predicted_vs_actual_by_variable)
else:
    test_predicted_vs_actual_by_variable()