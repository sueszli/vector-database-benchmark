import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from tests import pyunit_utils

def test_rollup_stats():
    if False:
        return 10
    df = h2o.import_file(pyunit_utils.locate('smalldata/glm_test/rollup_stat_test.csv'))
    df['RACE'] = df['RACE'].asfactor()
    glm = H2OGeneralizedLinearEstimator(generate_scoring_history=True, score_iteration_interval=5, non_negative=True, alpha=[0.5, 1.0], standardize=False, nfolds=5, seed=7)
    try:
        glm.train(y='RACE', training_frame=df)
    except (OSError, EnvironmentError) as e:
        assert 'non_negative:  does not work with multinomial family.' in str(e)
if __name__ == '__main__':
    pyunit_utils.standalone_test(test_rollup_stats)
else:
    test_rollup_stats()