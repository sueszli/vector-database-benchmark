from past.utils import old_div
import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.glm import H2OGeneralizedLinearEstimator as glm

def test_glm_scoring_history_TomasF2():
    if False:
        for i in range(10):
            print('nop')
    df = h2o.import_file('https://h2o-public-test-data.s3.amazonaws.com/smalldata/prostate/prostate.csv')
    df['CAPSULE'] = df['CAPSULE'].asfactor()
    glmModel = glm(generate_scoring_history=False, lambda_search=False)
    glmModel.train(y='CAPSULE', training_frame=df)
    glmModel.scoring_history()
    assert len(glmModel._model_json['output']['scoring_history'].cell_values) == 5
    glmModel2 = glm(generate_scoring_history=True, lambda_search=False)
    glmModel2.train(y='CAPSULE', training_frame=df)
    glmModel2.scoring_history()
    assert len(glmModel._model_json['output']['scoring_history'].cell_values) > len(glmModel2._model_json['output']['scoring_history'].cell_values)
if __name__ == '__main__':
    pyunit_utils.standalone_test(test_glm_scoring_history_TomasF2)
else:
    test_glm_scoring_history_TomasF2()