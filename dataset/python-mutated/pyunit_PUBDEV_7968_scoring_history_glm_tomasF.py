from past.utils import old_div
import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.glm import H2OGeneralizedLinearEstimator as glm

def test_glm_scoring_history_TomasF():
    if False:
        return 10
    df = h2o.import_file(pyunit_utils.locate('smalldata/prostate/prostate.csv'))
    df['CAPSULE'] = df['CAPSULE'].asfactor()
    glmModel = glm(generate_scoring_history=True)
    glmModel.train(y='CAPSULE', training_frame=df)
    glmModel.scoring_history()
if __name__ == '__main__':
    pyunit_utils.standalone_test(test_glm_scoring_history_TomasF)
else:
    test_glm_scoring_history_TomasF()