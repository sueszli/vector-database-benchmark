import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.anovaglm import H2OANOVAGLMEstimator

def testFrameTransform():
    if False:
        print('Hello World!')
    train = h2o.import_file(path=pyunit_utils.locate('smalldata/prostate/prostate_complete.csv.zip'))
    y = 'CAPSULE'
    x = ['AGE', 'VOL', 'DCAPS']
    train[10, 2] = None
    train[20, 7] = None
    train[y] = train[y].asfactor()
    model1 = H2OANOVAGLMEstimator(family='binomial', lambda_=0, missing_values_handling='skip')
    model1.train(x=x, y=y, training_frame=train)
    train.drop([10, 20], axis=0)
    model2 = H2OANOVAGLMEstimator(family='binomial', lambda_=0, missing_values_handling='skip')
    model2.train(x=x, y=y, training_frame=train)
    summary1 = model1._model_json['output']['model_summary']
    summary2 = model2._model_json['output']['model_summary']
    pyunit_utils.assert_H2OTwoDimTable_equal_upto(summary1, summary2, summary1.col_header)
if __name__ == '__main__':
    pyunit_utils.standalone_test(testFrameTransform)
else:
    testFrameTransform()