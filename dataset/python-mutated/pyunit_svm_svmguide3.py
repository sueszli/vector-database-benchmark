import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.psvm import H2OSupportVectorMachineEstimator

def svm_svmguide3():
    if False:
        for i in range(10):
            print('nop')
    svmguide3 = h2o.import_file(pyunit_utils.locate('smalldata/svm_test/svmguide3scale.svm'))
    svmguide3_test = h2o.import_file(pyunit_utils.locate('smalldata/svm_test/svmguide3scale_test.svm'))
    svm_tuned = H2OSupportVectorMachineEstimator(hyper_param=128, gamma=0.125, disable_training_metrics=False)
    svm_tuned.train(y='C1', training_frame=svmguide3, validation_frame=svmguide3_test)
    accuracy = svm_tuned.model_performance(valid=True).accuracy()[0][1]
    assert accuracy >= 0.8
if __name__ == '__main__':
    pyunit_utils.standalone_test(svm_svmguide3)
else:
    svm_svmguide3()