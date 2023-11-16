import sys, os
sys.path.insert(1, os.path.join('..', '..', '..'))
import h2o
from tests import pyunit_utils
from h2o.estimators.deeplearning import H2ODeepLearningEstimator

def deeplearning_multi():
    if False:
        return 10
    print('Test checks if Deep Learning works fine with a categorical dataset')
    prostate = h2o.import_file(path=pyunit_utils.locate('smalldata/logreg/prostate.csv'))
    prostate[1] = prostate[1].asfactor()
    prostate[2] = prostate[2].asfactor()
    prostate[3] = prostate[3].asfactor()
    prostate[4] = prostate[4].asfactor()
    prostate[5] = prostate[5].asfactor()
    prostate = prostate.drop('ID')
    prostate.describe()
    hh = H2ODeepLearningEstimator(loss='CrossEntropy', hidden=[10, 10], use_all_factor_levels=False)
    hh.train(x=list(set(prostate.names) - {'CAPSULE'}), y='CAPSULE', training_frame=prostate)
    hh.show()
if __name__ == '__main__':
    pyunit_utils.standalone_test(deeplearning_multi)
else:
    deeplearning_multi()