import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
from h2o.estimators.random_forest import H2ORandomForestEstimator

def bigcatRF():
    if False:
        return 10
    bigcat = h2o.import_file(path=pyunit_utils.locate('smalldata/gbm_test/bigcat_5000x2.csv'))
    bigcat['y'] = bigcat['y'].asfactor()
    model = H2ORandomForestEstimator(ntrees=1, max_depth=1, nbins=100, nbins_cats=10)
    model.train(x='X', y='y', training_frame=bigcat)
    model.show()
if __name__ == '__main__':
    pyunit_utils.standalone_test(bigcatRF)
else:
    bigcatRF()