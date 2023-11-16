from builtins import range
import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
import random
from h2o.estimators.glm import H2OGeneralizedLinearEstimator

def covtype():
    if False:
        for i in range(10):
            print('nop')
    covtype = h2o.import_file(path=pyunit_utils.locate('smalldata/covtype/covtype.20k.data'))
    myY = 54
    myX = [x for x in range(0, 54) if x not in [20, 28]]
    res_class = random.randint(1, 4)
    covtype[54] = covtype[54] == res_class
    covtype_mod1 = H2OGeneralizedLinearEstimator(family='binomial', alpha=0, Lambda=0)
    covtype_mod1.train(x=myX, y=myY, training_frame=covtype)
    covtype_mod1.show()
    covtype_mod2 = H2OGeneralizedLinearEstimator(family='binomial', alpha=0.5, Lambda=0.0001)
    covtype_mod2.train(x=myX, y=myY, training_frame=covtype)
    covtype_mod2.show()
    covtype_mod3 = H2OGeneralizedLinearEstimator(family='binomial', alpha=1, Lambda=0.0001)
    covtype_mod3.train(x=myX, y=myY, training_frame=covtype)
    covtype_mod3.show()
if __name__ == '__main__':
    pyunit_utils.standalone_test(covtype)
else:
    covtype()