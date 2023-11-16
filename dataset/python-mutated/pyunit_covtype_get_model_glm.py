from builtins import range
import sys
sys.path.insert(1, '../../../')
import h2o
from tests import pyunit_utils
import random
from h2o.estimators.glm import H2OGeneralizedLinearEstimator

def covtype_get_model():
    if False:
        return 10
    covtype = h2o.import_file(path=pyunit_utils.locate('smalldata/covtype/covtype.20k.data'))
    Y = 54
    X = list(range(0, 20)) + list(range(29, 54))
    res_class = random.randint(1, 4)
    covtype[54] = covtype[54] == res_class
    covtype_mod1 = H2OGeneralizedLinearEstimator(family='binomial', alpha=0, Lambda=0)
    covtype_mod1.train(x=X, y=Y, training_frame=covtype)
    covtype_mod1.show()
    covtype_mod1 = h2o.get_model(covtype_mod1.model_id)
    covtype_mod1.show()
    covtype_mod2 = H2OGeneralizedLinearEstimator(family='binomial', alpha=0.5, Lambda=0.0001)
    covtype_mod2.train(x=X, y=Y, training_frame=covtype)
    covtype_mod2.show()
    covtype_mod2 = h2o.get_model(covtype_mod2.model_id)
    covtype_mod2.show()
    covtype_mod3 = H2OGeneralizedLinearEstimator(family='binomial', alpha=1, Lambda=0.0001)
    covtype_mod3.train(x=X, y=Y, training_frame=covtype)
    covtype_mod3.show()
    covtype_mod3 = h2o.get_model(covtype_mod3.model_id)
    covtype_mod3.show()
if __name__ == '__main__':
    pyunit_utils.standalone_test(covtype_get_model)
else:
    covtype_get_model()