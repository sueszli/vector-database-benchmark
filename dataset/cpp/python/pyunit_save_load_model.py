import sys
sys.path.insert(1,"../..")
import h2o
from tests import pyunit_utils
import os
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.estimators.estimator_base import H2OEstimator


def save_load_model():
    prostate = h2o.import_file(pyunit_utils.locate("smalldata/prostate/prostate.csv"))
    prostate["CAPSULE"] = prostate["CAPSULE"].asfactor()

    prostate_glm = H2OGeneralizedLinearEstimator(family="binomial", alpha=[0.5])
    prostate_glm.train(x=["AGE","RACE","PSA","DCAPS"], y="CAPSULE", training_frame=prostate)
    path = pyunit_utils.locate("results")

    assert os.path.isdir(path), "Expected save directory {0} to exist, but it does not.".format(path)
    model_path = h2o.save_model(prostate_glm, path=path, force=True)

    assert os.path.isfile(model_path), "Expected load file {0} to exist, but it does not.".format(model_path)
    the_model = h2o.load_model(model_path)

    assert isinstance(the_model, H2OEstimator), "Expected and H2OBinomialModel, but got {0}".format(the_model)



if __name__ == "__main__":
    pyunit_utils.standalone_test(save_load_model)
else:
    save_load_model()
