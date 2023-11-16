from builtins import str
import sys
sys.path.insert(1,"../../../")
import h2o
from tests import pyunit_utils
from h2o.estimators.glm import H2OGeneralizedLinearEstimator


def pyunit_make_glm_model():
    # TODO: PUBDEV-1717
    pros = h2o.import_file(pyunit_utils.locate("smalldata/prostate/prostate.csv"))

    model = H2OGeneralizedLinearEstimator(family="gaussian", alpha=[0])
    model.train(x=["AGE","DPROS","DCAPS","PSA","VOL","GLEASON"],y="CAPSULE",training_frame=pros)
    # model = h2o.glm(x=pros[["AGE","DPROS","DCAPS","PSA","VOL","GLEASON"]], y=pros["CAPSULE"], family="gaussian", alpha=[0])
    new_betas = {"AGE":0.5, "DPROS":0.5, "DCAPS":0.5, "PSA":0.5, "VOL":0.5, "GLEASON":0.5}

    names = '['
    for n in list(new_betas.keys()): names += "\""+n+"\","
    names = names[0:len(names)-1]+"]"
    betas = '['

    for b in list(new_betas.values()): betas += str(b)+","
    betas = betas[0:len(betas)-1]+"]"
    res = h2o.H2OConnection.post_json("MakeGLMModel",model=model._id,names=names,beta=betas)




if __name__ == "__main__":
    pyunit_utils.standalone_test(pyunit_make_glm_model)
else:
    pyunit_make_glm_model()
