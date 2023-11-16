import sys

sys.path.insert(1, "../../../")
import h2o
from tests import pyunit_utils
from h2o.estimators.gbm import H2OGradientBoostingEstimator


def set_parallel(value):
    h2o.rapids("(setproperty \"{}\" \"{}\")".format("sys.ai.h2o.sharedtree.crossvalidation.parallelMainModelBuilding", str(value).lower()))


def set_best_cv(value):
    h2o.rapids("(setproperty \"{}\" \"{}\")".format("sys.ai.h2o.sharedtree.crossvalidation.useBestCVIteration", str(value).lower()))


def cv_nfolds_gbm():
    loan_data = h2o.import_file(path=pyunit_utils.locate("smalldata/prostate/prostate.csv"))
    loan_data["CAPSULE"] = loan_data["CAPSULE"].asfactor()

    try:
        # parallel main model building cannot be used when we use best CV iterations right now    
        set_best_cv(False)
        model_default = H2OGradientBoostingEstimator(nfolds=5, distribution="bernoulli", ntrees=500,
                                                     score_tree_interval=3, stopping_rounds=2, seed=42,
                                                     ignored_columns=["ID"])
        try:
            set_parallel(True)
            model_default.train(y="CAPSULE", training_frame=loan_data)
        finally:
            set_parallel(False)
        preds_default = model_default.predict(loan_data)
    
        model_sequential = H2OGradientBoostingEstimator(nfolds=5, distribution="bernoulli", ntrees=500,
                                                        score_tree_interval=3, stopping_rounds=2, seed=42,
                                                        ignored_columns=["ID"])
        model_sequential.train(y="CAPSULE", training_frame=loan_data)
        preds_sequential = model_sequential.predict(loan_data)
    
        assert model_default.actual_params["ntrees"] == model_sequential.actual_params["ntrees"]
        pyunit_utils.compare_frames_local(preds_default, preds_sequential, prob=1.0)
    finally:
        set_best_cv(False)


if __name__ == "__main__":
    pyunit_utils.standalone_test(cv_nfolds_gbm)
else:
    cv_nfolds_gbm()
