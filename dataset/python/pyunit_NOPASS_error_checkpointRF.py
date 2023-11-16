import sys
sys.path.insert(1,"../../../")
import h2o
from tests import pyunit_utils
from h2o.estimators.random_forest import H2ORandomForestEstimator

def cars_checkpoint():

    cars = h2o.upload_file(pyunit_utils.locate("smalldata/junit/cars_20mpg.csv"))
    predictors = ["displacement","power","weight","acceleration","year"]
    response_col = "economy"

    # build first model
    model1 = H2ORandomForestEstimator(ntrees=10,max_depth=2, min_rows=10)
    model1.train(x=predictors,y=response_col,training_frame=cars)
    # model1 = h2o.random_forest(x=cars[predictors],y=cars[response_col],ntrees=10,max_depth=2, min_rows=10)

    # continue building the model
    model2 = H2ORandomForestEstimator(ntrees=11,max_depth=3, min_rows=9,r2_stopping=0.8,
                                      checkpoint=model1._id)
    model2.train(x=predictors,y=response_col,training_frame=cars)
    # model2 = h2o.random_forest(x=cars[predictors],y=cars[response_col],ntrees=11,max_depth=3, min_rows=9,r2_stopping=0.8,
    #                            checkpoint=model1._id)

    #   erroneous, not MODIFIABLE_BY_CHECKPOINT_FIELDS
    # PUBDEV-1833

    #   mtries
    try:

        model = H2ORandomForestEstimator(mtries=2,checkpoint=model1._id)
        model.train(x=predictors,y=response_col,training_frame=cars)
        # model = h2o.random_forest(y=cars[response_col], x=cars[predictors],mtries=2,checkpoint=model1._id)
        assert False, "Expected model-build to fail because mtries not modifiable by checkpoint"
    except EnvironmentError:
        assert True

    #   sample_rate
    try:
        model = H2ORandomForestEstimator(sample_rate=0.5,checkpoint=model1._id)
        model.train(x=predictors,y=response_col,training_frame=cars)
        # model = h2o.random_forest(y=cars[response_col], x=cars[predictors],sample_rate=0.5,checkpoint=model1._id)
        assert False, "Expected model-build to fail because sample_rate not modifiable by checkpoint"
    except EnvironmentError:
        assert True

    #   nbins_cats
    try:
        model = H2ORandomForestEstimator(sample_rate=0.5,checkpoint=model1._id)
        model.train(x=predictors,y=response_col,training_frame=cars)
        # model = h2o.random_forest(y=cars[response_col], x=cars[predictors],nbins_cats=99,checkpoint=model1._id)
        assert False, "Expected model-build to fail because nbins_cats not modifiable by checkpoint"
    except EnvironmentError:
        assert True

    #   nbins
    try:
        model = H2ORandomForestEstimator(nbins=99,checkpoint=model1._id)
        model.train(x=predictors,y=response_col,training_frame=cars)
        # model = h2o.random_forest(y=cars[response_col], x=cars[predictors],nbins=99,checkpoint=model1._id)
        assert False, "Expected model-build to fail because nbins not modifiable by checkpoint"
    except EnvironmentError:
        assert True

    #   balance_classes
    try:
        model = H2ORandomForestEstimator(balance_classes=True,checkpoint=model1._id)
        model.train(x=predictors,y=response_col,training_frame=cars)
        # model = h2o.random_forest(y=cars[response_col], x=cars[predictors],balance_classes=True,checkpoint=model1._id)
        assert False, "Expected model-build to fail because balance_classes not modifiable by checkpoint"
    except EnvironmentError:
        assert True

    #   nfolds
    try:
        model = H2ORandomForestEstimator(nfolds=3,checkpoint=model1._id)
        model.train(x=predictors,y=response_col,training_frame=cars)
        # model = h2o.random_forest(y=cars[response_col], x=cars[predictors],nfolds=3,checkpoint=model1._id)
        assert False, "Expected model-build to fail because nfolds not modifiable by checkpoint"
    except EnvironmentError:
        assert True




if __name__ == "__main__":
    pyunit_utils.standalone_test(cars_checkpoint)
else:
    cars_checkpoint()
