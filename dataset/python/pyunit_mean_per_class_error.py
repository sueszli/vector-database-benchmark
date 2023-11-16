import sys
sys.path.insert(1,"../../")
import h2o
import math
from tests import pyunit_utils
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.grid import H2OGridSearch


def test_mean_per_class_error_binomial():
    gbm = H2OGradientBoostingEstimator(nfolds=3, fold_assignment="Random", seed=1234)
    cars = h2o.import_file(pyunit_utils.locate("smalldata/junit/cars_20mpg.csv"))
    cars["economy_20mpg"] = cars["economy_20mpg"].asfactor()
    r = cars[0].runif(seed=1234)
    train = cars[r > .2]
    valid = cars[r <= .2]
    response_col = "economy_20mpg"
    predictors = ["displacement","power","weight","acceleration","year"]
    gbm.distribution = "bernoulli"
    gbm.train(y=response_col, x=predictors, validation_frame=valid, training_frame=train)
    # print(gbm)
    mpce = gbm.mean_per_class_error([0.5,0.8]) ## different thresholds
    assert (abs(mpce[0][1] - 0.008264) < 1e-5)
    assert (abs(mpce[1][1] - 0.018716) < 1e-5)

    ## score on train first
    print(gbm.model_performance(train).mean_per_class_error(thresholds=[0.3,0.5]))


def test_mean_per_class_error_multinomial():
    gbm = H2OGradientBoostingEstimator(nfolds=3, fold_assignment="Random", seed=1234)
    cars = h2o.import_file(pyunit_utils.locate("smalldata/junit/cars_20mpg.csv"))
    cars["cylinders"] = cars["cylinders"].asfactor()
    r = cars[0].runif(seed=1234)
    train = cars[r > .2]
    valid = cars[r <= .2]
    response_col = "cylinders"
    predictors = ["displacement","power","weight","acceleration","year"]
    gbm.distribution="multinomial"
    gbm.train(x=predictors,y=response_col, training_frame=train, validation_frame=valid)
    # print(gbm)
    print(gbm.__class__.__mro__)
    mpce = gbm.mean_per_class_error(train=True)
    assert( mpce == 0 )
    mpce = gbm.mean_per_class_error(valid=True)
    # assert(abs(mpce - 0.207142857143 ) < 1e-5)
    assert(abs(mpce - 0.407142857143 ) < 1e-5)
    mpce = gbm.mean_per_class_error(xval=True)
    # assert(abs(mpce - 0.350071715433 ) < 1e-5)
    assert(abs(mpce - 0.35127653471 ) < 1e-5)


def test_mean_per_class_error_grid():
    gbm = H2OGradientBoostingEstimator(nfolds=3, fold_assignment="Random", seed=1234)
    cars = h2o.import_file(pyunit_utils.locate("smalldata/junit/cars_20mpg.csv"))
    cars["cylinders"] = cars["cylinders"].asfactor()
    r = cars[0].runif(seed=1234)
    train = cars[r > .2]
    valid = cars[r <= .2]
    response_col = "cylinders"
    predictors = ["displacement","power","weight","acceleration","year"]
    gbm.distribution="multinomial"
    
    ## Early stopping
    gbm.stopping_rounds=2
    gbm.stopping_metric="mean_per_class_error"
    gbm.ntrees=10000
    gbm.max_depth=3
    gbm.min_rows=1
    gbm.learn_rate=0.01
    gbm.score_tree_interval=1
    gbm.nfolds=None
    gbm.fold_assignment=None
    gbm.train(x=predictors,y=response_col, training_frame=train, validation_frame=valid)
    print(gbm)
    print(gbm.scoring_history())


    ## Grid search
    hyper_params_tune = {'max_depth' : list(range(1,10+1,1)),
                  'sample_rate': [x/100. for x in range(20,101)],
                  'col_sample_rate' : [x/100. for x in range(20,101)],
                  'col_sample_rate_per_tree': [x/100. for x in range(20,101)],
                  'col_sample_rate_change_per_level': [x/100. for x in range(90,111)],
                  'min_rows': [2**x for x in range(0,int(math.log(train.nrow,2)-2)+1)],
                  'nbins': [2**x for x in range(4,11)],
                  'nbins_cats': [2**x for x in range(4,13)],
                  'min_split_improvement': [0,1e-8,1e-6,1e-4],
                  'histogram_type': ["UniformAdaptive","QuantilesGlobal","RoundRobin"]}

    search_criteria_tune = {'strategy': "RandomDiscrete",
                     'max_runtime_secs': 600,  ## limit the runtime to 10 minutes
                     'max_models': 10,
                     'seed' : 1234,
                     'stopping_rounds' : 5,
                     'stopping_metric' : "mean_per_class_error",
                     'stopping_tolerance': 1e-3
                     }

    grid = H2OGridSearch(H2OGradientBoostingEstimator, hyper_params=hyper_params_tune, search_criteria=search_criteria_tune)
    grid.train(x=predictors, y=response_col, training_frame=train, validation_frame=valid,distribution="multinomial", seed=1234,
        stopping_rounds = 10, stopping_metric = "mean_per_class_error", stopping_tolerance=1e-3)

    print(grid) ## sorted by logloss
    print(grid.get_grid("mean_per_class_error"))


pyunit_utils.run_tests([
    test_mean_per_class_error_binomial,
    test_mean_per_class_error_multinomial,
    test_mean_per_class_error_grid
])
