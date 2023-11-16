import os
import sys
sys.path.insert(1, os.path.join('..', '..', '..', '..'))
import h2o
from tests import pyunit_utils
from h2o.estimators import H2OTargetEncoderEstimator
from h2o.estimators import H2OGradientBoostingEstimator

def test_that_te_is_helpful_for_titanic_gbm_xval():
    if False:
        return 10
    titanic = h2o.import_file(pyunit_utils.locate('smalldata/gbm_test/titanic.csv'))
    titanic['survived'] = titanic['survived'].asfactor()
    response = 'survived'
    (train, test) = titanic.split_frame(ratios=[0.8], seed=1234)
    encoded_columns = ['home.dest', 'cabin', 'embarked']
    blended_avg = True
    inflection_point = 3
    smoothing = 10
    noise = 0.15
    data_leakage_handling = 'k_fold'
    fold_column = 'kfold_column'
    train[fold_column] = train.kfold_column(n_folds=5, seed=3456)
    titanic_te = H2OTargetEncoderEstimator(fold_column=fold_column, data_leakage_handling=data_leakage_handling, blending=blended_avg, inflection_point=inflection_point, smoothing=smoothing, seed=1234)
    titanic_te.train(x=encoded_columns, y=response, training_frame=train)
    train_te = titanic_te.transform(frame=train, as_training=True)
    test_te = titanic_te.transform(frame=test, noise=0.0)
    gbm_with_te = H2OGradientBoostingEstimator(max_depth=6, min_rows=1, fold_column=fold_column, score_tree_interval=5, ntrees=10000, sample_rate=0.8, col_sample_rate=0.8, seed=1234, stopping_rounds=5, stopping_metric='auto', stopping_tolerance=0.001, model_id='gbm_with_te')
    x_with_te = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'cabin_te', 'embarked_te', 'home.dest_te']
    gbm_with_te.train(x=x_with_te, y=response, training_frame=train_te)
    my_gbm_metrics_train_auc = gbm_with_te.model_performance(train_te).auc()
    print('TE train:' + str(my_gbm_metrics_train_auc))
    my_gbm_metrics = gbm_with_te.model_performance(test_te)
    auc_with_te = my_gbm_metrics.auc()
    print('TE test:' + str(auc_with_te))
    gbm_baseline = H2OGradientBoostingEstimator(max_depth=6, min_rows=1, fold_column=fold_column, score_tree_interval=5, ntrees=10000, sample_rate=0.8, col_sample_rate=0.8, seed=1234, stopping_rounds=5, stopping_metric='auto', stopping_tolerance=0.001, model_id='gbm_baseline')
    x_baseline = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'cabin', 'embarked', 'home.dest']
    gbm_baseline.train(x=x_baseline, y=response, training_frame=train)
    gbm_baseline_metrics = gbm_baseline.model_performance(test)
    auc_baseline = gbm_baseline_metrics.auc()
    print('Baseline test:' + str(auc_baseline))
    assert auc_with_te > auc_baseline
testList = [test_that_te_is_helpful_for_titanic_gbm_xval]
pyunit_utils.run_tests(testList)