import sys, os
sys.path.insert(1, os.path.join('..', '..'))
import h2o
from h2o.utils.typechecks import is_type
from h2o.estimators import H2OGradientBoostingEstimator, H2OGeneralizedLinearEstimator
from tests import pyunit_utils

def gbm_model_build():
    if False:
        print('Hello World!')
    '\n    Train gbm model\n    :returns model, training frame \n    '
    prostate_train = h2o.import_file(path=pyunit_utils.locate('smalldata/logreg/prostate_train.csv'))
    prostate_train['CAPSULE'] = prostate_train['CAPSULE'].asfactor()
    ntrees = 100
    learning_rate = 0.1
    depth = 5
    min_rows = 10
    gbm_h2o = H2OGradientBoostingEstimator(ntrees=ntrees, learn_rate=learning_rate, max_depth=depth, min_rows=min_rows, distribution='bernoulli')
    gbm_h2o.train(x=list(range(1, prostate_train.ncol)), y='CAPSULE', training_frame=prostate_train)
    return (gbm_h2o, prostate_train)

def test_metrics_gbm():
    if False:
        for i in range(10):
            print('nop')
    '\n    test metrics values from the Permutation Variable Importance\n    '
    (model, fr) = gbm_model_build()
    pm_h2o_df = model.permutation_importance(fr, use_pandas=False, metric='AUC')
    for col in ['Relative Importance', 'Scaled Importance', 'Percentage']:
        assert isinstance(pm_h2o_df[col][0], float)
    assert is_type(pm_h2o_df[0][0], str)
    pm_pd_df = model.permutation_importance(fr, use_pandas=True, metric='AUC')
    for col in pm_pd_df.columns:
        assert isinstance(pm_pd_df.iloc[0][col], float)
    metrics = ['AUTO', 'MSE', 'RMSE', 'AUC', 'logloss']
    for metric in metrics:
        pd_pfi = model.permutation_importance(fr, use_pandas=False, metric=metric)
        for col in pd_pfi.col_header[1:]:
            assert isinstance(pd_pfi[col][0], float)
    for metric in metrics:
        pd_pfi = model.permutation_importance(fr, use_pandas=False, n_repeats=5, metric=metric)
        for (i, col) in enumerate(pd_pfi.col_header[1:]):
            assert col == 'Run {}'.format(1 + i)
            assert isinstance(pd_pfi[col][0], float)
    try:
        pfi = model.permutation_importance(fr, use_pandas=False, n_samples=0, features=[], seed=42)
        assert False, 'This should fail on validation - n_samples=0.'
    except h2o.exceptions.H2OValueError:
        pass
    try:
        pfi = model.permutation_importance(fr, use_pandas=False, n_repeats=0, features=[], seed=42)
        assert False, 'This should fail on validation - n_repeats = 0.'
    except h2o.exceptions.H2OValueError:
        pass
    try:
        pfi = model.permutation_importance(fr[['AGE', 'PSA']], use_pandas=False, seed=42)
        assert False, 'This should fail on validation - missing response.'
    except h2o.exceptions.H2OValueError:
        pass
    try:
        pfi = model.permutation_importance(fr, use_pandas=False, features=['lorem', 'ipsum', 'dolor'], seed=42)
        assert False, 'This should fail on validation - non-existent features.'
    except h2o.exceptions.H2OValueError:
        pass
    try:
        pfi = model.permutation_importance(fr, use_pandas=False, n_samples=1, features=[])
        assert False, 'This should throw an exception since we cannot permute one row.'
    except h2o.exceptions.H2OValueError:
        pass
    pfi = model.permutation_importance(fr, use_pandas=False, n_samples=10, features=[])
    for col in pfi.col_header[1:]:
        assert isinstance(pfi[col][0], float)
    pfi = model.permutation_importance(fr, use_pandas=False, n_samples=-1, features=['PSA'])
    assert len(pfi.cell_values) == 1
    for col in pfi.col_header[1:]:
        assert isinstance(pfi[col][0], float)
    pfi = model.permutation_importance(fr, use_pandas=False, n_samples=-1, features=['PSA', 'AGE'])
    assert len(pfi.cell_values) == 2
    for col in pfi.col_header[1:]:
        assert isinstance(pfi[col][0], float)

def test_big_data_cars():
    if False:
        while True:
            i = 10
    '\n    Test big data dataset, with metric logloss. \n    '
    h2o_df = h2o.import_file(path=pyunit_utils.locate('bigdata/laptop/lending-club/loan.csv'))
    predictors = h2o_df.col_names
    response_col = h2o_df.col_names[12]
    predictors.remove(response_col)
    model = H2OGeneralizedLinearEstimator(family='binomial')
    model.train(y=response_col, x=predictors, training_frame=h2o_df)
    metric = 'logloss'
    pm_h2o_df = model.permutation_importance(h2o_df, use_pandas=True, n_samples=-1, metric=metric)
    for pred in predictors:
        if pred == 'Variable':
            continue
        assert isinstance(pm_h2o_df.loc[pred, 'Relative Importance'], float)
    pm_h2o_df = model.permutation_importance(h2o_df, use_pandas=True, n_samples=100, metric=metric)
    for pred in predictors:
        if pred == 'Variable':
            continue
        assert isinstance(pm_h2o_df.loc[pred, 'Relative Importance'], float)

def test_permutation_importance_plot_works():
    if False:
        while True:
            i = 10
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    (model, fr) = gbm_model_build()
    model.permutation_importance_plot(fr)
    model.permutation_importance_plot(fr, n_repeats=5)
    plt.close('all')
pyunit_utils.run_tests([test_metrics_gbm, test_big_data_cars, test_permutation_importance_plot_works])