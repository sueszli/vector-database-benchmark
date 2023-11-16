import itertools
import os
import sys
sys.path.insert(1, os.path.join('..', '..', '..', '..'))
import h2o
from h2o.estimators import H2OTargetEncoderEstimator, H2OGradientBoostingEstimator
from h2o.exceptions import H2OTypeError
from tests import pyunit_utils as pu
seed = 42

def load_dataset(incl_test=False, incl_foldc=False):
    if False:
        i = 10
        return i + 15
    fr = h2o.import_file(pu.locate('smalldata/titanic/titanic_expanded.csv'), header=1)
    target = 'survived'
    train = fr
    test = None
    if incl_test:
        fr = fr.split_frame(ratios=[0.8], destination_frames=['titanic_train', 'titanic_test'], seed=seed)
        train = fr[0]
        test = fr[1]
    if incl_foldc:
        train['foldc'] = train.kfold_column(3, seed)
    return pu.ns(train=train, test=test, target=target)

def test_default_strategy_is_none():
    if False:
        print('Hello World!')
    ds = load_dataset(incl_test=True)
    te = H2OTargetEncoderEstimator(noise=0)
    te.train(y=ds.target, training_frame=ds.train)
    encoded = te.predict(ds.test)
    te_none = H2OTargetEncoderEstimator(data_leakage_handling='none', noise=0)
    te_none.train(y=ds.target, training_frame=ds.train)
    encoded_none = te_none.predict(ds.test)
    assert pu.compare_frames(encoded, encoded_none, 0, tol_numeric=1e-05)

def test_fails_on_unknown_strategy():
    if False:
        print('Hello World!')
    try:
        H2OTargetEncoderEstimator(data_leakage_handling='foo')
        assert False, 'should have raised'
    except H2OTypeError as e:
        assert 'data_leakage_handling' in str(e)

def test_kfold_requires_fold_column():
    if False:
        return 10
    ds = load_dataset(incl_foldc=True)
    te = H2OTargetEncoderEstimator(data_leakage_handling='kfold')
    try:
        te.train(y=ds.target, training_frame=ds.train)
        assert False, 'should have raised'
    except Exception as e:
        assert 'Fold column is required when using KFold leakage handling strategy' in str(e)
    te.train(y=ds.target, training_frame=ds.train, fold_column='foldc')
    assert te.predict(ds.train) is not None

def test_loo_requires_target_to_encode_training_frame():
    if False:
        return 10
    ds = load_dataset()
    te = H2OTargetEncoderEstimator(data_leakage_handling='leave_one_out')
    te.train(y=ds.target, training_frame=ds.train)
    train_no_target = h2o.assign(ds.train.drop(ds.target), 'train_no_target')
    assert train_no_target is not None
    try:
        te.transform(train_no_target, as_training=True)
        assert False, 'should have raised'
    except Exception as e:
        assert 'LeaveOneOut strategy requires a response column' in str(e)
    assert te.predict(train_no_target) is not None

def test_strategies_produce_different_results_for_training():
    if False:
        i = 10
        return i + 15
    ds = load_dataset(incl_foldc=True)
    te_none = H2OTargetEncoderEstimator(noise=0)
    te_none.train(y=ds.target, training_frame=ds.train)
    encoded_none = te_none.transform(ds.train, as_training=True)
    te_loo = H2OTargetEncoderEstimator(data_leakage_handling='leave_one_out', noise=0)
    te_loo.train(y=ds.target, training_frame=ds.train)
    encoded_loo = te_loo.transform(ds.train, as_training=True)
    te_kfold = H2OTargetEncoderEstimator(data_leakage_handling='kfold', noise=0)
    te_kfold.train(y=ds.target, training_frame=ds.train, fold_column='foldc')
    encoded_kfold = te_kfold.transform(ds.train, as_training=True)
    for (l, r) in itertools.combinations([encoded_none, encoded_loo, encoded_kfold], 2):
        try:
            assert pu.compare_frames(l, r, 0, tol_numeric=0.01)
            assert False, 'should have raised'
        except AssertionError as ae:
            assert 'should have raised' not in str(ae)

def test_strategies_produce_same_results_when_applied_on_new_data():
    if False:
        for i in range(10):
            print('nop')
    ds = load_dataset(incl_test=True, incl_foldc=True)
    te_none = H2OTargetEncoderEstimator(noise=0)
    te_none.train(y=ds.target, training_frame=ds.train)
    encoded_none = te_none.transform(ds.test)
    te_loo = H2OTargetEncoderEstimator(data_leakage_handling='leave_one_out', noise=0)
    te_loo.train(y=ds.target, training_frame=ds.train)
    encoded_loo = te_loo.transform(ds.test)
    te_kfold = H2OTargetEncoderEstimator(data_leakage_handling='kfold', noise=0)
    te_kfold.train(y=ds.target, training_frame=ds.train, fold_column='foldc')
    encoded_kfold = te_kfold.transform(ds.test)
    for (l, r) in itertools.combinations([encoded_none, encoded_loo, encoded_kfold], 2):
        assert pu.compare_frames(l, r, 0, tol_numeric=0.01)

def test_use_kfold_strategy_to_train_a_model_with_cv():
    if False:
        return 10
    ds = load_dataset(incl_test=True, incl_foldc=True)
    te = H2OTargetEncoderEstimator(noise=0, data_leakage_handling='kfold')
    te.train(y=ds.target, training_frame=ds.train, fold_column='foldc')
    train_enc_cv = te.transform(ds.train, as_training=True)
    cols_to_remove = [n[:-3] for n in train_enc_cv.names if n.endswith('_te')]
    train_enc_cv = h2o.assign(train_enc_cv.drop(cols_to_remove), 'train_enc_cv')
    train_enc_no_cv = te.transform(ds.train)
    train_enc_no_cv = h2o.assign(train_enc_no_cv.drop(cols_to_remove), 'train_enc_no_cv')
    test_enc = te.transform(ds.test)
    test_enc = h2o.assign(test_enc.drop(cols_to_remove), 'test_enc')
    print(train_enc_cv)
    print(train_enc_no_cv)
    gbm = H2OGradientBoostingEstimator(seed=seed)
    gbm.train(y=ds.target, training_frame=train_enc_cv, fold_column='foldc')
    auc_with_ccv = gbm.model_performance(test_enc).auc()
    print('AUC with CCV : %s' % auc_with_ccv)
    gbm.train(y=ds.target, training_frame=train_enc_no_cv, fold_column='foldc')
    auc_no_ccv = gbm.model_performance(test_enc).auc()
    print('AUC without CCV : %s' % auc_no_ccv)
    assert auc_with_ccv > auc_no_ccv
pu.run_tests([test_default_strategy_is_none, test_fails_on_unknown_strategy, test_kfold_requires_fold_column, test_loo_requires_target_to_encode_training_frame, test_strategies_produce_different_results_for_training, test_use_kfold_strategy_to_train_a_model_with_cv])