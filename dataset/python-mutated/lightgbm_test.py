import sys
import pytest
pytest.importorskip('lightgbm')
import numpy as np
import lightgbm as lgb
import vaex.ml.lightgbm
from vaex.utils import _ensure_strings_from_expressions
params = {'learning_rate': 0.1, 'max_depth': 1, 'colsample_bytree': 0.8, 'subsample': 0.8, 'reg_lambda': 1, 'reg_alpha': 0, 'min_child_weight': 1, 'objective': 'softmax', 'num_class': 3, 'random_state': 42, 'n_jobs': -1}
params_reg = {'learning_rate': 0.1, 'max_depth': 3, 'colsample_bytree': 0.8, 'subsample': 0.8, 'reg_lambda': 1, 'reg_alpha': 0, 'min_child_weight': 1, 'objective': 'regression', 'random_state': 42, 'n_jobs': -1}

@pytest.mark.skipif(sys.version_info < (3, 6), reason='requires python3.6 or higher')
def test_light_gbm_virtual_columns(df_iris):
    if False:
        for i in range(10):
            print('nop')
    ds = df_iris
    ds['x'] = ds.sepal_length * 1
    ds['y'] = ds.sepal_width * 1
    ds['w'] = ds.petal_length * 1
    ds['z'] = ds.petal_width * 1
    (ds_train, ds_test) = ds.ml.train_test_split(test_size=0.2, verbose=False)
    features = ['x', 'y', 'z', 'w']
    booster = vaex.ml.lightgbm.LightGBMModel(num_boost_round=10, params=params, features=features, target='class_')
    booster.fit(ds_train)

@pytest.mark.skipif(sys.version_info < (3, 6), reason='requires python3.6 or higher')
def test_lightgbm(df_iris):
    if False:
        return 10
    ds = df_iris
    (ds_train, ds_test) = ds.ml.train_test_split(test_size=0.2, verbose=False)
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    features = _ensure_strings_from_expressions(features)
    booster = vaex.ml.lightgbm.LightGBMModel(num_boost_round=10, params=params, features=features, target='class_')
    booster.fit(ds_train)
    class_predict_train = booster.predict(ds_train)
    class_predict_test = booster.predict(ds_test)
    assert np.all(ds_test.col.class_.values == np.argmax(class_predict_test, axis=1))
    ds_train = booster.transform(ds_train)
    state = ds_train.state_get()
    ds_test.state_set(state)
    assert np.all(ds_test.col.class_.values == np.argmax(ds_test.lightgbm_prediction.values, axis=1))

@pytest.mark.skipif(sys.version_info < (3, 6), reason='requires python3.6 or higher')
def test_lightgbm_serialize(tmpdir, df_iris):
    if False:
        return 10
    ds = df_iris
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    target = 'class_'
    gbm = ds.ml.lightgbm_model(target=target, features=features, num_boost_round=100, params=params, transform=False)
    pl = vaex.ml.Pipeline([gbm])
    pl.save(str(tmpdir.join('test.json')))
    pl.load(str(tmpdir.join('test.json')))
    gbm = ds.ml.lightgbm_model(target=target, features=features, num_boost_round=100, params=params, transform=False)
    gbm.state_set(gbm.state_get())
    pl = vaex.ml.Pipeline([gbm])
    pl.save(str(tmpdir.join('test.json')))
    pl.load(str(tmpdir.join('test.json')))

@pytest.mark.skipif(sys.version_info < (3, 6), reason='requires python3.6 or higher')
def test_lightgbm_numerical_validation(df_iris):
    if False:
        print('Hello World!')
    ds = df_iris
    features = ['sepal_width', 'petal_length', 'sepal_length', 'petal_width']
    X = np.array(ds[features])
    dtrain = lgb.Dataset(X, label=ds.class_.to_numpy())
    lgb_bst = lgb.train(params, dtrain, 3)
    lgb_pred = lgb_bst.predict(X)
    booster = ds.ml.lightgbm_model(target=ds.class_, num_boost_round=3, features=features, params=params, transform=False)
    vaex_pred = booster.predict(ds)
    np.testing.assert_equal(vaex_pred, lgb_pred, verbose=True, err_msg='The predictions of vaex.ml do not match those of lightgbm')

@pytest.mark.skipif(sys.version_info < (3, 6), reason='requires python3.6 or higher')
def test_lightgbm_validation_set(df_example):
    if False:
        while True:
            i = 10
    ds = df_example
    (train, test) = ds.ml.train_test_split(verbose=False)
    features = ['vx', 'vy', 'vz', 'Lz', 'L']
    history = {}
    booster = vaex.ml.lightgbm.LightGBMModel(features=features, target='E', num_boost_round=10, params=params_reg)
    booster.fit(train, valid_sets=[train, test], valid_names=['train', 'test'], early_stopping_rounds=2, evals_result=history)
    assert booster.booster.best_iteration == 10
    assert len(history['train']['l2']) == 10
    assert len(history['test']['l2']) == 10
    booster.fit(train, valid_sets=[train, test], valid_names=['train', 'test'], early_stopping_rounds=2, evals_result=history)
    assert booster.booster.best_iteration == 10
    assert len(history['train']['l2']) == 10
    assert len(history['test']['l2']) == 10

@pytest.mark.skipif(sys.version_info < (3, 6), reason='requires python3.6 or higher')
def test_lightgbm_pipeline(df_example):
    if False:
        for i in range(10):
            print('nop')
    ds = df_example
    (train, test) = ds.ml.train_test_split(verbose=False)
    train['r'] = np.sqrt(train.x ** 2 + train.y ** 2 + train.z ** 2)
    features = ['vx', 'vy', 'vz', 'Lz', 'L']
    pca = train.ml.pca(n_components=3, features=features, transform=False)
    train = pca.transform(train)
    st = train.ml.state_transfer()
    features = ['r', 'PCA_0', 'PCA_1', 'PCA_2']
    booster = train.ml.lightgbm_model(target='E', num_boost_round=10, features=features, params=params_reg, transform=False)
    pp = vaex.ml.Pipeline([st, booster])
    pred = pp.predict(test)
    trans = pp.transform(test)
    np.testing.assert_equal(pred, trans.evaluate('lightgbm_prediction'), verbose=True, err_msg='The predictions from the fit and transform method do not match')