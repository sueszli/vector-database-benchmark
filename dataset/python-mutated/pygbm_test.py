import pytest
pytest.importorskip('pygbm')
import os
import numpy as np
import pygbm as lgb
import vaex.ml.incubator.pygbm
from vaex.utils import _ensure_strings_from_expressions
import test_utils
param = {'learning_rate': 0.1, 'max_depth': 1, 'colsample_bytree': 0.8, 'subsample': 0.8, 'reg_lambda': 1, 'reg_alpha': 0, 'min_child_weight': 1, 'objective': 'softmax', 'num_class': 3, 'random_state': 42, 'n_jobs': -1}

@test_utils.skip_incubator
def test_py_gbm_virtual_columns():
    if False:
        return 10
    ds = vaex.datasets.iris()
    ds['x'] = ds.sepal_length * 1
    ds['y'] = ds.sepal_width * 1
    ds['w'] = ds.petal_length * 1
    ds['z'] = ds.petal_width * 1
    (ds_train, ds_test) = ds.ml.train_test_split(test_size=0.2, verbose=False)
    features = ['x', 'y', 'z', 'w']
    booster = vaex.ml.incubator.pygbm.PyGBMModel(num_round=10, param=param, features=_ensure_strings_from_expressions(features))
    booster.fit(ds_train, ds_train.class_)

@test_utils.skip_incubator
def test_pygbm():
    if False:
        for i in range(10):
            print('nop')
    for filename in 'blah.col.meta blah.col.page'.split():
        if os.path.exists(filename):
            os.remove(filename)
    ds = vaex.datasets.iris()
    (ds_train, ds_test) = ds.ml.train_test_split(test_size=0.2, verbose=False)
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    booster = vaex.ml.incubator.pygbm.PyGBMModel(num_round=10, param=param, features=_ensure_strings_from_expressions(features))
    booster.fit(ds_train, ds_train.class_)
    class_predict = booster.predict(ds_test)
    booster.fit(ds_train, ds_train.class_)
    class_predict = booster.predict(ds_test)
    assert np.all(ds.col.class_ == class_predict)
    ds = booster.transform(ds)
    state = ds.state_get()
    ds = vaex.datasets.iris()
    ds.state_set(state)
    assert np.all(ds.col.class_ == ds.evaluate(ds.pygbm_prediction))

@test_utils.skip_incubator
def test_pygbm_serialize(tmpdir):
    if False:
        while True:
            i = 10
    ds = vaex.datasets.iris()
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    target = 'class_'
    gbm = ds.ml_pygbm_model(target, 20, features=features, param=param, classifier=False)
    pl = vaex.ml.Pipeline([gbm])
    pl.save(str(tmpdir.join('test.json')))
    pl.load(str(tmpdir.join('test.json')))
    gbm = ds.ml_pygbm_model(target, 20, features=features, param=param, classifier=True)
    gbm.state_set(gbm.state_get())
    pl = vaex.ml.Pipeline([gbm])
    pl.save(str(tmpdir.join('test.json')))
    pl.load(str(tmpdir.join('test.json')))

@test_utils.skip_incubator
def test_pygbm_invalid():
    if False:
        while True:
            i = 10
    for filename in 'blah.col.meta blah.col.page'.split():
        if os.path.exists(filename):
            os.remove(filename)
    ds = vaex.ml.iris()
    features = ['sepal_length', 'sepal_width', 'petal_length', 'wrong']
    booster = vaex.ml.pygbm.XGBModel(num_round=10, param=param, features=vaex.dataset._ensure_strings_from_expressions(features))
    booster.fit(ds, ds.class_)

@test_utils.skip_incubator
def test_pygbm_validation():
    if False:
        return 10
    ds = vaex.ml.iris()
    features = ['sepal_width', 'petal_length', 'sepal_length', 'petal_width']
    X = np.array(ds[features])
    lgb_bst = lgb.train(param, dtrain, 3)
    lgb_pred = np.argmax(lgb_bst.predict(X), axis=1)
    booster = ds.ml.pygbm_model(label='class_', num_round=3, features=features, param=param, classifier=True)
    vaex_pred = booster.predict(ds)
    np.testing.assert_equal(vaex_pred, lgb_pred, verbose=True, err_msg='The predictions of vaex.ml do not match those of pygbm')

@test_utils.skip_incubator
def test_pygbm_pipeline():
    if False:
        return 10
    param = {'learning_rate': 0.1, 'max_depth': 5, 'colsample_bytree': 0.8, 'subsample': 0.8, 'reg_lambda': 1, 'reg_alpha': 0, 'min_child_weight': 1, 'objective': 'regression', 'random_state': 42, 'silent': 1, 'n_jobs': -1}
    ds = vaex.example()
    (train, test) = ds.ml.train_test_split(verbose=False)
    train['r'] = np.sqrt(train.x ** 2 + train.y ** 2 + train.z ** 2)
    features = ['vx', 'vy', 'vz', 'Lz', 'L']
    pca = train.ml.pca(n_components=3, features=features)
    train = pca.transform(train)
    st = vaex.ml.state_transfer(train)
    features = ['r', 'PCA_0', 'PCA_1', 'PCA_2']
    booster = train.ml.pygbm_model(label='E', max_iter=10, features=features, param=param)
    pp = vaex.ml.Pipeline([st, booster])
    pred = pp.predict(test)
    trans = pp.transform(test)
    np.testing.assert_equal(pred, trans.evaluate('pygbm_prediction'), verbose=True, err_msg='The predictions from the fit and transform method do not match')