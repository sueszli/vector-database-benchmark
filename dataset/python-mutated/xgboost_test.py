import pytest
import numpy as np
pytest.importorskip('xgboost')
import xgboost as xgb
import vaex.ml.xgboost
params_multiclass = {'learning_rate': 0.1, 'max_depth': 3, 'colsample_bytree': 0.8, 'subsample': 0.8, 'reg_lambda': 1, 'reg_alpha': 0, 'min_child_weight': 1, 'objective': 'multi:softmax', 'num_class': 3, 'random_state': 42, 'silent': 1, 'n_jobs': -1}
params_reg = {'learning_rate': 0.1, 'max_depth': 3, 'colsample_bytree': 0.8, 'subsample': 0.8, 'reg_lambda': 1, 'reg_alpha': 0, 'min_child_weight': 1, 'objective': 'reg:linear', 'random_state': 42, 'silent': 1, 'n_jobs': -1}

def test_xgboost(df_iris):
    if False:
        i = 10
        return i + 15
    ds = df_iris
    (ds_train, ds_test) = ds.ml.train_test_split(test_size=0.2, verbose=False)
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    booster = vaex.ml.xgboost.XGBoostModel(num_boost_round=10, params=params_multiclass, features=features, target='class_')
    booster.fit(ds_train)
    class_predict = booster.predict(ds_test)
    assert np.all(ds_test.class_.values == class_predict)
    ds_train = booster.transform(ds_train)
    state = ds_train.state_get()
    ds_test.state_set(state)
    assert np.all(ds_test.class_.values == ds_test.xgboost_prediction.values)

def test_xgboost_numerical_validation(df_iris):
    if False:
        print('Hello World!')
    ds = df_iris
    features = ['sepal_width', 'petal_length', 'sepal_length', 'petal_width']
    dtrain = xgb.DMatrix(ds[features].values, label=ds.class_.to_numpy())
    xgb_bst = xgb.train(params=params_multiclass, dtrain=dtrain, num_boost_round=3)
    xgb_pred = xgb_bst.predict(dtrain)
    booster = vaex.ml.xgboost.XGBoostModel(features=features, target='class_', params=params_multiclass, num_boost_round=3)
    booster.fit(ds)
    vaex_pred = booster.predict(ds)
    np.testing.assert_equal(vaex_pred, xgb_pred, verbose=True, err_msg='The predictions of vaex.ml.xboost do not match those of pure xgboost')

def test_xgboost_serialize(tmpdir, df_iris):
    if False:
        return 10
    ds = df_iris
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    target = 'class_'
    gbm = ds.ml.xgboost_model(target=target, features=features, num_boost_round=20, params=params_multiclass, transform=False)
    pl = vaex.ml.Pipeline([gbm])
    pl.save(str(tmpdir.join('test.json')))
    pl.load(str(tmpdir.join('test.json')))
    gbm = ds.ml.xgboost_model(target=target, features=features, num_boost_round=20, params=params_multiclass, transform=False)
    gbm.state_set(gbm.state_get())
    pl = vaex.ml.Pipeline([gbm])
    pl.save(str(tmpdir.join('test.json')))
    pl.load(str(tmpdir.join('test.json')))

def test_xgboost_validation_set(df_example):
    if False:
        return 10
    ds = df_example
    (train, test) = ds.ml.train_test_split(verbose=False)
    features = ['vx', 'vy', 'vz', 'Lz', 'L']
    history = {}
    booster = vaex.ml.xgboost.XGBoostModel(features=features, target='E', num_boost_round=10, params=params_reg)
    booster.fit(train, evals=[(train, 'train'), (test, 'test')], early_stopping_rounds=2, evals_result=history)
    assert booster.booster.best_ntree_limit == 10
    assert booster.booster.best_iteration == 9
    assert len(history['train']['rmse']) == 10
    assert len(history['test']['rmse']) == 10

def test_xgboost_pipeline(df_example):
    if False:
        return 10
    ds = df_example
    (train, test) = ds.ml.train_test_split(verbose=False)
    train['r'] = np.sqrt(train.x ** 2 + train.y ** 2 + train.z ** 2)
    features = ['vx', 'vy', 'vz', 'Lz', 'L']
    pca = train.ml.pca(n_components=3, features=features, transform=False)
    train = pca.transform(train)
    st = train.ml.state_transfer()
    features = ['r', 'PCA_0', 'PCA_1', 'PCA_2']
    booster = train.ml.xgboost_model(target='E', num_boost_round=10, features=features, params=params_reg, transform=False)
    pp = vaex.ml.Pipeline([st, booster])
    pred = pp.predict(test)
    trans = pp.transform(test)
    np.testing.assert_equal(pred, trans.evaluate('xgboost_prediction'), verbose=True, err_msg='The predictions from the predict and transform method do not match')