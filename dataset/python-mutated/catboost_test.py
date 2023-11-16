import pytest
pytest.importorskip('catboost')
import catboost as cb
import numpy as np
import vaex.ml.catboost
import vaex.datasets
from sklearn.metrics import roc_auc_score, accuracy_score
params_multiclass = {'leaf_estimation_method': 'Gradient', 'learning_rate': 0.1, 'max_depth': 3, 'bootstrap_type': 'Bernoulli', 'subsample': 0.8, 'sampling_frequency': 'PerTree', 'colsample_bylevel': 0.8, 'reg_lambda': 1, 'objective': 'MultiClass', 'eval_metric': 'MultiClass', 'random_state': 42, 'verbose': 0}
params_reg = {'leaf_estimation_method': 'Gradient', 'learning_rate': 0.1, 'max_depth': 3, 'bootstrap_type': 'Bernoulli', 'subsample': 0.8, 'sampling_frequency': 'PerTree', 'colsample_bylevel': 0.8, 'reg_lambda': 1, 'objective': 'MAE', 'eval_metric': 'R2', 'random_state': 42, 'verbose': 0}

def test_catboost(df_iris):
    if False:
        print('Hello World!')
    ds = df_iris
    (ds_train, ds_test) = ds.ml.train_test_split(test_size=0.2, verbose=False)
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    booster = vaex.ml.catboost.CatBoostModel(num_boost_round=10, params=params_multiclass, features=features, target='class_', prediction_type='Probability')
    booster.fit(ds_train)
    class_predict = booster.predict(ds_test)
    assert np.all(ds_test.col.class_.values == np.argmax(class_predict, axis=1))
    ds_train = booster.transform(ds_train)
    state = ds_train.state_get()
    ds_test.state_set(state)
    assert np.all(ds_test.col.class_.values == np.argmax(ds_test.catboost_prediction.values, axis=1))

def test_catboost_batch_training(df_iris):
    if False:
        return 10
    '\n    We train three models. One on 10 samples. the second on 100 samples with batches of 10,\n    and the third too on 100 samples with batches of 10, but we weight the models as if only the first batch matters.\n    A model trained on more data, should do better than the model who only trained on 10 samples,\n    and the weighted model will do exactly as good as the one who trained on 10 samples as it ignore the rest by weighting.\n    '
    ds = df_iris
    (ds_train, ds_test) = ds.ml.train_test_split(test_size=0.2, verbose=False)
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    target = 'class_'
    prediction_type = 'Class'
    vanilla = vaex.ml.catboost.CatBoostModel(num_boost_round=1, params=params_multiclass, features=features, target=target, prediction_type=prediction_type)
    batch_booster = vaex.ml.catboost.CatBoostModel(num_boost_round=1, params=params_multiclass, features=features, target=target, prediction_type=prediction_type, batch_size=10)
    weights = [1.0] + [0.0] * 9
    weights_booster = vaex.ml.catboost.CatBoostModel(num_boost_round=1, params=params_multiclass, features=features, target=target, prediction_type=prediction_type, batch_size=10, batch_weights=weights)
    vanilla.fit(ds_train.head(10), evals=[ds_test])
    batch_booster.fit(ds_train.head(100), evals=[ds_test])
    weights_booster.fit(ds_train.head(100), evals=[ds_test])
    ground_truth = ds_test[target].values
    vanilla_accuracy = accuracy_score(ground_truth, vanilla.predict(ds_test))
    batch_accuracy = accuracy_score(ground_truth, batch_booster.predict(ds_test))
    weighted_accuracy = accuracy_score(ground_truth, weights_booster.predict(ds_test))
    assert vanilla_accuracy == weighted_accuracy
    assert vanilla_accuracy < batch_accuracy
    assert list(weights_booster.booster.get_feature_importance()) == list(vanilla.booster.get_feature_importance())
    assert list(weights_booster.booster.get_feature_importance()) != list(batch_booster.booster.get_feature_importance())

def test_catboost_numerical_validation(df_iris):
    if False:
        return 10
    ds = df_iris
    features = ['sepal_width', 'petal_length', 'sepal_length', 'petal_width']
    dtrain = cb.Pool(ds[features].values, label=ds.class_.to_numpy())
    cb_bst = cb.train(params=params_multiclass, dtrain=dtrain, num_boost_round=3)
    cb_pred = cb_bst.predict(dtrain, prediction_type='Probability')
    booster = vaex.ml.catboost.CatBoostModel(features=features, target='class_', params=params_multiclass, num_boost_round=3)
    booster.fit(ds)
    vaex_pred = booster.predict(ds)
    np.testing.assert_equal(vaex_pred, cb_pred, verbose=True, err_msg='The predictions of vaex.ml.catboost do not match those of pure catboost')

def test_lightgbm_serialize(tmpdir, df_iris):
    if False:
        i = 10
        return i + 15
    ds = df_iris
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    target = 'class_'
    gbm = ds.ml.catboost_model(target=target, features=features, num_boost_round=20, params=params_multiclass, transform=False)
    pl = vaex.ml.Pipeline([gbm])
    pl.save(str(tmpdir.join('test.json')))
    pl.load(str(tmpdir.join('test.json')))
    gbm = ds.ml.catboost_model(target=target, features=features, num_boost_round=20, params=params_multiclass, transform=False)
    gbm.state_set(gbm.state_get())
    pl = vaex.ml.Pipeline([gbm])
    pl.save(str(tmpdir.join('test.json')))
    pl.load(str(tmpdir.join('test.json')))

def test_catboost_validation_set():
    if False:
        for i in range(10):
            print('nop')
    ds = vaex.example()
    (train, test) = ds.ml.train_test_split(verbose=False)
    features = ['vx', 'vy', 'vz', 'Lz', 'L']
    booster = vaex.ml.catboost.CatBoostModel(features=features, target='E', num_boost_round=10, params=params_reg)
    booster.fit(train, evals=[train, test])
    assert hasattr(booster, 'booster')
    assert len(booster.booster.evals_result_['learn']['MAE']) == 10
    assert len(booster.booster.evals_result_['learn']['R2']) == 10
    assert len(booster.booster.evals_result_['validation_0']['MAE']) == 10
    assert len(booster.booster.evals_result_['validation_0']['R2']) == 10
    assert hasattr(booster.booster, 'best_iteration_')
    assert booster.booster.best_iteration_ is not None

def test_catboost_pipeline(df_example):
    if False:
        while True:
            i = 10
    ds = df_example
    (train, test) = ds.ml.train_test_split(verbose=False)
    train['r'] = np.sqrt(train.x ** 2 + train.y ** 2 + train.z ** 2)
    features = ['vx', 'vy', 'vz', 'Lz', 'L']
    pca = train.ml.pca(n_components=3, features=features, transform=False)
    train = pca.transform(train)
    st = train.ml.state_transfer()
    features = ['r', 'PCA_0', 'PCA_1', 'PCA_2']
    booster = train.ml.catboost_model(target='E', num_boost_round=10, features=features, params=params_reg, transform=False)
    pp = vaex.ml.Pipeline([st, booster])
    pred = pp.predict(test)
    trans = pp.transform(test)
    np.testing.assert_equal(pred, trans.evaluate('catboost_prediction'), verbose=True, err_msg='The predictions from the predict and transform method do not match')