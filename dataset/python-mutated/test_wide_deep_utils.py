import os
import pytest
import pandas as pd
from recommenders.utils.constants import DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL
try:
    from recommenders.utils.tf_utils import pandas_input_fn, MODEL_DIR
    from recommenders.models.wide_deep.wide_deep_utils import build_model, build_feature_columns
    import tensorflow as tf
except ImportError:
    pass
ITEM_FEAT_COL = 'itemFeat'

@pytest.fixture(scope='module')
def pd_df():
    if False:
        for i in range(10):
            print('nop')
    df = pd.DataFrame({DEFAULT_USER_COL: [1, 1, 1, 2, 2, 2], DEFAULT_ITEM_COL: [1, 2, 3, 1, 4, 5], ITEM_FEAT_COL: [[1, 1, 1], [2, 2, 2], [3, 3, 3], [1, 1, 1], [4, 4, 4], [5, 5, 5]], DEFAULT_RATING_COL: [5, 4, 3, 5, 5, 3]})
    users = df.drop_duplicates(DEFAULT_USER_COL)[DEFAULT_USER_COL].values
    items = df.drop_duplicates(DEFAULT_ITEM_COL)[DEFAULT_ITEM_COL].values
    return (df, users, items)

@pytest.mark.gpu
def test_wide_model(pd_df, tmp):
    if False:
        for i in range(10):
            print('nop')
    (data, users, items) = pd_df
    (wide_columns, _) = build_feature_columns(users, items, model_type='wide', crossed_feat_dim=10)
    assert len(wide_columns) == 3
    assert wide_columns[2].hash_bucket_size == 10
    model = build_model(os.path.join(tmp, 'wide_' + MODEL_DIR), wide_columns=wide_columns)
    assert isinstance(model, tf.compat.v1.estimator.LinearRegressor)
    model.train(input_fn=pandas_input_fn(df=data, y_col=DEFAULT_RATING_COL, batch_size=1, num_epochs=None, shuffle=True), steps=1)
    summary_writer = tf.compat.v1.summary.FileWriterCache.get(model.model_dir)
    summary_writer.close()

@pytest.mark.gpu
def test_deep_model(pd_df, tmp):
    if False:
        i = 10
        return i + 15
    (data, users, items) = pd_df
    (_, deep_columns) = build_feature_columns(users, items, model_type='deep')
    assert len(deep_columns) == 2
    model = build_model(os.path.join(tmp, 'deep_' + MODEL_DIR), deep_columns=deep_columns)
    assert isinstance(model, tf.compat.v1.estimator.DNNRegressor)
    model.train(input_fn=pandas_input_fn(df=data, y_col=DEFAULT_RATING_COL, batch_size=1, num_epochs=1, shuffle=False))
    summary_writer = tf.compat.v1.summary.FileWriterCache.get(model.model_dir)
    summary_writer.close()

@pytest.mark.gpu
def test_wide_deep_model(pd_df, tmp):
    if False:
        for i in range(10):
            print('nop')
    (data, users, items) = pd_df
    (wide_columns, deep_columns) = build_feature_columns(users, items, model_type='wide_deep')
    assert len(wide_columns) == 3
    assert len(deep_columns) == 2
    model = build_model(os.path.join(tmp, 'wide_deep_' + MODEL_DIR), wide_columns=wide_columns, deep_columns=deep_columns)
    assert isinstance(model, tf.compat.v1.estimator.DNNLinearCombinedRegressor)
    model.train(input_fn=pandas_input_fn(df=data, y_col=DEFAULT_RATING_COL, batch_size=1, num_epochs=None, shuffle=True), steps=1)
    summary_writer = tf.compat.v1.summary.FileWriterCache.get(model.model_dir)
    summary_writer.close()