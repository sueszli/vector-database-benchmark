#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import pytest

from bigdl.orca.automl.xgboost import AutoXGBRegressor

import numpy as np
from unittest import TestCase
from bigdl.orca.automl import hp


def get_data():
    def get_x_y(size):
        x = np.random.randn(size, 2)
        y = np.random.randn(size)
        return x, y
    train_x, train_y = get_x_y(1000)
    val_x, val_y = get_x_y(400)
    data = (train_x, train_y)
    validation_data = (val_x, val_y)
    return data, validation_data


def get_data_creators():
    def get_x_y(size, config):
        import pandas as pd
        values = np.random.randn(size, 4)
        df = pd.DataFrame(values, columns=["f1", "f2", "f3", "t"])
        selected_features = config["features"]
        x = df[selected_features].to_numpy()
        y = df["t"].to_numpy()
        return x, y

    from functools import partial
    train_data_creator = partial(get_x_y, 1000)
    val_data_creator = partial(get_x_y, 400)
    return train_data_creator, val_data_creator


def get_spark_df():
    def get_df(size):
        import pandas as pd
        values = np.random.randn(size, 4)
        df = pd.DataFrame(values, columns=["f1", "f2", "f3", "t"])
        spark_df = spark.createDataFrame(df)
        return spark_df

    from bigdl.orca import OrcaContext
    from pyspark.sql import SparkSession
    sc = OrcaContext.get_spark_context()
    spark = SparkSession(sc)

    feature_cols = ["f1", "f2", "f3"]
    label_cols = ["t"]
    train_df = get_df(size=100)
    val_df = get_df(size=30)
    return train_df, val_df, feature_cols, label_cols


def get_xgb_search_space():
    return {
        "n_estimators": hp.randint(5, 10),
        "max_depth": hp.randint(2, 5),
        "lr": hp.loguniform(1e-4, 1e-1),
    }


class TestAutoXGBRegressor(TestCase):
    def setUp(self) -> None:
        from bigdl.orca import init_orca_context
        init_orca_context(cores=8, init_ray_on_spark=True)

    def tearDown(self) -> None:
        from bigdl.orca import stop_orca_context
        stop_orca_context()

    def test_fit(self):
        auto_xgb_reg = AutoXGBRegressor(cpus_per_trial=2,
                                        name="auto_xgb_regressor",
                                        tree_method='hist')
        data, validation_data = get_data()
        auto_xgb_reg.fit(data=data,
                         validation_data=validation_data,
                         search_space=get_xgb_search_space(),
                         n_sampling=4,
                         epochs=1,
                         metric="mae")
        best_model = auto_xgb_reg.get_best_model()
        assert 5 <= best_model.n_estimators <= 10
        assert 2 <= best_model.max_depth <= 5
        best_config = auto_xgb_reg.get_best_config()
        assert all(k in best_config.keys() for k in get_xgb_search_space().keys())

    def test_metric(self):
        auto_xgb_reg = AutoXGBRegressor(cpus_per_trial=2,
                                        name="auto_xgb_regressor",
                                        tree_method='hist')
        data, validation_data = get_data()
        with pytest.raises(RuntimeError) as exeinfo:
            auto_xgb_reg.fit(data=data,
                             epochs=1,
                             validation_data=validation_data,
                             metric="logloss",
                             search_space=get_xgb_search_space(),
                             n_sampling=4)
        assert "metric logloss" in str(exeinfo)
        auto_xgb_reg.fit(data=data,
                         epochs=1,
                         validation_data=validation_data,
                         metric="logloss",
                         metric_mode="min",
                         search_space=get_xgb_search_space(),
                         n_sampling=4)

    def test_metric_func(self):
        auto_xgb_reg = AutoXGBRegressor(cpus_per_trial=2,
                                        name="auto_xgb_regressor",
                                        tree_method='hist')
        data, validation_data = get_data()

        def pyrmsle(y_true, y_pred):
            y_pred[y_pred < -1] = -1 + 1e-6
            elements = np.power(np.log1p(y_true) - np.log1p(y_pred), 2)
            return float(np.sqrt(np.sum(elements) / len(y_true)))

        with pytest.raises(RuntimeError) as exeinfo:
            auto_xgb_reg.fit(data=data,
                             epochs=1,
                             validation_data=validation_data,
                             metric=pyrmsle,
                             search_space=get_xgb_search_space(),
                             n_sampling=4)
        assert "metric_mode" in str(exeinfo)

        auto_xgb_reg.fit(data=data,
                         epochs=1,
                         validation_data=validation_data,
                         metric=pyrmsle,
                         metric_mode="min",
                         search_space=get_xgb_search_space(),
                         n_sampling=4)

    def test_data_creator(self):
        train_data_creator, val_data_creator = get_data_creators()
        auto_xgb_reg = AutoXGBRegressor(cpus_per_trial=2,
                                        name="auto_xgb_regressor",
                                        tree_method='hist')
        model_search_space = get_xgb_search_space()
        # todo: change to hp.choice_n
        search_space = {"features": hp.sample_from(lambda spec:
                                                   np.random.choice(["f1", "f2", "f3"], size=2))}
        search_space.update(model_search_space)
        auto_xgb_reg.fit(data=train_data_creator,
                         epochs=1,
                         validation_data=val_data_creator,
                         metric="logloss",
                         metric_mode="min",
                         search_space=search_space,
                         n_sampling=2)
        best_config = auto_xgb_reg.get_best_config()
        assert all(k in best_config.keys() for k in search_space.keys())
        assert len(best_config["features"]) == 2

    def test_spark_df(self):
        df, val_df, feature_cols, label_cols = get_spark_df()
        auto_xgb_reg = AutoXGBRegressor(cpus_per_trial=2,
                                        name="auto_xgb_regressor",
                                        tree_method='hist')
        search_space = get_xgb_search_space()
        auto_xgb_reg.fit(data=df,
                         epochs=1,
                         validation_data=val_df,
                         metric="logloss",
                         metric_mode="min",
                         search_space=search_space,
                         n_sampling=2,
                         feature_cols=feature_cols,
                         label_cols=label_cols,
                         )
        best_model = auto_xgb_reg.get_best_model()
        assert 5 <= best_model.n_estimators <= 10
        assert 2 <= best_model.max_depth <= 5
        best_config = auto_xgb_reg.get_best_config()
        assert all(k in best_config.keys() for k in search_space.keys())


if __name__ == "__main__":
    pytest.main([__file__])
