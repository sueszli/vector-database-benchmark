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
import os
from unittest import TestCase

import numpy as np
import pytest
import logging
import math

import torch
import torch.nn as nn

from pyspark.sql import SparkSession
from pyspark.sql.types import FloatType, ArrayType, StructType, StructField

from bigdl.orca import OrcaContext
from bigdl.orca.data.pandas import read_csv
from bigdl.orca.learn.metrics import Accuracy

from bigdl.dllib.nncontext import init_nncontext
from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca.data import SparkXShards
from bigdl.orca.data.image.utils import chunks

import tempfile
import shutil

from bigdl.orca.learn.pytorch.callbacks.base import Callback

np.random.seed(1337)  # for reproducibility
resource_path = os.path.join(
    os.path.realpath(os.path.dirname(__file__)), "../../../resources")


class LinearDataset(torch.utils.data.Dataset):
    """y = a * x + b"""

    def __init__(self, size=1000):
        X1 = torch.randn(size // 2, 50)
        X2 = torch.randn(size // 2, 50) + 1.5
        self.x = torch.cat([X1, X2], dim=0)
        Y1 = torch.zeros(size // 2, 1)
        Y2 = torch.ones(size // 2, 1)
        self.y = torch.cat([Y1, Y2], dim=0)

    def __getitem__(self, index):
        return self.x[index, None], self.y[index, None]

    def __len__(self):
        return len(self.x)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(50, 50)
        self.relu1 = nn.ReLU()
        self.dout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, 100)
        self.prelu = nn.PReLU(1)
        self.out = nn.Linear(100, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        dout = self.dout(h1)
        a2 = self.fc2(dout)
        h2 = self.prelu(a2)
        a3 = self.out(h2)
        y = self.out_act(a3)
        return y


class IdentityNet(nn.Module):
    def __init__(self):
        super().__init__()
        # need this line to avoid optimizer raise empty variable list
        self.fc1 = nn.Linear(50, 50)

    def forward(self, input_):
        return input_


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        # need this line to avoid optimizer raise empty variable list
        self.fc1 = nn.Linear(1, 1, bias=False)
        self.fc1.weight.data.fill_(1.0)

    def forward(self, input_):
        return self.fc1(input_)


class MultiInputNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(50, 50)
        self.out = nn.Linear(50, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, input1, input2):
        x = torch.cat((input1, input2), 1)
        x = self.fc1(x)
        x = self.out(x)
        x = self.out_act(x)
        return x


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, input1, input2):
        x = torch.stack((input1, input2), dim=1)
        x = self.fc(x)
        x = self.out_act(x).flatten()
        return x


class CustomCallback(Callback):

    def on_train_end(self, logs=None):
        assert "train_loss" in logs
        assert "val_loss" in logs
        assert self.model

    def on_epoch_end(self, epoch, logs=None):
        assert "train_loss" in logs
        assert "val_loss" in logs
        assert self.model


def train_data_loader(config, batch_size):
    train_dataset = LinearDataset(size=config.get("data_size", 1000))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size
    )
    return train_loader


def val_data_loader(config, batch_size):
    val_dataset = LinearDataset(size=config.get("val_size", 400))
    validation_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size
    )
    return validation_loader


def get_model(config):
    torch.manual_seed(0)
    return Net()


def get_optimizer(model, config):
    return torch.optim.SGD(model.parameters(), lr=config.get("lr", 1e-2))


def get_zero_optimizer(model, config):
    return torch.optim.SGD(model.parameters(), lr=0.0)


def customized_metric(pred, target):
    return torch.sum((pred - target) ** 4)


def get_estimator(workers_per_node=1, model_fn=get_model, sync_stats=False,
                  log_level=logging.INFO, loss=nn.BCELoss(), optimizer=get_optimizer, metrics=Accuracy()):
    estimator = Estimator.from_torch(model=model_fn,
                                     optimizer=optimizer,
                                     loss=loss,
                                     metrics=metrics,
                                     config={"lr": 1e-2},
                                     workers_per_node=workers_per_node,
                                     backend="ray",
                                     sync_stats=sync_stats,
                                     log_level=log_level)
    return estimator


class TestPyTorchEstimator(TestCase):
    def test_data_creator(self):
        estimator = get_estimator(workers_per_node=2)
        start_val_stats = estimator.evaluate(val_data_loader, batch_size=64)
        print(start_val_stats)
        train_stats = estimator.fit(train_data_loader, epochs=4, batch_size=128,
                                    validation_data=val_data_loader)
        print(train_stats)
        # 1000 // 2 is the data size for each worker
        # batch_count is the average batches of all workers.
        # In this unit test, two workers have the same data size.
        assert train_stats[0]["batch_count"] == math.ceil(1000 // 2 / (128 // 2))
        assert "val_loss" in train_stats[0]
        end_val_stats = estimator.evaluate(val_data_loader, batch_size=64)
        print(end_val_stats)
        assert 0 < end_val_stats["Accuracy"] < 1
        assert estimator.get_model()

        # sanity check that training worked
        dloss = end_val_stats["val_loss"] - start_val_stats["val_loss"]
        dacc = (end_val_stats["Accuracy"] -
                start_val_stats["Accuracy"])
        print(f"dLoss: {dloss}, dAcc: {dacc}")

        assert dloss < 0 < dacc, "training sanity check failed. loss increased!"
        # Verify syncing weights, i.e. the two workers have the same weights after training
        import ray
        remote_workers = estimator.remote_workers
        state_dicts = ray.get([worker.get_state_dict.remote() for worker in remote_workers])
        weights = [state["models"] for state in state_dicts]
        worker1_weights = weights[0][0]
        worker2_weights = weights[1][0]
        for layer in list(worker1_weights.keys()):
            assert np.allclose(worker1_weights[layer].numpy(),
                               worker2_weights[layer].numpy())
        estimator.shutdown()

    def test_spark_xshards(self):
        from bigdl.orca import OrcaContext
        from bigdl.orca.data import SparkXShards
        estimator = get_estimator(workers_per_node=1)
        sc = OrcaContext.get_spark_context()
        x_rdd = sc.parallelize(np.random.rand(4000, 1, 50).astype(np.float32))
        # torch 1.7.1+ requires target size same as output size, which is (batch, 1)
        y_rdd = sc.parallelize(np.random.randint(0, 2, size=(4000, 1, 1)).astype(np.float32))
        rdd = x_rdd.zip(y_rdd).map(lambda x_y: {'x': x_y[0], 'y': x_y[1]})
        train_rdd, val_rdd = rdd.randomSplit([0.9, 0.1])
        train_xshards = SparkXShards(train_rdd)
        val_xshards = SparkXShards(val_rdd)
        train_stats = estimator.fit(train_xshards, validation_data=val_xshards,
                                    batch_size=256, epochs=2)
        assert "val_loss" in train_stats[0]
        print(train_stats)
        val_stats = estimator.evaluate(val_xshards, batch_size=128)
        print(val_stats)
        estimator.shutdown()

    def test_dataframe_train_eval(self):

        sc = init_nncontext()
        spark = SparkSession.builder.getOrCreate()
        rdd = sc.range(0, 100)
        data = rdd.map(lambda x: (np.random.randn(50).astype(np.float32).tolist(),
                                  [float(np.random.randint(0, 2, size=()))])
                       )
        schema = StructType([
            StructField("feature", ArrayType(FloatType()), True),
            StructField("label", ArrayType(FloatType()), True)
        ])

        df = spark.createDataFrame(data=data, schema=schema)

        val_rdd = sc.range(0, 40)
        val_data = val_rdd.map(lambda x: (np.random.randn(50).astype(np.float32).tolist(),
                                          [float(np.random.randint(0, 2, size=()))])
                               )
        val_df = spark.createDataFrame(data=val_data, schema=schema)

        estimator = get_estimator(workers_per_node=2)
        train_worker_stats = estimator.fit(df, batch_size=4, epochs=2,
                                           validation_data=val_df,
                                           feature_cols=["feature"],
                                           label_cols=["label"])
        assert train_worker_stats[0]["num_samples"] == 100
        eval_worker_stats = estimator.evaluate(val_df, batch_size=4,
                                               feature_cols=["feature"],
                                               label_cols=["label"],
                                               reduce_results=False, profile=True)
        acc = [stat["Accuracy"].data.item() for stat in eval_worker_stats]
        loss = [stat["val_loss"] for stat in eval_worker_stats]
        validation_time = [stat["profile"]["mean_validation_s"] for stat in eval_worker_stats]
        forward_time = [stat["profile"]["mean_eval_fwd_s"] for stat in eval_worker_stats]
        from bigdl.orca.learn.pytorch.utils import process_stats
        agg_worker_stats = process_stats(eval_worker_stats)
        assert round(agg_worker_stats["Accuracy"].data.item(), 4) == \
               round(sum(acc) / 2, 4)
        assert round(agg_worker_stats["val_loss"], 4) == round(sum(loss) / 2, 4)
        assert round(agg_worker_stats["profile"]["mean_validation_s"], 4) == \
               round(sum(validation_time) / 2, 4)
        assert round(agg_worker_stats["profile"]["mean_eval_fwd_s"], 4) == \
               round(sum(forward_time) / 2, 4)
        assert agg_worker_stats["num_samples"] == 40

    def test_dataframe_shard_size_train_eval(self):
        from bigdl.orca import OrcaContext
        OrcaContext._shard_size = 30
        sc = init_nncontext()
        spark = SparkSession.builder.getOrCreate()
        rdd = sc.range(0, 100)
        data = rdd.map(lambda x: (np.random.randn(50).astype(np.float32).tolist(),
                                  [float(np.random.randint(0, 2, size=()))])
                       )
        schema = StructType([
            StructField("feature", ArrayType(FloatType()), True),
            StructField("label", ArrayType(FloatType()), True)
        ])

        df = spark.createDataFrame(data=data, schema=schema)

        estimator = get_estimator(workers_per_node=2)
        estimator.fit(df, batch_size=4, epochs=2,
                      feature_cols=["feature"],
                      label_cols=["label"])
        estimator.evaluate(df, batch_size=4,
                           feature_cols=["feature"],
                           label_cols=["label"])

    def test_partition_num_less_than_workers(self):
        sc = init_nncontext()
        spark = SparkSession.builder.getOrCreate()
        rdd = sc.range(200, numSlices=1)
        data = rdd.map(lambda x: (np.random.randn(50).astype(np.float32).tolist(),
                                  [float(np.random.randint(0, 2, size=()))])
                       )
        schema = StructType([
            StructField("feature", ArrayType(FloatType()), True),
            StructField("label", ArrayType(FloatType()), True)
        ])

        df = spark.createDataFrame(data=data, schema=schema)

        estimator = get_estimator(workers_per_node=2)
        assert df.rdd.getNumPartitions() < estimator.num_workers

        estimator.fit(df, batch_size=4, epochs=2,
                      validation_data=df,
                      feature_cols=["feature"],
                      label_cols=["label"])
        estimator.evaluate(df, batch_size=4,
                           feature_cols=["feature"],
                           label_cols=["label"])
        estimator.predict(df, feature_cols=["feature"]).collect()

    def test_dataframe_predict(self):

        sc = init_nncontext()
        rdd = sc.parallelize(range(20))
        df = rdd.map(lambda x: ([float(x)] * 5,
                                [int(np.random.randint(0, 2, size=()))])
                     ).toDF(["feature", "label"])

        estimator = get_estimator(workers_per_node=2,
                                  model_fn=lambda config: IdentityNet())
        result = estimator.predict(df, batch_size=4,
                                   feature_cols=["feature"])
        expr = "sum(cast(feature <> to_array(prediction) as int)) as error"
        assert result.selectExpr(expr).first()["error"] == 0

    def test_xshards_predict(self):

        sc = init_nncontext()
        rdd = sc.range(0, 110).map(lambda x: np.array([x] * 50))
        shards = rdd.mapPartitions(lambda iter: chunks(iter, 5)).map(
            lambda x: {"x": np.stack([list(i) for i in x])})  # convert chain to list
        shards = SparkXShards(shards)

        estimator = get_estimator(workers_per_node=2,
                                  model_fn=lambda config: IdentityNet())
        result_shards = estimator.predict(shards, batch_size=4)
        result = np.concatenate([shard["prediction"] for shard in result_shards.collect()])
        expected_result = np.concatenate([shard["x"] for shard in result_shards.collect()])

        assert np.array_equal(result, expected_result)

    def test_pandas_dataframe(self):

        OrcaContext.pandas_read_backend = "pandas"
        file_path = os.path.join(resource_path, "orca/learn/ncf.csv")
        data_shard = read_csv(file_path, usecols=[0, 1, 2], dtype={0: np.float32, 1: np.float32,
                                                                   2: np.float32})

        estimator = get_estimator(model_fn=lambda config: SimpleModel())
        estimator.fit(data_shard, batch_size=2, epochs=2,
                      validation_data=data_shard,
                      feature_cols=["user", "item"],
                      label_cols=["label"])

        estimator.evaluate(data_shard, batch_size=2, feature_cols=["user", "item"],
                           label_cols=["label"])
        result = estimator.predict(data_shard, batch_size=2, feature_cols=["user", "item"])
        predictions = result.collect()[0]
        import pandas as pd
        assert isinstance(predictions, pd.DataFrame), "predict should return a pandas dataframe"
        assert isinstance(predictions["prediction"], pd.Series), \
               "predict dataframe should have a column named prediction"

    def test_multiple_inputs_model(self):

        sc = init_nncontext()
        rdd = sc.parallelize(range(100))

        spark = SparkSession.builder.getOrCreate()
        data = rdd.map(lambda x: ([float(x)] * 25, [float(x)] * 25,
                                  [float(np.random.randint(0, 2, size=()))])
                       )
        schema = StructType([
            StructField("f1", ArrayType(FloatType()), True),
            StructField("f2", ArrayType(FloatType()), True),
            StructField("label", ArrayType(FloatType()), True)
        ])

        df = spark.createDataFrame(data=data, schema=schema)

        estimator = get_estimator(workers_per_node=2,
                                  model_fn=lambda config: MultiInputNet())
        estimator.fit(df, batch_size=4, epochs=2,
                      validation_data=df,
                      feature_cols=["f1", "f2"],
                      label_cols=["label"])
        estimator.evaluate(df, batch_size=4,
                           feature_cols=["f1", "f2"],
                           label_cols=["label"])
        result = estimator.predict(df, batch_size=4,
                                   feature_cols=["f1", "f2"])
        result.collect()

    def test_uneven_data(self):
        sc = init_nncontext()
        spark = SparkSession.builder.getOrCreate()
        rdd = sc.range(0, 100).repartition(3)
        # the data and model are constructed that loss on worker 0 is always 0.0
        # and loss on worker 1 is always 1.0

        data = rdd.mapPartitionsWithIndex(lambda idx, iter: [([float(idx)], [0.0]) for _ in iter])
        schema = StructType([
            StructField("feature", ArrayType(FloatType()), True),
            StructField("label", ArrayType(FloatType()), True)
        ])

        df = spark.createDataFrame(data=data, schema=schema)

        estimator = get_estimator(workers_per_node=2,
                                  model_fn=lambda config: LinearModel(),
                                  loss=nn.MSELoss())
        stats = estimator.fit(df, batch_size=4, epochs=2,
                              validation_data=df,
                              feature_cols=["feature"],
                              label_cols=["label"])
        estimator.evaluate(df, batch_size=4,
                           feature_cols=["feature"],
                           label_cols=["label"])

    def test_sync_stats(self):
        sc = init_nncontext()
        spark = SparkSession.builder.getOrCreate()
        rdd = sc.range(0, 100).repartition(2)
        # the data and model are constructed that loss on worker 0 is always 0.0
        # and loss on worker 1 is always 1.0

        data = rdd.mapPartitionsWithIndex(lambda idx, iter: [([float(idx)], [0.0]) for _ in iter])
        schema = StructType([
            StructField("feature", ArrayType(FloatType()), True),
            StructField("label", ArrayType(FloatType()), True)
        ])

        df = spark.createDataFrame(data=data, schema=schema)

        estimator = get_estimator(workers_per_node=2,
                                  model_fn=lambda config: LinearModel(),
                                  loss=nn.MSELoss(),
                                  optimizer=get_zero_optimizer,
                                  sync_stats=True)
        stats = estimator.fit(df, batch_size=4, epochs=2,
                              feature_cols=["feature"],
                              label_cols=["label"],
                              reduce_results=False)
        worker_0_stat0, worker_1_stats = stats[0]

        for k in worker_0_stat0:
            if k in {"num_samples", "batch_count"}:
                continue
            v0 = worker_0_stat0[k]
            v1 = worker_1_stats[k]
            error_msg = f"stats from all workers should be the same, " \
                        f"but got worker_0_stat0: {worker_0_stat0}, " \
                        f"worker_1_stats: {worker_1_stats}"
            assert abs(v1 - v0) < 1e-6, error_msg

    def test_not_sync_stats(self):
        sc = init_nncontext()
        spark = SparkSession.builder.getOrCreate()
        rdd = sc.range(0, 100).repartition(2)

        # the data and model are constructed that loss on worker 0 is always 0.0
        # and loss on worker 1 is always 1.0

        data = rdd.mapPartitionsWithIndex(lambda idx, iter: [([float(idx)], [0.0]) for _ in iter])
        schema = StructType([
            StructField("feature", ArrayType(FloatType()), True),
            StructField("label", ArrayType(FloatType()), True)
        ])

        df = spark.createDataFrame(data=data, schema=schema)

        estimator = get_estimator(workers_per_node=2,
                                  model_fn=lambda config: LinearModel(),
                                  loss=nn.MSELoss(),
                                  optimizer=get_zero_optimizer,
                                  sync_stats=False)
        stats = estimator.fit(df, batch_size=4, epochs=2,
                              feature_cols=["feature"],
                              label_cols=["label"],
                              reduce_results=False)
        worker_0_stats, worker_1_stats = stats[0]
        train_loss_0 = worker_0_stats["train_loss"]
        train_loss_1 = worker_1_stats["train_loss"]
        error_msg = f"stats from all workers should not be the same, " \
                    f"but got worker_0_stats: {worker_0_stats}, worker_1_stats: {worker_1_stats}"
        assert abs(train_loss_0 - train_loss_1) > 0.9, error_msg

    def test_data_parallel_sgd_correctness(self):
        sc = init_nncontext()
        spark = SparkSession.builder.getOrCreate()
        rdd = sc.range(0, 100).repartition(2)

        # partition 0: [(0, 0), (0, 0)]
        # partition 1: [(1, 0), (1, 0)]
        # model: y = w * x
        # loss = (wx)^2
        # dloss/dw = 2x^2*w
        # end of first iteration:
        #    partition 0 loss: 0.0
        #    partition 1 loss: 1.0
        #    avg_grad = avg([0, 0, 2, 2]) = 1
        #    weight = 1.0 - 0.5 * avg_grad = 0.5
        # end of second iteration:
        #    partition 0 loss: 0.0
        #    partition 1 loss: 0.25
        #    avg_grad = avg([0, 0, 1, 1]) = 0.5
        #    weight = 0.5 - 0.5 * avg_grad = 0.25
        data = rdd.mapPartitionsWithIndex(lambda idx, iter: [([float(idx)], [0.0]) for _ in iter][:2])
        schema = StructType([
            StructField("feature", ArrayType(FloatType()), True),
            StructField("label", ArrayType(FloatType()), True)
        ])
        df = spark.createDataFrame(data=data, schema=schema)

        def get_optimizer(model, config):
            return torch.optim.SGD(model.parameters(), lr=0.5)

        estimator = Estimator.from_torch(model=lambda config: LinearModel(),
                                         optimizer=get_optimizer,
                                         loss=torch.nn.MSELoss(),
                                         metrics=Accuracy(),
                                         config={},
                                         workers_per_node=2,
                                         backend="ray",
                                         sync_stats=False)

        stats = estimator.fit(df, batch_size=4, epochs=2,
                              feature_cols=["feature"],
                              label_cols=["label"],
                              reduce_results=False)

        state = estimator.get_state_dict()
        assert state['models'][0]['fc1.weight'].item() == 0.25

    # not work right now
    # ray logs cannot be captured
    # not sure why
    # def test_logging_train_stats(self):

    #     sc = init_nncontext()
    #     rdd = sc.range(0, 100)
    #     df = rdd.map(lambda x: (np.random.randn(50).astype(np.float32).tolist(),
    #                             [int(np.random.randint(0, 2, size=()))])
    #                  ).toDF(["feature", "label"])

    #     estimator = get_estimator(workers_per_node=2, sync_stats=False, log_level=logging.DEBUG)
    #     captured_before = self._capsys.readouterr().out
    #     stats = estimator.fit(df, batch_size=4, epochs=2,
    #                             feature_cols=["feature"],
    #                             label_cols=["label"])

    #     captured_after = self._capsys.readouterr().out
    #     message = captured_after[len(captured_before):]
    #     assert "Finished training epoch 1, stats: {" in message
    #     assert "Finished training epoch 2, stats: {" in message

    # @pytest.fixture(autouse=True)
    # def inject_fixtures(self, capsys):
    #     self._capsys = capsys

    def test_checkpoint_callback(self):
        from bigdl.orca.learn.pytorch.callbacks.model_checkpoint import ModelCheckpoint
        sc = OrcaContext.get_spark_context()
        spark = SparkSession.builder.getOrCreate()
        rdd = sc.range(0, 100)
        epochs = 2
        data = rdd.map(lambda x: (np.random.randn(50).astype(np.float32).tolist(),
                                  [float(np.random.randint(0, 2, size=()))])
                       )
        schema = StructType([
            StructField("feature", ArrayType(FloatType()), True),
            StructField("label", ArrayType(FloatType()), True)
        ])
        df = spark.createDataFrame(data=data, schema=schema)
        df = df.cache()

        estimator = get_estimator(workers_per_node=2, log_level=logging.DEBUG)

        try:
            temp_dir = tempfile.mkdtemp()
            callbacks = [
                ModelCheckpoint(filepath=os.path.join(temp_dir, "test-{epoch}"),
                                save_weights_only=True)
            ]
            estimator.fit(df, batch_size=4, epochs=epochs,
                          callbacks=callbacks,
                          feature_cols=["feature"],
                          label_cols=["label"])
            eval_before = estimator.evaluate(df, batch_size=4,
                                             feature_cols=["feature"],
                                             label_cols=["label"])
            for i in range(epochs):
                assert os.path.isfile(os.path.join(temp_dir, f"test-epoch={i + 1}.ckpt"))

            latest_checkpoint_path = Estimator.latest_checkpoint(temp_dir)
            assert os.path.isfile(latest_checkpoint_path)
            estimator.shutdown()
            new_estimator = get_estimator(workers_per_node=2, log_level=logging.DEBUG)
            new_estimator.load_checkpoint(latest_checkpoint_path)
            eval_after = new_estimator.evaluate(df, batch_size=4,
                                                feature_cols=["feature"],
                                                label_cols=["label"])
            for name, value in eval_before.items():
                np.testing.assert_almost_equal(value, eval_after[name])
            res = new_estimator.predict(df, feature_cols=["feature"]).collect()
        finally:
            shutil.rmtree(temp_dir)

        with pytest.raises(RuntimeError):
            Estimator.latest_checkpoint(temp_dir)

    def test_checkpoint_callback_by_iter(self):
        from bigdl.orca.learn.pytorch.callbacks.model_checkpoint import ModelCheckpoint
        sc = OrcaContext.get_spark_context()
        spark = SparkSession.builder.getOrCreate()
        rdd = sc.range(0, 100)
        epochs = 1
        data = rdd.map(lambda x: (np.random.randn(50).astype(np.float32).tolist(),
                                  [float(np.random.randint(0, 2, size=()))])
                       )
        schema = StructType([
            StructField("feature", ArrayType(FloatType()), True),
            StructField("label", ArrayType(FloatType()), True)
        ])
        df = spark.createDataFrame(data=data, schema=schema)
        df = df.cache()

        size = df.count()
        batch_size = 4
        interval = 5

        estimator = get_estimator(workers_per_node=2, log_level=logging.DEBUG)

        try:
            temp_dir = tempfile.mkdtemp()
            callbacks = [
                ModelCheckpoint(filepath=os.path.join(temp_dir, "test-{iter}"),
                                save_weights_only=True,
                                by_epoch=False,
                                interval=interval)
            ]
            estimator.fit(df, batch_size=batch_size, epochs=epochs,
                          callbacks=callbacks,
                          feature_cols=["feature"],
                          label_cols=["label"])

            for i in range(interval, int(size / batch_size) + 1, interval):
                assert os.path.isfile(os.path.join(temp_dir, f"test-iter={i}.ckpt"))

            estimator.shutdown()
        finally:
            shutil.rmtree(temp_dir)

    def test_manual_ckpt(self):
        sc = OrcaContext.get_spark_context()
        spark = SparkSession.builder.getOrCreate()
        rdd = sc.range(0, 100)
        epochs = 2
        data = rdd.map(lambda x: (np.random.randn(50).astype(np.float32).tolist(),
                                  [float(np.random.randint(0, 2, size=()))])
                       )
        schema = StructType([
            StructField("feature", ArrayType(FloatType()), True),
            StructField("label", ArrayType(FloatType()), True)
        ])

        df = spark.createDataFrame(data=data, schema=schema)
        df = df.cache()

        estimator = get_estimator(workers_per_node=2)
        estimator.fit(df, batch_size=4, epochs=epochs,
                      feature_cols=["feature"],
                      label_cols=["label"])
        eval_before = estimator.evaluate(df, batch_size=4,
                                         feature_cols=["feature"],
                                         label_cols=["label"])

        try:
            temp_dir = tempfile.mkdtemp()
            ckpt_file = os.path.join(temp_dir, "manual.ckpt")
            estimator.save_checkpoint(ckpt_file)
            estimator.shutdown()
            new_estimator = get_estimator(workers_per_node=2)
            new_estimator.load_checkpoint(ckpt_file)
            eval_after = new_estimator.evaluate(df, batch_size=4,
                                                feature_cols=["feature"],
                                                label_cols=["label"])
            for name, value in eval_before.items():
                np.testing.assert_almost_equal(value, eval_after[name])
        finally:
            shutil.rmtree(temp_dir)

    def test_custom_callback(self):
        estimator = get_estimator(workers_per_node=2)
        callbacks = [CustomCallback()]
        estimator.fit(train_data_loader, epochs=4, batch_size=128,
                      validation_data=val_data_loader, callbacks=callbacks)

    def test_customized_metric(self):
        estimator = get_estimator(metrics=customized_metric, workers_per_node=2)
        estimator.fit(train_data_loader, epochs=4, batch_size=128,
                      validation_data=val_data_loader)

    def test_tensorboard_callback(self):
        from bigdl.orca.learn.pytorch.callbacks.tensorboard import TensorBoardCallback
        sc = OrcaContext.get_spark_context()
        spark = SparkSession.builder.getOrCreate()
        rdd = sc.range(0, 100)
        epochs = 2
        data = rdd.map(lambda x: (np.random.randn(50).astype(np.float32).tolist(),
                                  [float(np.random.randint(0, 2, size=()))])
                       )
        schema = StructType([
            StructField("feature", ArrayType(FloatType()), True),
            StructField("label", ArrayType(FloatType()), True)
        ])
        df = spark.createDataFrame(data=data, schema=schema)
        df = df.cache()

        estimator = get_estimator(workers_per_node=2, log_level=logging.DEBUG)

        try:
            temp_dir = tempfile.mkdtemp()
            log_dir = os.path.join(temp_dir, "runs_epoch")

            callbacks = [
                TensorBoardCallback(log_dir=log_dir, freq="epoch")
            ]
            estimator.fit(df, batch_size=4, epochs=epochs,
                          callbacks=callbacks,
                          feature_cols=["feature"],
                          label_cols=["label"])

            assert len(os.listdir(log_dir)) > 0

            log_dir = os.path.join(temp_dir, "runs_batch")

            callbacks = [
                TensorBoardCallback(log_dir=log_dir, freq="batch")
            ]
            estimator.fit(df, batch_size=4, epochs=epochs,
                          callbacks=callbacks,
                          feature_cols=["feature"],
                          label_cols=["label"])

            assert len(os.listdir(log_dir)) > 0
        finally:
            shutil.rmtree(temp_dir)

        estimator.shutdown()

    def test_optional_optimizer(self):
        sc = OrcaContext.get_spark_context()
        rdd = sc.range(0, 110).map(lambda x: np.array([x] * 50))
        shards = rdd.mapPartitions(lambda iter: chunks(iter, 5)).map(
            lambda x: {"x": np.stack([list(i) for i in x])})  # convert chain to list
        shards = SparkXShards(shards)

        try:
            estimator = get_estimator(model_fn=lambda config: IdentityNet())
            path = "/tmp/optimizer_model"
            estimator.save(path)
            estimator.shutdown()

            estimator = Estimator.from_torch(model=lambda config: IdentityNet(),
                                             workers_per_node=2,
                                             backend="ray")
            estimator.load(path)

            result = estimator.predict(shards, batch_size=4)
            predicted_result = np.concatenate([shard["prediction"] for shard in result.collect()])
            expected_result = np.concatenate([shard["x"] for shard in result.collect()])
        finally:
            os.remove(path)

        assert np.array_equal(predicted_result, expected_result)

if __name__ == "__main__":
    pytest.main([__file__])
