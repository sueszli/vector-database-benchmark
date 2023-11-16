import os
import math
from unittest import TestCase
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.nn import functional as F
from pyspark.sql import SparkSession
from pyspark.sql.types import FloatType, ArrayType, DoubleType, StructType, StructField, MapType, StringType
from bigdl.orca import OrcaContext
from bigdl.orca.learn.metrics import Accuracy
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.dllib.nncontext import init_nncontext
from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca.data import SparkXShards
from bigdl.orca.data.image.utils import chunks
from bigdl.orca.learn.pytorch.callbacks import Callback, MainCallback
import tempfile
import shutil
import logging
np.random.seed(1337)
resource_path = os.path.join(os.path.realpath(os.path.dirname(__file__)), '../../../resources')

class LinearDataset(torch.utils.data.Dataset):
    """y = a * x + b"""

    def __init__(self, size=1000):
        if False:
            for i in range(10):
                print('nop')
        X1 = torch.randn(size // 2, 50)
        X2 = torch.randn(size // 2, 50) + 1.5
        self.x = torch.cat([X1, X2], dim=0)
        Y1 = torch.zeros(size // 2, 1)
        Y2 = torch.ones(size // 2, 1)
        self.y = torch.cat([Y1, Y2], dim=0)

    def __getitem__(self, index):
        if False:
            i = 10
            return i + 15
        return (self.x[index, None], self.y[index, None])

    def __len__(self):
        if False:
            while True:
                i = 10
        return len(self.x)

class Net(nn.Module):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.fc1 = nn.Linear(50, 50)
        self.relu1 = nn.ReLU()
        self.dout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, 100)
        self.prelu = nn.PReLU(1)
        self.out = nn.Linear(100, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, input_):
        if False:
            i = 10
            return i + 15
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        dout = self.dout(h1)
        a2 = self.fc2(dout)
        h2 = self.prelu(a2)
        a3 = self.out(h2)
        y = self.out_act(a3)
        return y

class ComplicatedOutputNet(nn.Module):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.fc1 = nn.Linear(50, 50)
        self.relu1 = nn.ReLU()
        self.dout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, 100)
        self.prelu = nn.PReLU(1)
        self.out = nn.Linear(100, 3)
        self.out_act = nn.Sigmoid()

    def forward(self, input_):
        if False:
            for i in range(10):
                print('nop')
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        dout = self.dout(h1)
        a2 = self.fc2(dout)
        h2 = self.prelu(a2)
        a3 = self.out(h2)
        y = self.out_act(a3)
        return (y[:, 0], {'y1': y[:, 1], 'y2': y[:, 2]})

class IdentityNet(nn.Module):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.fc1 = nn.Linear(50, 50)

    def forward(self, input_):
        if False:
            return 10
        return input_

class LinearModel(nn.Module):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.fc1 = nn.Linear(1, 1, bias=False)
        self.fc1.weight.data.fill_(1.0)

    def forward(self, input_):
        if False:
            for i in range(10):
                print('nop')
        return self.fc1(input_)

class MultiInputNet(nn.Module):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.fc1 = nn.Linear(50, 50)
        self.out = nn.Linear(50, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, input1, input2):
        if False:
            return 10
        x = torch.cat((input1, input2), 1)
        x = self.fc1(x)
        x = self.out(x)
        x = self.out_act(x)
        return x

class DictNet(nn.Module):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.fc1 = nn.Linear(50, 50)
        self.out = nn.Linear(50, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        if False:
            while True:
                i = 10
        x = self.fc1(x)
        x = self.out(x)
        x = self.out_act(x)
        return {'y': x}

class MultiDictNet(nn.Module):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.fc1 = nn.Linear(50, 50)
        self.out = nn.Linear(50, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        if False:
            while True:
                i = 10
        x = self.fc1(x)
        x = self.out(x)
        x = self.out_act(x)
        return {'y': x, 'PlaceHolder': torch.ones_like(x)}

def multi_dict_loss_fn(config):
    if False:
        for i in range(10):
            print('nop')

    def mock_BCELoss(x, y):
        if False:
            return 10
        assert x['PlaceHolder'].size() == y['y'].size()
        assert x['PlaceHolder'][0][0].item() == 1.0
        return F.binary_cross_entropy(x['y'], y['y'])
    return mock_BCELoss

class SimpleModel(nn.Module):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.fc = nn.Linear(2, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, input1, input2):
        if False:
            while True:
                i = 10
        x = torch.stack((input1, input2), dim=1)
        x = self.fc(x)
        x = self.out_act(x).flatten()
        return x

class CustomCallback(Callback):

    def on_train_end(self, logs=None):
        if False:
            for i in range(10):
                print('nop')
        assert 'train_loss' in logs
        assert 'val_loss' in logs
        assert self.model

    def on_epoch_end(self, epoch, logs=None):
        if False:
            for i in range(10):
                print('nop')
        assert 'train_loss' in logs
        assert 'val_loss' in logs
        assert self.model

class DictMCB(MainCallback):

    def on_pred_forward(self, runner):
        if False:
            for i in range(10):
                print('nop')
        output = runner.model(*runner.batch)
        runner.output = {k: v.detach().numpy() for (k, v) in output.items()}

class ComplicatedMCB(MainCallback):

    def on_pred_forward(self, runner):
        if False:
            for i in range(10):
                print('nop')
        output = runner.model(*runner.batch)
        runner.output = (output[0].detach().numpy(), {k: v.detach().numpy() for (k, v) in output[1].items()})

def train_data_loader(config, batch_size):
    if False:
        for i in range(10):
            print('nop')
    train_dataset = LinearDataset(size=config.get('data_size', 1000))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    return train_loader

def val_data_loader(config, batch_size):
    if False:
        for i in range(10):
            print('nop')
    val_dataset = LinearDataset(size=config.get('val_size', 400))
    validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    return validation_loader

def get_model(config):
    if False:
        while True:
            i = 10
    torch.manual_seed(0)
    return Net()

def get_optimizer(model, config):
    if False:
        for i in range(10):
            print('nop')
    return torch.optim.SGD(model.parameters(), lr=config.get('lr', 0.01))

def get_estimator(workers_per_node=1, model_fn=get_model, loss_fn=nn.BCELoss(), metrics=Accuracy(), sync_stats=False, log_level=logging.INFO, model_dir=None):
    if False:
        i = 10
        return i + 15
    estimator = Estimator.from_torch(model=model_fn, optimizer=get_optimizer, loss=loss_fn, metrics=metrics, config={'lr': 0.01}, workers_per_node=workers_per_node, backend='spark', sync_stats=sync_stats, model_dir=model_dir, log_level=log_level)
    return estimator

class TestPyTorchEstimatorBasic(TestCase):

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        self.sc = init_orca_context(cores=4)
        self.model_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        if False:
            i = 10
            return i + 15
        stop_orca_context()
        shutil.rmtree(self.model_dir)

    def test_data_creator_convergence(self):
        if False:
            for i in range(10):
                print('nop')
        estimator = get_estimator(workers_per_node=2)
        start_val_stats = estimator.evaluate(val_data_loader, batch_size=64)
        print(start_val_stats)
        train_stats = estimator.fit(train_data_loader, epochs=4, batch_size=128, validation_data=val_data_loader)
        print(train_stats)
        assert train_stats[0]['batch_count'] == math.ceil(1000 // 2 / (128 // 2))
        end_val_stats = estimator.evaluate(val_data_loader, batch_size=64)
        print(end_val_stats)
        assert 0 < end_val_stats['Accuracy'] < 1
        assert estimator.get_model()
        dloss = end_val_stats['val_loss'] - start_val_stats['val_loss']
        dacc = end_val_stats['Accuracy'] - start_val_stats['Accuracy']
        print(f'dLoss: {dloss}, dAcc: {dacc}')
        assert dloss < 0 < dacc, 'training sanity check failed. loss increased!'

    def test_spark_xshards(self):
        if False:
            return 10
        from bigdl.dllib.nncontext import init_nncontext
        from bigdl.orca.data import SparkXShards
        estimator = get_estimator(workers_per_node=1)
        sc = init_nncontext()
        x_rdd = sc.parallelize(np.random.rand(4000, 1, 50).astype(np.float32))
        y_rdd = sc.parallelize(np.random.randint(0, 2, size=(4000, 1, 1)).astype(np.float32))
        rdd = x_rdd.zip(y_rdd).map(lambda x_y: {'x': x_y[0], 'y': x_y[1]})
        (train_rdd, val_rdd) = rdd.randomSplit([0.9, 0.1])
        train_xshards = SparkXShards(train_rdd)
        val_xshards = SparkXShards(val_rdd)
        train_stats = estimator.fit(train_xshards, validation_data=val_xshards, batch_size=256, epochs=2)
        print(train_stats)
        val_stats = estimator.evaluate(val_xshards, batch_size=128)
        print(val_stats)

    def test_dataframe_train_eval(self):
        if False:
            print('Hello World!')
        sc = init_nncontext()
        spark = SparkSession.builder.getOrCreate()
        rdd = sc.range(0, 100)
        data = rdd.map(lambda x: (np.random.randn(50).astype(np.float32).tolist(), [float(np.random.randint(0, 2, size=()))]))
        schema = StructType([StructField('feature', ArrayType(FloatType()), True), StructField('label', ArrayType(FloatType()), True)])
        df = spark.createDataFrame(data=data, schema=schema)
        estimator = get_estimator(workers_per_node=2)
        estimator.fit(df, batch_size=4, epochs=2, validation_data=df, feature_cols=['feature'], label_cols=['label'])
        estimator.evaluate(df, batch_size=4, feature_cols=['feature'], label_cols=['label'])
        estimator.shutdown()

    def test_partition_num_less_than_workers(self):
        if False:
            i = 10
            return i + 15
        sc = init_nncontext()
        spark = SparkSession.builder.getOrCreate()
        rdd = sc.range(200, numSlices=1)
        data = rdd.map(lambda x: (np.random.randn(50).astype(np.float32).tolist(), [float(np.random.randint(0, 2, size=()))]))
        schema = StructType([StructField('feature', ArrayType(FloatType()), True), StructField('label', ArrayType(FloatType()), True)])
        df = spark.createDataFrame(data=data, schema=schema)
        estimator = get_estimator(workers_per_node=2)
        assert df.rdd.getNumPartitions() < estimator.num_workers
        estimator.fit(df, batch_size=4, epochs=2, validation_data=df, feature_cols=['feature'], label_cols=['label'])
        estimator.evaluate(df, batch_size=4, feature_cols=['feature'], label_cols=['label'])
        estimator.predict(df, feature_cols=['feature']).collect()

    def test_dataframe_predict(self):
        if False:
            i = 10
            return i + 15

        def to_array_(v):
            if False:
                for i in range(10):
                    print('nop')
            return v.toArray().tolist()
        sc = init_nncontext()
        spark = SparkSession(sc)
        spark.udf.register('to_array', to_array_, ArrayType(DoubleType()))
        rdd = sc.parallelize(range(20))
        df = rdd.map(lambda x: ([float(x)] * 5, [int(np.random.randint(0, 2, size=()))])).toDF(['feature', 'label'])
        estimator = get_estimator(workers_per_node=2, model_fn=lambda config: IdentityNet())
        result = estimator.predict(df, batch_size=4, feature_cols=['feature'])
        assert 'prediction' in result.columns
        expr = 'sum(cast(feature <> to_array(prediction) as int)) as error'
        assert result.selectExpr(expr).first()['error'] == 0
        predictions = result.collect()
        assert len(predictions) == 20

    def test_xshards_predict_save_load(self):
        if False:
            for i in range(10):
                print('nop')
        sc = init_nncontext()
        rdd = sc.range(0, 110).map(lambda x: np.array([x] * 50))
        shards = rdd.mapPartitions(lambda iter: chunks(iter, 5)).map(lambda x: {'x': np.stack([list(i) for i in x])})
        shards = SparkXShards(shards)
        estimator = get_estimator(workers_per_node=2, model_fn=lambda config: IdentityNet())
        result_shards = estimator.predict(shards, batch_size=4)
        result_before = np.concatenate([shard['prediction'] for shard in result_shards.collect()])
        expected_result = np.concatenate([shard['x'] for shard in result_shards.collect()])
        assert np.array_equal(result_before, expected_result)
        path = '/tmp/model.pth'
        try:
            estimator.save(path)
            estimator.load(path)
            result_shards = estimator.predict(shards, batch_size=4)
            result_after = np.concatenate([shard['prediction'] for shard in result_shards.collect()])
        finally:
            os.remove(path)
        assert np.array_equal(result_before, result_after)

    def test_multiple_inputs_model(self):
        if False:
            while True:
                i = 10
        sc = init_nncontext()
        rdd = sc.parallelize(range(100))
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        data = rdd.map(lambda x: ([float(x)] * 25, [float(x)] * 25, [float(np.random.randint(0, 2, size=()))]))
        schema = StructType([StructField('f1', ArrayType(FloatType()), True), StructField('f2', ArrayType(FloatType()), True), StructField('label', ArrayType(FloatType()), True)])
        df = spark.createDataFrame(data=data, schema=schema)
        estimator = get_estimator(workers_per_node=2, model_fn=lambda config: MultiInputNet())
        estimator.fit(df, batch_size=4, epochs=2, validation_data=df, feature_cols=['f1', 'f2'], label_cols=['label'])
        estimator.evaluate(df, batch_size=4, feature_cols=['f1', 'f2'], label_cols=['label'])
        result = estimator.predict(df, batch_size=4, feature_cols=['f1', 'f2'])
        result.collect()

    def test_dict_outputs_model(self):
        if False:
            while True:
                i = 10
        sc = init_nncontext()
        rdd = sc.parallelize(range(100))
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        data = rdd.map(lambda x: ([float(x)] * 50, {'y': [float(np.random.randint(0, 2, size=()))]}))
        schema = StructType([StructField('f', ArrayType(FloatType()), True), StructField('label', MapType(StringType(), ArrayType(FloatType())), True)])
        df = spark.createDataFrame(data=data, schema=schema)
        estimator = get_estimator(workers_per_node=2, model_fn=lambda config: DictNet(), loss_fn=lambda config: lambda x, y: F.binary_cross_entropy(x['y'], y['y']), metrics=None)
        estimator.fit(df, batch_size=4, epochs=2, validation_data=df, feature_cols=['f'], label_cols=['label'])
        estimator.evaluate(df, batch_size=4, feature_cols=['f'], label_cols=['label'])
        result = estimator.predict(df, batch_size=4, callbacks=[DictMCB()], feature_cols=['f'])
        result.collect()

    def test_dict_multi_outputs_model(self):
        if False:
            print('Hello World!')
        sc = init_nncontext()
        rdd = sc.parallelize(range(100))
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        data = rdd.map(lambda x: ([float(x)] * 50, {'y': [float(np.random.randint(0, 2, size=()))]}))
        schema = StructType([StructField('f', ArrayType(FloatType()), True), StructField('label', MapType(StringType(), ArrayType(FloatType())), True)])
        df = spark.createDataFrame(data=data, schema=schema)
        estimator = get_estimator(workers_per_node=2, model_fn=lambda config: MultiDictNet(), loss_fn=multi_dict_loss_fn, metrics=None)
        estimator.fit(df, batch_size=4, epochs=2, validation_data=df, feature_cols=['f'], label_cols=['label'])
        estimator.evaluate(df, batch_size=4, feature_cols=['f'], label_cols=['label'])
        result = estimator.predict(df, batch_size=4, callbacks=[DictMCB()], feature_cols=['f'])
        result.collect()

    def test_complicated_outputs_model_predict(self):
        if False:
            i = 10
            return i + 15
        sc = init_nncontext()
        rdd = sc.parallelize(range(100))
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        data = rdd.map(lambda x: ([float(x)] * 50, {'y': [float(np.random.randint(0, 2, size=()))]}))
        schema = StructType([StructField('f', ArrayType(FloatType()), True), StructField('label', MapType(StringType(), ArrayType(FloatType())), True)])
        df = spark.createDataFrame(data=data, schema=schema)
        estimator = get_estimator(workers_per_node=2, model_fn=lambda config: ComplicatedOutputNet())
        result = estimator.predict(df, batch_size=4, callbacks=[ComplicatedMCB()], feature_cols=['f'], output_cols=['scalar', 'dict'])
        result.collect()
        assert 'scalar' and 'dict' in result.columns

    def test_data_parallel_sgd_correctness(self):
        if False:
            for i in range(10):
                print('nop')
        sc = init_nncontext()
        spark = SparkSession.builder.getOrCreate()
        rdd = sc.range(0, 100).repartition(2)
        data = rdd.mapPartitionsWithIndex(lambda idx, iter: [([float(idx)], [0.0]) for _ in iter][:2])
        schema = StructType([StructField('feature', ArrayType(FloatType()), True), StructField('label', ArrayType(FloatType()), True)])
        df = spark.createDataFrame(data=data, schema=schema)

        def get_optimizer(model, config):
            if False:
                for i in range(10):
                    print('nop')
            return torch.optim.SGD(model.parameters(), lr=0.5)
        estimator = Estimator.from_torch(model=lambda config: LinearModel(), optimizer=get_optimizer, loss=torch.nn.MSELoss(), metrics=Accuracy(), config={}, workers_per_node=2, backend='spark', sync_stats=False)
        stats = estimator.fit(df, batch_size=4, epochs=2, validation_data=df, feature_cols=['feature'], label_cols=['label'], reduce_results=False)
        state = estimator.get_state_dict()
        assert state['models'][0]['fc1.weight'].item() == 0.25

    def test_checkpoint_callback(self):
        if False:
            print('Hello World!')
        from bigdl.orca.learn.pytorch.callbacks.model_checkpoint import ModelCheckpoint
        sc = OrcaContext.get_spark_context()
        spark = SparkSession.builder.getOrCreate()
        rdd = sc.range(0, 100)
        epochs = 2
        data = rdd.map(lambda x: (np.random.randn(50).astype(np.float32).tolist(), [float(np.random.randint(0, 2, size=()))]))
        schema = StructType([StructField('feature', ArrayType(FloatType()), True), StructField('label', ArrayType(FloatType()), True)])
        df = spark.createDataFrame(data=data, schema=schema)
        df = df.cache()
        estimator = get_estimator(workers_per_node=2, log_level=logging.DEBUG)
        callbacks = [ModelCheckpoint(filepath=os.path.join(self.model_dir, 'test-{epoch}'), save_weights_only=True)]
        estimator.fit(df, batch_size=4, epochs=epochs, callbacks=callbacks, feature_cols=['feature'], label_cols=['label'])
        eval_before = estimator.evaluate(df, batch_size=4, feature_cols=['feature'], label_cols=['label'])
        for i in range(epochs):
            assert os.path.isfile(os.path.join(self.model_dir, f'test-epoch={i + 1}.ckpt'))
        latest_checkpoint_path = Estimator.latest_checkpoint(self.model_dir)
        assert os.path.isfile(latest_checkpoint_path)
        estimator.shutdown()
        new_estimator = get_estimator(workers_per_node=2, log_level=logging.DEBUG)
        new_estimator.load_checkpoint(latest_checkpoint_path)
        eval_after = new_estimator.evaluate(df, batch_size=4, feature_cols=['feature'], label_cols=['label'])
        for (name, value) in eval_before.items():
            print(f'Comparing evaluate result of {name}')
            np.testing.assert_almost_equal(value, eval_after[name])
        res = new_estimator.predict(df, feature_cols=['feature']).collect()

    def test_manual_ckpt(self):
        if False:
            for i in range(10):
                print('nop')
        sc = OrcaContext.get_spark_context()
        spark = SparkSession.builder.getOrCreate()
        rdd = sc.range(0, 100)
        epochs = 2
        data = rdd.map(lambda x: (np.random.randn(50).astype(np.float32).tolist(), [float(np.random.randint(0, 2, size=()))]))
        schema = StructType([StructField('feature', ArrayType(FloatType()), True), StructField('label', ArrayType(FloatType()), True)])
        df = spark.createDataFrame(data=data, schema=schema)
        df = df.cache()
        estimator = get_estimator(workers_per_node=2, log_level=logging.DEBUG)
        estimator.fit(df, batch_size=4, epochs=epochs, feature_cols=['feature'], label_cols=['label'])
        eval_before = estimator.evaluate(df, batch_size=4, feature_cols=['feature'], label_cols=['label'])
        try:
            temp_dir = tempfile.mkdtemp()
            ckpt_file = os.path.join(temp_dir, 'manual.ckpt')
            estimator.save_checkpoint(ckpt_file)
            estimator.shutdown()
            new_estimator = get_estimator(workers_per_node=2, log_level=logging.DEBUG)
            new_estimator.load_checkpoint(ckpt_file)
            eval_after = new_estimator.evaluate(df, batch_size=4, feature_cols=['feature'], label_cols=['label'])
            for (name, value) in eval_before.items():
                np.testing.assert_almost_equal(value, eval_after[name])
        finally:
            shutil.rmtree(temp_dir)

    def test_custom_callback(self):
        if False:
            while True:
                i = 10
        estimator = get_estimator(workers_per_node=2)
        callbacks = [CustomCallback()]
        estimator.fit(train_data_loader, epochs=4, batch_size=128, validation_data=val_data_loader, callbacks=callbacks)

    def test_optional_optimizer(self):
        if False:
            for i in range(10):
                print('nop')
        sc = init_nncontext()
        rdd = sc.range(0, 110).map(lambda x: np.array([x] * 50))
        shards = rdd.mapPartitions(lambda iter: chunks(iter, 5)).map(lambda x: {'x': np.stack([list(i) for i in x])})
        shards = SparkXShards(shards)
        try:
            trainer = get_estimator(model_fn=lambda config: IdentityNet())
            path = '/tmp/optimizer_model.pth'
            trainer.save(path)
            trainer.shutdown()
            estimator = Estimator.from_torch(model=lambda config: IdentityNet(), workers_per_node=2, backend='spark')
            estimator.load(path)
            result_shards = estimator.predict(shards, batch_size=4)
            result_before = np.concatenate([shard['prediction'] for shard in result_shards.collect()])
            expected_result = np.concatenate([shard['x'] for shard in result_shards.collect()])
            assert np.array_equal(result_before, expected_result)
        finally:
            os.remove(path)

    def test_optional_model_creator(self):
        if False:
            print('Hello World!')
        sc = OrcaContext.get_spark_context()
        rdd = sc.range(0, 110).map(lambda x: np.array([x] * 50))
        shards = rdd.mapPartitions(lambda iter: chunks(iter, 5)).map(lambda x: {'x': np.stack([list(i) for i in x])})
        shards = SparkXShards(shards)
        try:
            estimator = get_estimator(model_fn=lambda config: IdentityNet())
            result_shards = estimator.predict(shards, batch_size=4)
            result_before = np.concatenate([shard['prediction'] for shard in result_shards.collect()])
            expected_result = np.concatenate([shard['x'] for shard in result_shards.collect()])
            assert np.array_equal(result_before, expected_result)
            path = '/tmp/entire_model.pth'
            estimator.save(path, entire=True)
            estimator.shutdown()
            trainer = Estimator.from_torch(optimizer=get_optimizer, loss=nn.BCELoss(), metrics=Accuracy(), config={'lr': 0.01}, workers_per_node=2, backend='spark')
            trainer.load(path)
            result_shards = trainer.predict(shards, batch_size=4)
            result_after = np.concatenate([shard['prediction'] for shard in result_shards.collect()])
        finally:
            os.remove(path)
        assert np.array_equal(expected_result, result_after)
if __name__ == '__main__':
    pytest.main([__file__])