from unittest import TestCase
import os.path
import pytest
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from bigdl.orca import OrcaContext
import bigdl.orca.data.pandas
from bigdl.orca.learn.mxnet import Estimator, create_config

def prepare_data_symbol(df):
    if False:
        i = 10
        return i + 15
    data = {'input': np.array(df['data'].values.tolist())}
    label = {'label': df['label'].values}
    return {'x': data, 'y': label}

def prepare_data_gluon(df):
    if False:
        i = 10
        return i + 15
    data = np.array(df['data'].values.tolist())
    label = df['label'].values
    return {'x': data, 'y': label}

def get_loss(config):
    if False:
        return 10
    return gluon.loss.SoftmaxCrossEntropyLoss()

def get_gluon_metrics(config):
    if False:
        while True:
            i = 10
    return mx.metric.Accuracy()

def get_metrics(config):
    if False:
        while True:
            i = 10
    return 'accuracy'

def get_symbol_model(config):
    if False:
        i = 10
        return i + 15
    input_data = mx.symbol.Variable('input')
    y_true = mx.symbol.Variable('label')
    fc1 = mx.symbol.FullyConnected(data=input_data, num_hidden=20, name='fc1')
    fc2 = mx.symbol.FullyConnected(data=fc1, num_hidden=10, name='fc2')
    output = mx.symbol.SoftmaxOutput(data=fc2, label=y_true, name='output')
    mod = mx.mod.Module(symbol=output, data_names=['input'], label_names=['label'], context=mx.cpu())
    return mod

def get_gluon_model(config):
    if False:
        while True:
            i = 10

    class SimpleModel(gluon.Block):

        def __init__(self, **kwargs):
            if False:
                return 10
            super(SimpleModel, self).__init__(**kwargs)
            self.fc1 = nn.Dense(20)
            self.fc2 = nn.Dense(10)

        def forward(self, x):
            if False:
                i = 10
                return i + 15
            x = self.fc1(x)
            x = self.fc2(x)
            return x
    net = SimpleModel()
    net.initialize(mx.init.Xavier(magnitude=2.24), ctx=[mx.cpu()])
    return net

class TestMXNetSparkXShards(TestCase):

    def setup_method(self, method):
        if False:
            i = 10
            return i + 15
        self.resource_path = os.path.join(os.path.split(__file__)[0], '../resources')
        OrcaContext.pandas_read_backend = 'pandas'

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        OrcaContext.pandas_read_backend = 'spark'

    def test_xshards_symbol_with_val(self):
        if False:
            for i in range(10):
                print('nop')
        resource_path = os.path.join(os.path.split(__file__)[0], '../../../resources')
        train_file_path = os.path.join(resource_path, 'orca/learn/single_input_json/train')
        train_data_shard = bigdl.orca.data.pandas.read_json(train_file_path, orient='records', lines=False).transform_shard(prepare_data_symbol)
        test_file_path = os.path.join(resource_path, 'orca/learn/single_input_json/test')
        test_data_shard = bigdl.orca.data.pandas.read_json(test_file_path, orient='records', lines=False).transform_shard(prepare_data_symbol)
        config = create_config(log_interval=1, seed=42)
        estimator = Estimator.from_mxnet(config=config, model_creator=get_symbol_model, validation_metrics_creator=get_metrics, eval_metrics_creator=get_metrics, num_workers=2)
        estimator.fit(train_data_shard, epochs=2)
        train_data_shard2 = bigdl.orca.data.pandas.read_json(train_file_path, orient='records', lines=False).transform_shard(prepare_data_symbol)
        estimator.fit(train_data_shard2, validation_data=test_data_shard, epochs=1, batch_size=32)
        estimator.shutdown()

    def test_xshards_symbol_without_val(self):
        if False:
            i = 10
            return i + 15
        resource_path = os.path.join(os.path.split(__file__)[0], '../../../resources')
        train_file_path = os.path.join(resource_path, 'orca/learn/single_input_json/train')
        train_data_shard = bigdl.orca.data.pandas.read_json(train_file_path, orient='records', lines=False).transform_shard(prepare_data_symbol)
        config = create_config(log_interval=1, seed=42)
        estimator = Estimator.from_mxnet(config=config, model_creator=get_symbol_model, eval_metrics_creator=get_metrics, num_workers=2)
        estimator.fit(train_data_shard, epochs=2, batch_size=16)
        estimator.shutdown()

    def test_xshards_gluon(self):
        if False:
            for i in range(10):
                print('nop')
        resource_path = os.path.join(os.path.split(__file__)[0], '../../../resources')
        train_file_path = os.path.join(resource_path, 'orca/learn/single_input_json/train')
        train_data_shard = bigdl.orca.data.pandas.read_json(train_file_path, orient='records', lines=False).transform_shard(prepare_data_gluon)
        test_file_path = os.path.join(resource_path, 'orca/learn/single_input_json/train')
        test_data_shard = bigdl.orca.data.pandas.read_json(test_file_path, orient='records', lines=False).transform_shard(prepare_data_gluon)
        config = create_config(log_interval=1, seed=42)
        estimator = Estimator.from_mxnet(config=config, model_creator=get_gluon_model, loss_creator=get_loss, validation_metrics_creator=get_gluon_metrics, eval_metrics_creator=get_gluon_metrics, num_workers=2)
        estimator.fit(train_data_shard, validation_data=test_data_shard, epochs=2, batch_size=8)
        estimator.shutdown()
if __name__ == '__main__':
    pytest.main([__file__])