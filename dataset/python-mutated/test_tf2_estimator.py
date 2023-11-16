import os
import pytest
import shutil
import tempfile
from unittest import TestCase
import numpy as np
import tensorflow as tf
from bigdl.orca.learn.tf2 import Estimator
from bigdl.orca import OrcaContext

def simple_model(config):
    if False:
        while True:
            i = 10
    model = tf.keras.models.Sequential([tf.keras.layers.Dense(10, input_shape=(1,)), tf.keras.layers.Dense(1)])
    return model

def compile_args(config):
    if False:
        for i in range(10):
            print('nop')
    import tensorflow as tf
    if 'lr' in config:
        lr = config['lr']
    else:
        lr = 0.001
    args = {'optimizer': tf.keras.optimizers.Adam(lr), 'loss': 'mean_squared_error', 'metrics': ['mean_squared_error']}
    return args

def model_creator(config):
    if False:
        return 10
    model = simple_model(config)
    model.compile(**compile_args(config))
    return model

class TestTF2Estimator(TestCase):

    def test_dataframe_with_empty_partition(self):
        if False:
            for i in range(10):
                print('nop')
        sc = OrcaContext.get_spark_context()
        rdd = sc.range(0, 100)
        rdd_with_empty = rdd.repartition(4).mapPartitionsWithIndex(lambda idx, part: [] if idx == 0 else part)
        spark = OrcaContext.get_spark_session()
        from pyspark.ml.linalg import DenseVector
        df = rdd_with_empty.map(lambda x: (DenseVector(np.random.randn(1).astype(np.float32)), int(np.random.randint(0, 2, size=())))).toDF(['feature', 'label'])
        config = {'lr': 0.2}
        try:
            temp_dir = tempfile.mkdtemp()
            trainer = Estimator.from_keras(model_creator=model_creator, verbose=True, config=config, workers_per_node=3, backend='spark')
            res = trainer.fit(df, epochs=5, batch_size=4, steps_per_epoch=25, feature_cols=['feature'], label_cols=['label'])
            print('start saving')
            trainer.save_weights(os.path.join(temp_dir, 'cifar10_keras.h5'))
            trainer.load_weights(os.path.join(temp_dir, 'cifar10_keras.h5'))
            res = trainer.evaluate(df, batch_size=4, num_steps=25, feature_cols=['feature'], label_cols=['label'])
            print('validation result: ', res)
            res = trainer.predict(df, feature_cols=['feature']).collect()
            print('predict result: ', res)
        finally:
            shutil.rmtree(temp_dir)

    def test_xshards_pandas_dataframe(self):
        if False:
            while True:
                i = 10
        from bigdl.orca.data.pandas import read_csv
        sc = OrcaContext.get_spark_context()

        def model_creator(config):
            if False:
                while True:
                    i = 10
            import tensorflow as tf
            input1 = tf.keras.layers.Input(shape=(1,))
            input2 = tf.keras.layers.Input(shape=(1,))
            concatenation = tf.concat([input1, input2], axis=-1)
            outputs = tf.keras.layers.Dense(units=1, activation='softmax')(concatenation)
            model = tf.keras.Model(inputs=[input1, input2], outputs=outputs)
            model.compile(**compile_args(config))
            return model
        resource_path = os.path.join(os.path.split(__file__)[0], '../../resources')
        file_path = os.path.join(resource_path, 'orca/learn/ncf.csv')
        xshards = read_csv(file_path, usecols=[0, 1, 2], dtype={0: np.float32, 1: np.float32, 2: np.float32})
        config = {'lr': 0.2}
        try:
            temp_dir = tempfile.mkdtemp()
            trainer = Estimator.from_keras(model_creator=model_creator, verbose=True, config=config, workers_per_node=1, backend='spark')
            res = trainer.fit(data=xshards, epochs=5, batch_size=4, steps_per_epoch=25, feature_cols=['user', 'item'], label_cols=['label'])
            assert isinstance(res, dict), 'fit should return a dict'
            print('start saving')
            trainer.save_weights(os.path.join(temp_dir, 'cifar10_keras.h5'))
            trainer.load_weights(os.path.join(temp_dir, 'cifar10_keras.h5'))
            res = trainer.evaluate(data=xshards, num_steps=25, batch_size=4, feature_cols=['user', 'item'], label_cols=['label'])
            assert isinstance(res, dict), 'evaluate should return a dict'
            print('validation result: ', res)
            predictions = trainer.predict(data=xshards, batch_size=4, feature_cols=['user', 'item'])
            assert predictions._get_class_name() == 'pandas.core.frame.DataFrame'
            prediction_df = predictions.collect()[0]
            import pandas as pd
            assert isinstance(prediction_df, pd.DataFrame)
            assert 'prediction' in prediction_df
        finally:
            shutil.rmtree(temp_dir)

    def test_checkpoint_weights_h5(self):
        if False:
            return 10
        sc = OrcaContext.get_spark_context()
        rdd = sc.range(0, 100)
        spark = OrcaContext.get_spark_session()
        from pyspark.ml.linalg import DenseVector
        df = rdd.map(lambda x: (DenseVector(np.random.randn(1).astype(np.float32)), int(np.random.randint(0, 2, size=())))).toDF(['feature', 'label'])
        config = {'lr': 0.2}
        try:
            temp_dir = tempfile.mkdtemp()
            trainer = Estimator.from_keras(model_creator=model_creator, verbose=True, config=config, workers_per_node=2, backend='spark')
            callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(temp_dir, 'ckpt_weights.h5'), save_weights_only=True)]
            res = trainer.fit(df, epochs=3, batch_size=4, steps_per_epoch=25, callbacks=callbacks, feature_cols=['feature'], label_cols=['label'], validation_data=df, validation_steps=1)
            trainer.load_weights(os.path.join(temp_dir, 'ckpt_weights.h5'))
            res = trainer.evaluate(df, batch_size=4, num_steps=25, feature_cols=['feature'], label_cols=['label'])
            print('validation result: ', res)
            res = trainer.predict(df, feature_cols=['feature']).collect()
            print('predict result: ', res)
        finally:
            shutil.rmtree(temp_dir)

    def test_save_load_model_h5(self):
        if False:
            print('Hello World!')
        sc = OrcaContext.get_spark_context()
        rdd = sc.range(0, 100)
        spark = OrcaContext.get_spark_session()
        from pyspark.ml.linalg import DenseVector
        df = rdd.map(lambda x: (DenseVector(np.random.randn(1).astype(np.float32)), int(np.random.randint(0, 2, size=())))).toDF(['feature', 'label'])
        config = {'lr': 0.2}
        try:
            temp_dir = tempfile.mkdtemp()
            trainer = Estimator.from_keras(model_creator=model_creator, verbose=True, config=config, workers_per_node=2, backend='spark')
            res = trainer.fit(df, epochs=5, batch_size=4, steps_per_epoch=25, feature_cols=['feature'], label_cols=['label'], validation_data=df, validation_steps=1)
            print('start saving')
            trainer.save(os.path.join(temp_dir, 'a.h5'))
            res = trainer.evaluate(df, batch_size=4, num_steps=25, feature_cols=['feature'], label_cols=['label'])
            print('validation result: ', res)
            before_res = trainer.predict(df, feature_cols=['feature']).collect()
            expect_res = np.concatenate([part['prediction'] for part in before_res])
            trainer.load(os.path.join(temp_dir, 'a.h5'))
            after_res = trainer.predict(df, feature_cols=['feature']).collect()
            pred_res = np.concatenate([part['prediction'] for part in after_res])
            assert np.array_equal(expect_res, pred_res)
        finally:
            shutil.rmtree(temp_dir)

    def test_save_load_model_savemodel(self):
        if False:
            print('Hello World!')
        sc = OrcaContext.get_spark_context()
        rdd = sc.range(0, 100)
        spark = OrcaContext.get_spark_session()
        from pyspark.ml.linalg import DenseVector
        df = rdd.map(lambda x: (DenseVector(np.random.randn(1).astype(np.float32)), int(np.random.randint(0, 2, size=())))).toDF(['feature', 'label'])
        config = {'lr': 0.2}
        try:
            temp_dir = tempfile.mkdtemp()
            trainer = Estimator.from_keras(model_creator=model_creator, verbose=True, config=config, workers_per_node=2, backend='spark')
            res = trainer.fit(df, epochs=5, batch_size=4, steps_per_epoch=25, feature_cols=['feature'], label_cols=['label'], validation_data=df, validation_steps=1)
            print('start saving')
            trainer.save(os.path.join(temp_dir, 'saved_model'))
            res = trainer.evaluate(df, batch_size=4, num_steps=25, feature_cols=['feature'], label_cols=['label'])
            print('validation result: ', res)
            before_res = trainer.predict(df, feature_cols=['feature']).collect()
            expect_res = np.concatenate([part['prediction'] for part in before_res])
            trainer.load(os.path.join(temp_dir, 'saved_model'))
            after_res = trainer.predict(df, feature_cols=['feature']).collect()
            pred_res = np.concatenate([part['prediction'] for part in after_res])
            assert np.array_equal(expect_res, pred_res)
        finally:
            shutil.rmtree(temp_dir)

    def test_save_load_model_architecture(self):
        if False:
            while True:
                i = 10
        config = {'lr': 0.2}
        import uuid
        model_path = os.path.join(tempfile.gettempdir(), str(uuid.uuid1()) + '.json')
        try:
            model = simple_model(config)
            with open(model_path, 'w') as f:
                f.write(model.to_json())
            from bigdl.dllib.utils.file_utils import enable_hdfs_load

            @enable_hdfs_load
            def load_model_architecture(path):
                if False:
                    i = 10
                    return i + 15
                with open(path, 'rb') as f:
                    model = tf.keras.models.model_from_json(f.read())
                return model
            model_load = load_model_architecture(model_path)
            assert model.summary() == model_load.summary()
        finally:
            if os.path.exists(model_path):
                os.remove(model_path)

    def test_optional_model_creator_h5(self):
        if False:
            while True:
                i = 10
        sc = OrcaContext.get_spark_context()
        rdd = sc.range(0, 100)
        spark = OrcaContext.get_spark_session()
        from pyspark.ml.linalg import DenseVector
        df = rdd.map(lambda x: (DenseVector(np.random.randn(1).astype(np.float32)), int(np.random.randint(0, 2, size=())))).toDF(['feature', 'label'])
        config = {'lr': 0.2}
        try:
            temp_dir = tempfile.mkdtemp()
            trainer = Estimator.from_keras(model_creator=model_creator, verbose=True, config=config, workers_per_node=2, backend='spark')
            trainer.fit(df, epochs=5, batch_size=4, steps_per_epoch=25, feature_cols=['feature'], label_cols=['label'], validation_data=df, validation_steps=1)
            pre_model = trainer.get_model(set_weights=True)
            if hasattr(pre_model.optimizer, 'get_weights'):
                pre_opt_weights = pre_model.optimizer.get_weights()
            else:
                pre_opt_weights = [var.numpy() for var in pre_model.optimizer.variables()]
            trainer.save(os.path.join(temp_dir, 'saved_model.h5'))
            before_res = trainer.predict(df, feature_cols=['feature']).collect()
            expect_res = np.concatenate([part['prediction'] for part in before_res])
            trainer.shutdown()
            est = Estimator.from_keras(verbose=True, config=config, workers_per_node=2, backend='spark')
            est.load(os.path.join(temp_dir, 'saved_model.h5'))
            after_model = est.get_model(set_weights=True)
            if hasattr(after_model.optimizer, 'get_weights'):
                after_opt_weights = after_model.optimizer.get_weights()
            else:
                after_opt_weights = [var.numpy() for var in after_model.optimizer.variables()]
            after_res = est.predict(df, feature_cols=['feature']).collect()
            pred_res = np.concatenate([part['prediction'] for part in after_res])
            assert np.array_equal(expect_res, pred_res)
            for i in range(len(pre_opt_weights)):
                assert np.array_equal(pre_opt_weights[i], after_opt_weights[i])
            est.fit(df, epochs=5, batch_size=4, steps_per_epoch=25, feature_cols=['feature'], label_cols=['label'], validation_data=df, validation_steps=1)
            new_model = est.get_model(set_weights=True)
            if hasattr(new_model.optimizer, 'get_weights'):
                new_opt_weights = new_model.optimizer.get_weights()
            else:
                new_opt_weights = [var.numpy() for var in new_model.optimizer.variables()]
            assert not np.array_equal(after_opt_weights, new_opt_weights)
            res = est.evaluate(df, batch_size=4, num_steps=25, feature_cols=['feature'], label_cols=['label'])
            print('validation result: ', res)
        finally:
            shutil.rmtree(temp_dir)

    def test_optional_model_creator_savemodel(self):
        if False:
            for i in range(10):
                print('nop')
        sc = OrcaContext.get_spark_context()
        rdd = sc.range(0, 100)
        spark = OrcaContext.get_spark_session()
        from pyspark.ml.linalg import DenseVector
        df = rdd.map(lambda x: (DenseVector(np.random.randn(1).astype(np.float32)), int(np.random.randint(0, 2, size=())))).toDF(['feature', 'label'])
        config = {'lr': 0.2}
        try:
            temp_dir = tempfile.mkdtemp()
            trainer = Estimator.from_keras(model_creator=model_creator, verbose=True, config=config, workers_per_node=2, backend='spark')
            trainer.fit(df, epochs=5, batch_size=4, steps_per_epoch=25, feature_cols=['feature'], label_cols=['label'], validation_data=df, validation_steps=1)
            trainer.save(os.path.join(temp_dir, 'saved_model'))
            before_res = trainer.predict(df, feature_cols=['feature']).collect()
            expect_res = np.concatenate([part['prediction'] for part in before_res])
            trainer.shutdown()
            est = Estimator.from_keras(verbose=True, config=config, workers_per_node=2, backend='spark')
            est.load(os.path.join(temp_dir, 'saved_model'))
            after_res = est.predict(df, feature_cols=['feature']).collect()
            pred_res = np.concatenate([part['prediction'] for part in after_res])
            assert np.array_equal(expect_res, pred_res)
            est.fit(df, epochs=5, batch_size=4, steps_per_epoch=25, feature_cols=['feature'], label_cols=['label'], validation_data=df, validation_steps=1)
            res = est.evaluate(df, batch_size=4, num_steps=25, feature_cols=['feature'], label_cols=['label'])
            print('validation result: ', res)
        finally:
            shutil.rmtree(temp_dir)

    def test_get_model(self):
        if False:
            print('Hello World!')
        sc = OrcaContext.get_spark_context()
        rdd = sc.range(0, 100)
        spark = OrcaContext.get_spark_session()
        from pyspark.ml.linalg import DenseVector
        df = rdd.map(lambda x: (DenseVector(np.random.randn(1).astype(np.float32)), int(np.random.randint(0, 2, size=())))).toDF(['feature', 'label'])
        config = {'lr': 0.2}
        try:
            temp_dir = tempfile.mkdtemp()
            trainer = Estimator.from_keras(model_creator=model_creator, verbose=True, config=config, workers_per_node=3, backend='spark')
            trainer.fit(df, epochs=5, batch_size=4, steps_per_epoch=25, feature_cols=['feature'], label_cols=['label'], validation_data=df, validation_steps=1)
            trainer.save(os.path.join(temp_dir, 'cifar10.h5'))
            pre_model = trainer.get_model()
            if hasattr(pre_model.optimizer, 'get_weights'):
                pre_model_weights = pre_model.optimizer.get_weights()
            else:
                pre_model_weights = [var.numpy() for var in pre_model.optimizer.variables()]
            after_res = trainer.predict(df, feature_cols=['feature']).collect()
            expect_res = np.concatenate([part['prediction'] for part in after_res])
            trainer.shutdown()
            est = Estimator.from_keras(verbose=True, config=config, workers_per_node=3, backend='spark')
            est.load(os.path.join(temp_dir, 'cifar10.h5'))
            after_model = est.get_model()
            if hasattr(after_model.optimizer, 'get_weights'):
                after_model_weights = after_model.optimizer.get_weights()
            else:
                after_model_weights = [var.numpy() for var in after_model.optimizer.variables()]
            est.save(os.path.join(temp_dir, 'cifar10_new.h5'))
            after_res = est.predict(df, feature_cols=['feature']).collect()
            pred_res = np.concatenate([part['prediction'] for part in after_res])
            assert np.array_equal(expect_res, pred_res)
            for (pre_tensor, after_tensor) in list(zip(pre_model_weights, after_model_weights)):
                assert np.allclose(pre_tensor, after_tensor)
        finally:
            shutil.rmtree(temp_dir)
if __name__ == '__main__':
    pytest.main([__file__])