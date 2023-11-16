import os
import shutil
import tempfile
from unittest import TestCase
import numpy as np
import tensorflow as tf
from bigdl.orca.learn.trigger import SeveralIteration
from pyspark.sql.context import SQLContext
import bigdl.orca.data.pandas
from bigdl.dllib.nncontext import init_nncontext
from bigdl.orca import OrcaContext
from bigdl.orca.data.tf.data import Dataset
from bigdl.orca.learn.tf.estimator import Estimator
from bigdl.dllib.utils.tf import save_tf_checkpoint, load_tf_checkpoint, get_checkpoint_state
resource_path = os.path.join(os.path.split(__file__)[0], '../../resources')

class SimpleModel(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.user = tf.placeholder(dtype=tf.int32, shape=(None,))
        self.item = tf.placeholder(dtype=tf.int32, shape=(None,))
        self.label = tf.placeholder(dtype=tf.int32, shape=(None,))
        feat = tf.stack([self.user, self.item], axis=1)
        self.logits = tf.layers.dense(tf.to_float(feat), 2)
        self.loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=self.logits, labels=self.label))

class TestEstimatorForGraph(TestCase):

    def setup_method(self, method):
        if False:
            while True:
                i = 10
        OrcaContext.train_data_store = 'DRAM'

    def test_estimator_graph(self):
        if False:
            for i in range(10):
                print('nop')
        import bigdl.orca.data.pandas
        tf.reset_default_graph()
        model = SimpleModel()
        file_path = os.path.join(resource_path, 'orca/learn/ncf.csv')
        data_shard = bigdl.orca.data.pandas.read_csv(file_path)

        def transform(df):
            if False:
                print('Hello World!')
            result = {'x': (df['user'].to_numpy(), df['item'].to_numpy()), 'y': df['label'].to_numpy()}
            return result
        data_shard = data_shard.transform_shard(transform)
        est = Estimator.from_graph(inputs=[model.user, model.item], labels=[model.label], outputs=[model.logits], loss=model.loss, optimizer=tf.train.AdamOptimizer(), metrics={'loss': model.loss})
        est.fit(data=data_shard, batch_size=8, epochs=10, validation_data=data_shard)
        data_shard = bigdl.orca.data.pandas.read_csv(file_path)

        def transform(df):
            if False:
                for i in range(10):
                    print('nop')
            result = {'x': (df['user'].to_numpy(), df['item'].to_numpy())}
            return result
        data_shard = data_shard.transform_shard(transform)
        predictions = est.predict(data_shard).collect()
        assert 'prediction' in predictions[0]
        print(predictions)

    def test_estimator_graph_fit(self):
        if False:
            while True:
                i = 10
        import bigdl.orca.data.pandas
        tf.reset_default_graph()
        model = SimpleModel()
        file_path = os.path.join(resource_path, 'orca/learn/ncf.csv')
        data_shard = bigdl.orca.data.pandas.read_csv(file_path)

        def transform(df):
            if False:
                while True:
                    i = 10
            result = {'x': (df['user'].to_numpy(), df['item'].to_numpy()), 'y': df['label'].to_numpy()}
            return result
        data_shard = data_shard.transform_shard(transform)
        est = Estimator.from_graph(inputs=[model.user, model.item], labels=[model.label], loss=model.loss, optimizer=tf.train.AdamOptimizer(), metrics={'loss': model.loss})
        est.fit(data=data_shard, batch_size=8, epochs=10, validation_data=data_shard)

    def test_estimator_graph_evaluate(self):
        if False:
            i = 10
            return i + 15
        import bigdl.orca.data.pandas
        tf.reset_default_graph()
        model = SimpleModel()
        file_path = os.path.join(resource_path, 'orca/learn/ncf.csv')
        data_shard = bigdl.orca.data.pandas.read_csv(file_path)

        def transform(df):
            if False:
                return 10
            result = {'x': (df['user'].to_numpy(), df['item'].to_numpy()), 'y': df['label'].to_numpy()}
            return result
        data_shard = data_shard.transform_shard(transform)
        est = Estimator.from_graph(inputs=[model.user, model.item], labels=[model.label], loss=model.loss, optimizer=tf.train.AdamOptimizer(), metrics={'loss': model.loss})
        result = est.evaluate(data_shard)
        assert 'loss' in result
        print(result)

    def test_estimator_graph_predict(self):
        if False:
            print('Hello World!')
        import bigdl.orca.data.pandas
        tf.reset_default_graph()
        model = SimpleModel()
        file_path = os.path.join(resource_path, 'orca/learn/ncf.csv')
        data_shard = bigdl.orca.data.pandas.read_csv(file_path)
        est = Estimator.from_graph(inputs=[model.user, model.item], outputs=[model.logits])

        def transform(df):
            if False:
                i = 10
                return i + 15
            result = {'x': (df['user'].to_numpy(), df['item'].to_numpy())}
            return result
        data_shard = data_shard.transform_shard(transform)
        predictions = est.predict(data_shard).collect()
        print(predictions)

    def test_estimator_graph_pandas_dataframe(self):
        if False:
            return 10
        import bigdl.orca.data.pandas
        tf.reset_default_graph()
        model = SimpleModel()
        file_path = os.path.join(resource_path, 'orca/learn/ncf.csv')
        data_shard = bigdl.orca.data.pandas.read_csv(file_path)
        est = Estimator.from_graph(inputs=[model.user, model.item], labels=[model.label], loss=model.loss, optimizer=tf.train.AdamOptimizer(), metrics={'loss': model.loss})
        est.fit(data=data_shard, batch_size=8, epochs=10, feature_cols=['user', 'item'], label_cols=['label'], validation_data=data_shard)
        result = est.evaluate(data_shard, feature_cols=['user', 'item'], label_cols=['label'])
        assert 'loss' in result
        print(result)
        est = Estimator.from_graph(inputs=[model.user, model.item], outputs=[model.logits])
        predictions = est.predict(data_shard, feature_cols=['user', 'item']).collect()
        print(predictions)

    def test_estimator_graph_fit_clip(self):
        if False:
            return 10
        import bigdl.orca.data.pandas
        tf.reset_default_graph()
        model = SimpleModel()
        file_path = os.path.join(resource_path, 'orca/learn/ncf.csv')
        data_shard = bigdl.orca.data.pandas.read_csv(file_path)

        def transform(df):
            if False:
                i = 10
                return i + 15
            result = {'x': (df['user'].to_numpy(), df['item'].to_numpy()), 'y': df['label'].to_numpy()}
            return result
        data_shard = data_shard.transform_shard(transform)
        est = Estimator.from_graph(inputs=[model.user, model.item], labels=[model.label], loss=model.loss, optimizer=tf.train.AdamOptimizer(), clip_norm=1.2, metrics={'loss': model.loss})
        est.fit(data=data_shard, batch_size=8, epochs=10, validation_data=data_shard)
        est = Estimator.from_graph(inputs=[model.user, model.item], labels=[model.label], loss=model.loss, optimizer=tf.train.AdamOptimizer(), clip_value=0.2, metrics={'loss': model.loss})
        est.fit(data=data_shard, batch_size=8, epochs=10, validation_data=data_shard)

    def test_estimator_graph_checkpoint(self):
        if False:
            print('Hello World!')
        import bigdl.orca.data.pandas
        tf.reset_default_graph()
        model = SimpleModel()
        file_path = os.path.join(resource_path, 'orca/learn/ncf.csv')
        data_shard = bigdl.orca.data.pandas.read_csv(file_path)

        def transform(df):
            if False:
                while True:
                    i = 10
            result = {'x': (df['user'].to_numpy(), df['item'].to_numpy()), 'y': df['label'].to_numpy()}
            return result
        data_shard = data_shard.transform_shard(transform)
        temp = tempfile.mkdtemp()
        model_dir = os.path.join(temp, 'test_model')
        est = Estimator.from_graph(inputs=[model.user, model.item], labels=[model.label], loss=model.loss, optimizer=tf.train.AdamOptimizer(), metrics={'loss': model.loss}, model_dir=model_dir)
        est.fit(data=data_shard, batch_size=8, epochs=6, validation_data=data_shard, checkpoint_trigger=SeveralIteration(4))
        est.sess.close()
        tf.reset_default_graph()
        model = SimpleModel()
        est = Estimator.from_graph(inputs=[model.user, model.item], labels=[model.label], loss=model.loss, optimizer=tf.train.AdamOptimizer(), metrics={'loss': model.loss}, model_dir=model_dir)
        est.load_orca_checkpoint(model_dir)
        est.fit(data=data_shard, batch_size=8, epochs=10, validation_data=data_shard)
        result = est.evaluate(data_shard)
        assert 'loss' in result
        print(result)
        shutil.rmtree(temp)

    def test_estimator_graph_fit_dataset(self):
        if False:
            while True:
                i = 10
        import bigdl.orca.data.pandas
        tf.reset_default_graph()
        model = SimpleModel()
        file_path = os.path.join(resource_path, 'orca/learn/ncf.csv')
        data_shard = bigdl.orca.data.pandas.read_csv(file_path)

        def transform(df):
            if False:
                return 10
            result = {'x': (df['user'].to_numpy(), df['item'].to_numpy()), 'y': df['label'].to_numpy()}
            return result
        data_shard = data_shard.transform_shard(transform)
        dataset = Dataset.from_tensor_slices(data_shard)
        est = Estimator.from_graph(inputs=[model.user, model.item], labels=[model.label], loss=model.loss, optimizer=tf.train.AdamOptimizer(), metrics={'loss': model.loss})
        est.fit(data=dataset, batch_size=8, epochs=10, validation_data=dataset)
        result = est.evaluate(dataset, batch_size=4)
        assert 'loss' in result

    def test_estimator_graph_predict_dataset(self):
        if False:
            i = 10
            return i + 15
        tf.reset_default_graph()
        model = SimpleModel()
        file_path = os.path.join(resource_path, 'orca/learn/ncf.csv')
        data_shard = bigdl.orca.data.pandas.read_csv(file_path)
        est = Estimator.from_graph(inputs=[model.user, model.item], outputs=[model.logits])

        def transform(df):
            if False:
                while True:
                    i = 10
            result = {'x': (df['user'].to_numpy(), df['item'].to_numpy())}
            return result
        data_shard = data_shard.transform_shard(transform)
        dataset = Dataset.from_tensor_slices(data_shard)
        predictions = est.predict(dataset).collect()
        assert len(predictions) == 48

    def test_estimator_graph_dataframe(self):
        if False:
            print('Hello World!')
        tf.reset_default_graph()
        model = SimpleModel()
        file_path = os.path.join(resource_path, 'orca/learn/ncf.csv')
        sc = init_nncontext()
        sqlcontext = SQLContext(sc)
        df = sqlcontext.read.csv(file_path, header=True, inferSchema=True)
        est = Estimator.from_graph(inputs=[model.user, model.item], labels=[model.label], outputs=[model.logits], loss=model.loss, optimizer=tf.train.AdamOptimizer(), metrics={'loss': model.loss})
        est.fit(data=df, batch_size=8, epochs=10, feature_cols=['user', 'item'], label_cols=['label'], validation_data=df)
        result = est.evaluate(df, batch_size=4, feature_cols=['user', 'item'], label_cols=['label'])
        print(result)
        prediction_df = est.predict(df, batch_size=4, feature_cols=['user', 'item'])
        assert 'prediction' in prediction_df.columns
        predictions = prediction_df.collect()
        assert len(predictions) == 48

    def test_estimator_graph_dataframe_exception(self):
        if False:
            print('Hello World!')
        tf.reset_default_graph()
        model = SimpleModel()
        file_path = os.path.join(resource_path, 'orca/learn/ncf.csv')
        sc = init_nncontext()
        sqlcontext = SQLContext(sc)
        df = sqlcontext.read.csv(file_path, header=True, inferSchema=True)
        est = Estimator.from_graph(inputs=[model.user, model.item], labels=[model.label], outputs=[model.logits], loss=model.loss, optimizer=tf.train.AdamOptimizer(), metrics={'loss': model.loss})
        with self.assertRaises(Exception) as context:
            est.fit(data=df, batch_size=8, epochs=10, feature_cols=['user', 'item'], validation_data=df)
        self.assertTrue('label columns is None; it should not be None in training' in str(context.exception))
        est.fit(data=df, batch_size=8, epochs=10, feature_cols=['user', 'item'], label_cols=['label'])
        with self.assertRaises(Exception) as context:
            predictions = est.predict(df, batch_size=4).collect()
        self.assertTrue('feature columns is None; it should not be None in prediction' in str(context.exception))
        with self.assertRaises(Exception) as context:
            est.fit(data=df, batch_size=8, epochs=10, feature_cols=['user', 'item'], label_cols=['label'], validation_data=[1, 2, 3])
        self.assertTrue('train data and validation data should be both Spark DataFrame' in str(context.exception))

    def test_checkpoint_remote(self):
        if False:
            i = 10
            return i + 15
        tf.reset_default_graph()
        model = SimpleModel()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        temp = tempfile.mkdtemp()
        save_tf_checkpoint(sess, os.path.join(temp, 'simple.ckpt'), saver)
        ckpt = get_checkpoint_state(temp)
        assert ckpt.model_checkpoint_path == os.path.join(temp, 'simple.ckpt')
        assert ckpt.all_model_checkpoint_paths[0] == os.path.join(temp, 'simple.ckpt')
        load_tf_checkpoint(sess, os.path.join(temp, 'simple.ckpt'), saver)
        shutil.rmtree(temp)

    def _test_estimator_graph_tf_dataset(self, dataset_creator):
        if False:
            return 10
        tf.reset_default_graph()
        model = SimpleModel()
        dataset = dataset_creator()
        est = Estimator.from_graph(inputs=[model.user, model.item], labels=[model.label], outputs=[model.logits], loss=model.loss, optimizer=tf.train.AdamOptimizer(), metrics={'loss': model.loss})
        est.fit(data=dataset, batch_size=8, epochs=10, validation_data=dataset)
        result = est.evaluate(dataset, batch_size=4)
        assert 'loss' in result

    def test_estimator_graph_tf_dataset(self):
        if False:
            for i in range(10):
                print('nop')

        def dataset_creator():
            if False:
                for i in range(10):
                    print('nop')
            dataset = tf.data.Dataset.from_tensor_slices((np.random.randint(0, 200, size=(100,)), np.random.randint(0, 50, size=(100,)), np.ones(shape=(100,), dtype=np.int32)))
            return dataset
        self._test_estimator_graph_tf_dataset(dataset_creator)

    def test_estimator_graph_tf_dataset_v2(self):
        if False:
            return 10

        def dataset_creator():
            if False:
                while True:
                    i = 10
            dataset = tf.data.Dataset.from_tensor_slices((np.random.randint(0, 200, size=(100,)), np.random.randint(0, 50, size=(100,)), np.ones(shape=(100,), dtype=np.int32)))
            return dataset._dataset
        self._test_estimator_graph_tf_dataset(dataset_creator)

    def test_estimator_graph_tensorboard(self):
        if False:
            i = 10
            return i + 15
        tf.reset_default_graph()
        model = SimpleModel()
        file_path = os.path.join(resource_path, 'orca/learn/ncf.csv')
        data_shard = bigdl.orca.data.pandas.read_csv(file_path)

        def transform(df):
            if False:
                i = 10
                return i + 15
            result = {'x': (df['user'].to_numpy(), df['item'].to_numpy()), 'y': df['label'].to_numpy()}
            return result
        data_shard = data_shard.transform_shard(transform)
        temp = tempfile.mkdtemp()
        model_dir = os.path.join(temp, 'test_model')
        est = Estimator.from_graph(inputs=[model.user, model.item], labels=[model.label], loss=model.loss, optimizer=tf.train.AdamOptimizer(), metrics={'loss': model.loss}, model_dir=model_dir)
        est.fit(data=data_shard, batch_size=8, epochs=5, validation_data=data_shard)
        train_tp = est.get_train_summary('Throughput')
        val_scores = est.get_validation_summary('loss')
        assert len(train_tp) > 0
        assert len(val_scores) > 0
        est.set_tensorboard('model', 'test')
        est.fit(data=data_shard, batch_size=8, epochs=5, validation_data=data_shard)
        train_tp = est.get_train_summary('Throughput')
        val_scores = est.get_validation_summary('loss')
        assert len(train_tp) > 0
        assert len(val_scores) > 0
        est2 = Estimator.from_graph(inputs=[model.user, model.item], labels=[model.label], loss=model.loss, optimizer=tf.train.AdamOptimizer(), metrics={'loss': model.loss})
        est2.fit(data=data_shard, batch_size=8, epochs=5, validation_data=data_shard)
        train_tp = est2.get_train_summary('Throughput')
        val_scores = est2.get_validation_summary('loss')
        assert train_tp is None
        assert val_scores is None
        shutil.rmtree(temp)

    def test_estimator_graph_save_load(self):
        if False:
            while True:
                i = 10
        import bigdl.orca.data.pandas
        tf.reset_default_graph()
        model = SimpleModel()
        file_path = os.path.join(resource_path, 'orca/learn/ncf.csv')
        data_shard = bigdl.orca.data.pandas.read_csv(file_path)

        def transform(df):
            if False:
                i = 10
                return i + 15
            result = {'x': (df['user'].to_numpy(), df['item'].to_numpy()), 'y': df['label'].to_numpy()}
            return result
        data_shard = data_shard.transform_shard(transform)
        est = Estimator.from_graph(inputs=[model.user, model.item], labels=[model.label], outputs=[model.logits], loss=model.loss, optimizer=tf.train.AdamOptimizer(), metrics={'loss': model.loss}, sess=None)
        est.fit(data=data_shard, batch_size=8, epochs=10, validation_data=data_shard)
        temp = tempfile.mkdtemp()
        model_checkpoint = os.path.join(temp, 'test.ckpt')
        est.save_tf_checkpoint(model_checkpoint)
        est.sess.close()
        tf.reset_default_graph()
        with tf.Session() as sess:
            model = SimpleModel()
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, model_checkpoint)
            est = Estimator.from_graph(inputs=[model.user, model.item], labels=[model.label], outputs=[model.logits], loss=model.loss, metrics={'loss': model.loss}, sess=sess)
            data_shard = bigdl.orca.data.pandas.read_csv(file_path)

            def transform(df):
                if False:
                    print('Hello World!')
                result = {'x': (df['user'].to_numpy(), df['item'].to_numpy())}
                return result
            data_shard = data_shard.transform_shard(transform)
            predictions = est.predict(data_shard).collect()
            assert 'prediction' in predictions[0]
            print(predictions)
        shutil.rmtree(temp)

    def test_estimator_save_load(self):
        if False:
            print('Hello World!')
        import bigdl.orca.data.pandas
        tf.reset_default_graph()
        model = SimpleModel()
        file_path = os.path.join(resource_path, 'orca/learn/ncf.csv')
        data_shard = bigdl.orca.data.pandas.read_csv(file_path)

        def transform(df):
            if False:
                return 10
            result = {'x': (df['user'].to_numpy(), df['item'].to_numpy()), 'y': df['label'].to_numpy()}
            return result
        data_shard = data_shard.transform_shard(transform)
        est = Estimator.from_graph(inputs=[model.user, model.item], labels=[model.label], outputs=[model.logits], loss=model.loss, optimizer=tf.train.AdamOptimizer(), metrics={'loss': model.loss}, sess=None)
        est.fit(data=data_shard, batch_size=8, epochs=5, validation_data=data_shard)
        temp = tempfile.mkdtemp()
        model_checkpoint = os.path.join(temp, 'tmp.ckpt')
        est.save(model_checkpoint)
        est.shutdown()
        tf.reset_default_graph()
        with tf.Session() as sess:
            model = SimpleModel()
            est = Estimator.from_graph(inputs=[model.user, model.item], labels=[model.label], outputs=[model.logits], loss=model.loss, metrics={'loss': model.loss}, sess=sess)
            est.load(model_checkpoint)
            data_shard = bigdl.orca.data.pandas.read_csv(file_path)

            def transform(df):
                if False:
                    for i in range(10):
                        print('nop')
                result = {'x': (df['user'].to_numpy(), df['item'].to_numpy())}
                return result
            data_shard = data_shard.transform_shard(transform)
            predictions = est.predict(data_shard).collect()
            assert 'prediction' in predictions[0]
            print(predictions)
        shutil.rmtree(temp)

    def test_estimator_graph_with_bigdl_optim_method(self):
        if False:
            while True:
                i = 10
        import bigdl.orca.data.pandas
        tf.reset_default_graph()
        model = SimpleModel()
        file_path = os.path.join(resource_path, 'orca/learn/ncf.csv')
        data_shard = bigdl.orca.data.pandas.read_csv(file_path)

        def transform(df):
            if False:
                print('Hello World!')
            result = {'x': (df['user'].to_numpy(), df['item'].to_numpy()), 'y': df['label'].to_numpy()}
            return result
        data_shard = data_shard.transform_shard(transform)
        from bigdl.orca.learn.optimizers import SGD
        from bigdl.orca.learn.optimizers.schedule import Plateau
        sgd = SGD(learningrate=0.1, learningrate_schedule=Plateau('score', factor=0.1, patience=10, mode='min'))
        est = Estimator.from_graph(inputs=[model.user, model.item], labels=[model.label], outputs=[model.logits], loss=model.loss, optimizer=sgd, metrics={'loss': model.loss})
        est.fit(data=data_shard, batch_size=8, epochs=10, validation_data=data_shard)

    def test_estimator_graph_fit_mem_type(self):
        if False:
            print('Hello World!')
        import bigdl.orca.data.pandas
        tf.reset_default_graph()
        model = SimpleModel()
        file_path = os.path.join(resource_path, 'orca/learn/ncf.csv')
        data_shard = bigdl.orca.data.pandas.read_csv(file_path)

        def transform(df):
            if False:
                while True:
                    i = 10
            result = {'x': (df['user'].to_numpy(), df['item'].to_numpy()), 'y': df['label'].to_numpy()}
            return result
        data_shard = data_shard.transform_shard(transform)
        est = Estimator.from_graph(inputs=[model.user, model.item], labels=[model.label], loss=model.loss, optimizer=tf.train.AdamOptimizer(), metrics={'loss': model.loss})
        OrcaContext.train_data_store = 'DISK_2'
        est.fit(data=data_shard, batch_size=4, epochs=10, validation_data=data_shard)
        OrcaContext.train_data_store = 'DRAM'
if __name__ == '__main__':
    import pytest
    pytest.main([__file__])