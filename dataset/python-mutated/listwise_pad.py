import os
import sys
import math
from optparse import OptionParser
import tensorflow as tf
import tensorflow_ranking as tfr
import tensorflow_recommenders as tfrs
from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.friesian.feature import FeatureTable
from bigdl.orca.learn.tf2 import Estimator
from bigdl.orca.data.tf.data import Dataset
from bigdl.dllib.utils.log4Error import invalidInputError
conf = {'spark.driver.maxResultSize': '10G'}

class RankingModel(tfrs.Model):

    def __init__(self, loss):
        if False:
            i = 10
            return i + 15
        super().__init__()
        embedding_dimension = 32
        self.user_embeddings = tf.keras.Sequential([tf.keras.layers.StringLookup(vocabulary=unique_user_ids), tf.keras.layers.Embedding(len(unique_user_ids) + 2, embedding_dimension)])
        self.movie_embeddings = tf.keras.Sequential([tf.keras.layers.StringLookup(vocabulary=unique_movie_titles), tf.keras.layers.Embedding(len(unique_movie_titles) + 2, embedding_dimension)])
        self.score_model = tf.keras.Sequential([tf.keras.layers.Dense(256, activation='relu'), tf.keras.layers.Dense(64, activation='relu'), tf.keras.layers.Dense(1)])
        self.task = tfrs.tasks.Ranking(loss=loss, metrics=[tfr.keras.metrics.NDCGMetric(name='ndcg_metric', ragged=True), tf.keras.metrics.RootMeanSquaredError()])

    def call(self, features, ragged_output=False):
        if False:
            return 10
        user_embeddings = self.user_embeddings(features['userid'])
        movie_embeddings = self.movie_embeddings(features['titles'])
        list_length = features['titles'].shape[1]
        user_embedding_repeated = tf.repeat(tf.expand_dims(user_embeddings, 1), [list_length], axis=1)
        concatenated_embeddings = tf.concat([user_embedding_repeated, movie_embeddings], 2)
        scores = self.score_model(concatenated_embeddings)
        scores = tf.squeeze(scores, axis=-1)
        if ragged_output:
            ragged_length = features['len']
            ragged_scores = tf.RaggedTensor.from_tensor(scores, ragged_length)
            return ragged_scores
        else:
            return scores

    def compute_loss(self, features, training=False):
        if False:
            for i in range(10):
                print('nop')
        labels = features.pop('ratings')
        ragged_length = features['len']
        scores = self(features, ragged_output=True)
        ragged_labels = tf.RaggedTensor.from_tensor(labels, ragged_length)
        return self.task(labels=ragged_labels, predictions=scores)
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('--cluster_mode', type=str, default='local', help='The cluster mode, such as local or yarn.')
    parser.add_option('--executor_cores', type=int, default=18, help='The executor core number.')
    parser.add_option('--executor_memory', type=str, default='20g', help='The executor memory.')
    parser.add_option('--num_executor', type=int, default=1, help='The number of executor.')
    parser.add_option('--driver_cores', type=int, default=4, help='The driver core number.')
    parser.add_option('--driver_memory', type=str, default='36g', help='The driver memory.')
    parser.add_option('--data_dir', dest='data_dir', default='./ml-1m', help='The directory to ml-1m .dat files')
    (options, args) = parser.parse_args(sys.argv)
    data_dir = options.data_dir
    if options.cluster_mode == 'local':
        init_orca_context('local', cores=options.executor_cores, memory=options.executor_memory, init_ray_on_spark=True, conf=conf)
    elif options.cluster_mode == 'yarn':
        init_orca_context('yarn-client', cores=options.executor_cores, num_nodes=options.num_executor, memory=options.executor_memory, driver_cores=options.driver_cores, driver_memory=options.driver_memory, init_ray_on_spark=True, conf=conf)
    else:
        invalidInputError(False, "cluster_mode should be 'local' or 'yarn', but got " + args.cluster_mode)
    (options, args) = parser.parse_args(sys.argv)
    data_dir = options.data_dir
    dataset = {'ratings': ['userid', 'movieid', 'rating', 'timestamp'], 'users': ['userid', 'gender', 'age', 'occupation', 'zip-code'], 'movies': ['movieid', 'title', 'genres']}
    tbl_dict = dict()
    for (data, cols) in dataset.items():
        tbl = FeatureTable.read_csv(os.path.join(data_dir, data + '.dat'), delimiter=':', header=False)
        tmp_cols = tbl.columns[::2]
        tbl = tbl.select(tmp_cols)
        col_dict = {c[0]: c[1] for c in zip(tmp_cols, cols)}
        tbl = tbl.rename(col_dict)
        tbl_dict[data] = tbl
    full_tbl = tbl_dict['ratings'].join(tbl_dict['movies'], 'movieid').dropna(columns=None).select(['userid', 'title', 'rating'])
    full_tbl = full_tbl.cast(['rating'], 'int')
    full_tbl = full_tbl.cast(['userid'], 'string')
    (train_tbl, test_tbl) = full_tbl.random_split([0.85, 0.15], seed=1)

    def preprocess(tbl, feature_cols):
        if False:
            return 10
        col_dict = {'collect_list(' + c + ')': c + 's' for c in feature_cols}
        tbl = tbl.group_by('userid', agg='collect_list')
        tbl = tbl.rename(col_dict)
        arr_len = lambda x: len(x)
        tbl = tbl.apply('ratings', 'len', arr_len, dtype='int')
        (min_len, max_len) = tbl.get_stats('len', ['min', 'max'])['len']
        print('max_min_len', (max_len, min_len))
        tbl = tbl.pad(['ratings'], max_len, mask_token=-1)
        tbl = tbl.pad(['titles'], max_len, mask_token='<MSK>')
        return tbl
    train_tbl = preprocess(train_tbl, ['title', 'rating'])
    test_tbl = preprocess(test_tbl, ['title', 'rating'])
    train_tbl.sort('len', ascending=False).show(3, truncate=False)
    train_count = train_tbl.size()
    steps = math.ceil(train_count / 256)
    print('train size: ', train_count, ', steps: ', steps)
    test_count = test_tbl.size()
    test_steps = math.ceil(test_count / 256)
    print('test size: ', test_count, ', steps: ', test_steps)
    train_dataset = Dataset.from_feature_table(train_tbl)
    test_dataset = Dataset.from_feature_table(test_tbl)
    unique_movie_titles = tbl_dict['movies'].get_vocabularies(['title'])['title']
    tbl_dict['users'] = tbl_dict['users'].cast('userid', 'string')
    unique_user_ids = tbl_dict['users'].get_vocabularies(['userid'])['userid']
    print(len(unique_movie_titles), len(unique_user_ids))
    print(unique_movie_titles[0:2])
    print(unique_user_ids[0:2])

    def model_creator(config):
        if False:
            while True:
                i = 10
        model = RankingModel(tfr.keras.losses.ListMLELoss(ragged=True))
        from bigdl.friesian.models import TFRSModel
        model = TFRSModel(model)
        model.compile(optimizer=tf.keras.optimizers.Adagrad(config['lr']))
        return model
    config = {'lr': 0.1}
    est = Estimator.from_keras(model_creator=model_creator, verbose=True, config=config, backend='ray')
    est.fit(train_dataset, 16, batch_size=256, steps_per_epoch=steps, validation_data=test_dataset, validation_steps=test_steps)
    est.evaluate(test_dataset, 256, num_steps=test_steps)
    pred_tbl = train_tbl
    pred_dataset = Dataset.from_feature_table(pred_tbl)

    def del_ratings(d):
        if False:
            while True:
                i = 10
        del d['ratings']
        return d
    pred_dataset = pred_dataset.map(del_ratings)
    pred_shards = est.predict(pred_dataset, min_partition_num=6)
    pred_collect = pred_shards.collect()
    est.shutdown()
    stop_orca_context()