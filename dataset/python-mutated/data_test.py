"""Test NCF data pipeline."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import defaultdict
import hashlib
import os
import mock
import numpy as np
import scipy.stats
import tensorflow as tf
from official.recommendation import constants as rconst
from official.recommendation import data_preprocessing
from official.recommendation import movielens
from official.recommendation import popen_helper
from official.utils.misc import keras_utils
DATASET = 'ml-test'
NUM_USERS = 1000
NUM_ITEMS = 2000
NUM_PTS = 50000
BATCH_SIZE = 2048
EVAL_BATCH_SIZE = 4000
NUM_NEG = 4
END_TO_END_TRAIN_MD5 = 'b218738e915e825d03939c5e305a2698'
END_TO_END_EVAL_MD5 = 'd753d0f3186831466d6e218163a9501e'
FRESH_RANDOMNESS_MD5 = '63d0dff73c0e5f1048fbdc8c65021e22'

def mock_download(*args, **kwargs):
    if False:
        print('Hello World!')
    return

@mock.patch.object(popen_helper, 'get_forkpool', popen_helper.get_fauxpool)
class BaseTest(tf.test.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        if keras_utils.is_v2_0:
            tf.compat.v1.disable_eager_execution()
        self.temp_data_dir = self.get_temp_dir()
        ratings_folder = os.path.join(self.temp_data_dir, DATASET)
        tf.io.gfile.makedirs(ratings_folder)
        np.random.seed(0)
        raw_user_ids = np.arange(NUM_USERS * 3)
        np.random.shuffle(raw_user_ids)
        raw_user_ids = raw_user_ids[:NUM_USERS]
        raw_item_ids = np.arange(NUM_ITEMS * 3)
        np.random.shuffle(raw_item_ids)
        raw_item_ids = raw_item_ids[:NUM_ITEMS]
        users = np.random.choice(raw_user_ids, NUM_PTS)
        items = np.random.choice(raw_item_ids, NUM_PTS)
        scores = np.random.randint(low=0, high=5, size=NUM_PTS)
        times = np.random.randint(low=1000000000, high=1200000000, size=NUM_PTS)
        self.rating_file = os.path.join(ratings_folder, movielens.RATINGS_FILE)
        self.seen_pairs = set()
        self.holdout = {}
        with tf.io.gfile.GFile(self.rating_file, 'w') as f:
            f.write('user_id,item_id,rating,timestamp\n')
            for (usr, itm, scr, ts) in zip(users, items, scores, times):
                pair = (usr, itm)
                if pair in self.seen_pairs:
                    continue
                self.seen_pairs.add(pair)
                if usr not in self.holdout or (ts, itm) > self.holdout[usr]:
                    self.holdout[usr] = (ts, itm)
                f.write('{},{},{},{}\n'.format(usr, itm, scr, ts))
        movielens.download = mock_download
        movielens.NUM_RATINGS[DATASET] = NUM_PTS
        data_preprocessing.DATASET_TO_NUM_USERS_AND_ITEMS[DATASET] = (NUM_USERS, NUM_ITEMS)

    def make_params(self, train_epochs=1):
        if False:
            i = 10
            return i + 15
        return {'train_epochs': train_epochs, 'batches_per_step': 1, 'use_seed': False, 'batch_size': BATCH_SIZE, 'eval_batch_size': EVAL_BATCH_SIZE, 'num_neg': NUM_NEG, 'match_mlperf': True, 'use_tpu': False, 'use_xla_for_gpu': False, 'stream_files': False}

    def test_preprocessing(self):
        if False:
            for i in range(10):
                print('nop')
        cache_path = os.path.join(self.temp_data_dir, 'test_cache.pickle')
        (data, valid_cache) = data_preprocessing._filter_index_sort(self.rating_file, cache_path=cache_path)
        assert len(data[rconst.USER_MAP]) == NUM_USERS
        assert len(data[rconst.ITEM_MAP]) == NUM_ITEMS

    def drain_dataset(self, dataset, g):
        if False:
            i = 10
            return i + 15
        with self.session(graph=g) as sess:
            with g.as_default():
                batch = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
            output = []
            while True:
                try:
                    output.append(sess.run(batch))
                except tf.errors.OutOfRangeError:
                    break
        return output

    def _test_end_to_end(self, constructor_type):
        if False:
            while True:
                i = 10
        params = self.make_params(train_epochs=1)
        (_, _, producer) = data_preprocessing.instantiate_pipeline(dataset=DATASET, data_dir=self.temp_data_dir, params=params, constructor_type=constructor_type, deterministic=True)
        producer.start()
        producer.join()
        assert producer._fatal_exception is None
        user_inv_map = {v: k for (k, v) in producer.user_map.items()}
        item_inv_map = {v: k for (k, v) in producer.item_map.items()}
        g = tf.Graph()
        with g.as_default():
            input_fn = producer.make_input_fn(is_training=True)
            dataset = input_fn(params)
        first_epoch = self.drain_dataset(dataset=dataset, g=g)
        counts = defaultdict(int)
        train_examples = {True: set(), False: set()}
        md5 = hashlib.md5()
        for (features, labels) in first_epoch:
            data_list = [features[movielens.USER_COLUMN].flatten(), features[movielens.ITEM_COLUMN].flatten(), features[rconst.VALID_POINT_MASK].flatten(), labels.flatten()]
            for i in data_list:
                md5.update(i.tobytes())
            for (u, i, v, l) in zip(*data_list):
                if not v:
                    continue
                u_raw = user_inv_map[u]
                i_raw = item_inv_map[i]
                if ((u_raw, i_raw) in self.seen_pairs) != l:
                    assert not l
                    self.assertEqual(i_raw, self.holdout[u_raw][1])
                train_examples[l].add((u_raw, i_raw))
                counts[u_raw, i_raw] += 1
        self.assertRegexpMatches(md5.hexdigest(), END_TO_END_TRAIN_MD5)
        num_positives_seen = len(train_examples[True])
        self.assertEqual(producer._train_pos_users.shape[0], num_positives_seen)
        self.assertGreater(len(train_examples[False]) / NUM_NEG / num_positives_seen, 0.9)
        self.assertLess(np.mean(list(counts.values())), 1.1)
        with g.as_default():
            input_fn = producer.make_input_fn(is_training=False)
            dataset = input_fn(params)
        eval_data = self.drain_dataset(dataset=dataset, g=g)
        current_user = None
        md5 = hashlib.md5()
        for features in eval_data:
            data_list = [features[movielens.USER_COLUMN].flatten(), features[movielens.ITEM_COLUMN].flatten(), features[rconst.DUPLICATE_MASK].flatten()]
            for i in data_list:
                md5.update(i.tobytes())
            for (idx, (u, i, d)) in enumerate(zip(*data_list)):
                u_raw = user_inv_map[u]
                i_raw = item_inv_map[i]
                if current_user is None:
                    current_user = u
                self.assertEqual(u, current_user)
                if not (idx + 1) % (rconst.NUM_EVAL_NEGATIVES + 1):
                    self.assertEqual(i_raw, self.holdout[u_raw][1])
                    current_user = None
                elif i_raw == self.holdout[u_raw][1]:
                    assert d
                else:
                    assert (u_raw, i_raw) not in self.seen_pairs
        self.assertRegexpMatches(md5.hexdigest(), END_TO_END_EVAL_MD5)

    def _test_fresh_randomness(self, constructor_type):
        if False:
            for i in range(10):
                print('nop')
        train_epochs = 5
        params = self.make_params(train_epochs=train_epochs)
        (_, _, producer) = data_preprocessing.instantiate_pipeline(dataset=DATASET, data_dir=self.temp_data_dir, params=params, constructor_type=constructor_type, deterministic=True)
        producer.start()
        results = []
        g = tf.Graph()
        with g.as_default():
            for _ in range(train_epochs):
                input_fn = producer.make_input_fn(is_training=True)
                dataset = input_fn(params)
                results.extend(self.drain_dataset(dataset=dataset, g=g))
        producer.join()
        assert producer._fatal_exception is None
        (positive_counts, negative_counts) = (defaultdict(int), defaultdict(int))
        md5 = hashlib.md5()
        for (features, labels) in results:
            data_list = [features[movielens.USER_COLUMN].flatten(), features[movielens.ITEM_COLUMN].flatten(), features[rconst.VALID_POINT_MASK].flatten(), labels.flatten()]
            for i in data_list:
                md5.update(i.tobytes())
            for (u, i, v, l) in zip(*data_list):
                if not v:
                    continue
                if l:
                    positive_counts[u, i] += 1
                else:
                    negative_counts[u, i] += 1
        self.assertRegexpMatches(md5.hexdigest(), FRESH_RANDOMNESS_MD5)
        self.assertAllEqual(list(positive_counts.values()), [train_epochs for _ in positive_counts])
        pair_cardinality = NUM_USERS * NUM_ITEMS
        neg_pair_cardinality = pair_cardinality - len(self.seen_pairs)
        e_sample = len(self.seen_pairs) * NUM_NEG / neg_pair_cardinality
        approx_pdf = scipy.stats.binom.pmf(k=np.arange(train_epochs + 1), n=train_epochs, p=e_sample)
        count_distribution = [0 for _ in range(train_epochs + 1)]
        for i in negative_counts.values():
            i = min([i, train_epochs])
            count_distribution[i] += 1
        count_distribution[0] = neg_pair_cardinality - sum(count_distribution[1:])
        for i in range(train_epochs + 1):
            if approx_pdf[i] < 0.05:
                continue
            observed_fraction = count_distribution[i] / neg_pair_cardinality
            deviation = 2 * abs(observed_fraction - approx_pdf[i]) / (observed_fraction + approx_pdf[i])
            self.assertLess(deviation, 0.2)

    def test_end_to_end_materialized(self):
        if False:
            i = 10
            return i + 15
        self._test_end_to_end('materialized')

    def test_end_to_end_bisection(self):
        if False:
            i = 10
            return i + 15
        self._test_end_to_end('bisection')

    def test_fresh_randomness_materialized(self):
        if False:
            return 10
        self._test_fresh_randomness('materialized')

    def test_fresh_randomness_bisection(self):
        if False:
            return 10
        self._test_fresh_randomness('bisection')
if __name__ == '__main__':
    tf.test.main()