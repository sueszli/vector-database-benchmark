from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import unittest
import numpy as np
import tensorflow as tf
from official.recommendation import movielens
from official.utils.misc import keras_utils
from official.utils.testing import integration
from official.r1.wide_deep import movielens_dataset
from official.r1.wide_deep import movielens_main
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
TEST_INPUT_VALUES = {'genres': np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 'user_id': [3], 'item_id': [4]}
TEST_ITEM_DATA = "item_id,titles,genres\n1,Movie_1,Comedy|Romance\n2,Movie_2,Adventure|Children's\n3,Movie_3,Comedy|Drama\n4,Movie_4,Comedy\n5,Movie_5,Action|Crime|Thriller\n6,Movie_6,Action\n7,Movie_7,Action|Adventure|Thriller"
TEST_RATING_DATA = 'user_id,item_id,rating,timestamp\n1,2,5,978300760\n1,3,3,978302109\n1,6,3,978301968\n2,1,4,978300275\n2,7,5,978824291\n3,1,3,978302268\n3,4,5,978302039\n3,5,5,978300719\n'

class BaseTest(tf.test.TestCase):
    """Tests for Wide Deep model."""

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        super(BaseTest, cls).setUpClass()
        movielens_main.define_movie_flags()

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.temp_dir = self.get_temp_dir()
        tf.io.gfile.makedirs(os.path.join(self.temp_dir, movielens.ML_1M))
        self.ratings_csv = os.path.join(self.temp_dir, movielens.ML_1M, movielens.RATINGS_FILE)
        self.item_csv = os.path.join(self.temp_dir, movielens.ML_1M, movielens.MOVIES_FILE)
        with tf.io.gfile.GFile(self.ratings_csv, 'w') as f:
            f.write(TEST_RATING_DATA)
        with tf.io.gfile.GFile(self.item_csv, 'w') as f:
            f.write(TEST_ITEM_DATA)

    @unittest.skipIf(keras_utils.is_v2_0(), 'TF 1.0 only test.')
    def test_input_fn(self):
        if False:
            while True:
                i = 10
        (train_input_fn, _, _) = movielens_dataset.construct_input_fns(dataset=movielens.ML_1M, data_dir=self.temp_dir, batch_size=8, repeat=1)
        dataset = train_input_fn()
        (features, labels) = dataset.make_one_shot_iterator().get_next()
        with self.session() as sess:
            (features, labels) = sess.run((features, labels))
            for key in TEST_INPUT_VALUES:
                self.assertTrue(key in features)
                self.assertAllClose(TEST_INPUT_VALUES[key], features[key][0])
            self.assertAllClose(labels[0], [1.0])

    @unittest.skipIf(keras_utils.is_v2_0(), 'TF 1.0 only test.')
    def test_end_to_end_deep(self):
        if False:
            for i in range(10):
                print('nop')
        integration.run_synthetic(main=movielens_main.main, tmp_root=self.temp_dir, extra_flags=['--data_dir', self.temp_dir, '--download_if_missing=false', '--train_epochs', '1', '--epochs_between_evals', '1'], synth=False)
if __name__ == '__main__':
    tf.test.main()