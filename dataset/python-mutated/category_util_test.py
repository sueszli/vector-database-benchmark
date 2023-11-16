"""Tests for object_detection.utils.category_util."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
from object_detection.utils import category_util

class EvalUtilTest(tf.test.TestCase):

    def test_load_categories_from_csv_file(self):
        if False:
            for i in range(10):
                print('nop')
        csv_data = '\n        0,"cat"\n        1,"dog"\n        2,"bird"\n    '.strip(' ')
        csv_path = os.path.join(self.get_temp_dir(), 'test.csv')
        with tf.gfile.Open(csv_path, 'wb') as f:
            f.write(csv_data)
        categories = category_util.load_categories_from_csv_file(csv_path)
        self.assertTrue({'id': 0, 'name': 'cat'} in categories)
        self.assertTrue({'id': 1, 'name': 'dog'} in categories)
        self.assertTrue({'id': 2, 'name': 'bird'} in categories)

    def test_save_categories_to_csv_file(self):
        if False:
            i = 10
            return i + 15
        categories = [{'id': 0, 'name': 'cat'}, {'id': 1, 'name': 'dog'}, {'id': 2, 'name': 'bird'}]
        csv_path = os.path.join(self.get_temp_dir(), 'test.csv')
        category_util.save_categories_to_csv_file(categories, csv_path)
        saved_categories = category_util.load_categories_from_csv_file(csv_path)
        self.assertEqual(saved_categories, categories)
if __name__ == '__main__':
    tf.test.main()