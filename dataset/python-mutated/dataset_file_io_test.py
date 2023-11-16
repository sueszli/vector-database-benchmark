"""Tests for dataset file IO module."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
from delf.python.google_landmarks_dataset import dataset_file_io

class DatasetFileIoTest(tf.test.TestCase):

    def testReadRecognitionSolutionWorks(self):
        if False:
            print('Hello World!')
        file_path = os.path.join(tf.test.get_temp_dir(), 'recognition_solution.csv')
        with tf.gfile.GFile(file_path, 'w') as f:
            f.write('id,landmarks,Usage\n')
            f.write('0123456789abcdef,0 12,Public\n')
            f.write('0223456789abcdef,,Public\n')
            f.write('0323456789abcdef,100,Ignored\n')
            f.write('0423456789abcdef,1,Private\n')
            f.write('0523456789abcdef,,Ignored\n')
        (public_solution, private_solution, ignored_ids) = dataset_file_io.ReadSolution(file_path, dataset_file_io.RECOGNITION_TASK_ID)
        expected_public_solution = {'0123456789abcdef': [0, 12], '0223456789abcdef': []}
        expected_private_solution = {'0423456789abcdef': [1]}
        expected_ignored_ids = ['0323456789abcdef', '0523456789abcdef']
        self.assertEqual(public_solution, expected_public_solution)
        self.assertEqual(private_solution, expected_private_solution)
        self.assertEqual(ignored_ids, expected_ignored_ids)

    def testReadRetrievalSolutionWorks(self):
        if False:
            for i in range(10):
                print('nop')
        file_path = os.path.join(tf.test.get_temp_dir(), 'retrieval_solution.csv')
        with tf.gfile.GFile(file_path, 'w') as f:
            f.write('id,images,Usage\n')
            f.write('0123456789abcdef,None,Ignored\n')
            f.write('0223456789abcdef,fedcba9876543210 fedcba9876543200,Public\n')
            f.write('0323456789abcdef,fedcba9876543200,Private\n')
            f.write('0423456789abcdef,fedcba9876543220,Private\n')
            f.write('0523456789abcdef,None,Ignored\n')
        (public_solution, private_solution, ignored_ids) = dataset_file_io.ReadSolution(file_path, dataset_file_io.RETRIEVAL_TASK_ID)
        expected_public_solution = {'0223456789abcdef': ['fedcba9876543210', 'fedcba9876543200']}
        expected_private_solution = {'0323456789abcdef': ['fedcba9876543200'], '0423456789abcdef': ['fedcba9876543220']}
        expected_ignored_ids = ['0123456789abcdef', '0523456789abcdef']
        self.assertEqual(public_solution, expected_public_solution)
        self.assertEqual(private_solution, expected_private_solution)
        self.assertEqual(ignored_ids, expected_ignored_ids)

    def testReadRecognitionPredictionsWorks(self):
        if False:
            while True:
                i = 10
        file_path = os.path.join(tf.test.get_temp_dir(), 'recognition_predictions.csv')
        with tf.gfile.GFile(file_path, 'w') as f:
            f.write('id,landmarks\n')
            f.write('0123456789abcdef,12 0.1 \n')
            f.write('0423456789abcdef,0 19.0\n')
            f.write('0223456789abcdef,\n')
            f.write('\n')
            f.write('0523456789abcdef,14 0.01\n')
        public_ids = ['0123456789abcdef', '0223456789abcdef']
        private_ids = ['0423456789abcdef']
        ignored_ids = ['0323456789abcdef', '0523456789abcdef']
        (public_predictions, private_predictions) = dataset_file_io.ReadPredictions(file_path, public_ids, private_ids, ignored_ids, dataset_file_io.RECOGNITION_TASK_ID)
        expected_public_predictions = {'0123456789abcdef': {'class': 12, 'score': 0.1}}
        expected_private_predictions = {'0423456789abcdef': {'class': 0, 'score': 19.0}}
        self.assertEqual(public_predictions, expected_public_predictions)
        self.assertEqual(private_predictions, expected_private_predictions)

    def testReadRetrievalPredictionsWorks(self):
        if False:
            return 10
        file_path = os.path.join(tf.test.get_temp_dir(), 'retrieval_predictions.csv')
        with tf.gfile.GFile(file_path, 'w') as f:
            f.write('id,images\n')
            f.write('0123456789abcdef,fedcba9876543250 \n')
            f.write('0423456789abcdef,fedcba9876543260\n')
            f.write('0223456789abcdef,fedcba9876543210 fedcba9876543200 fedcba9876543220\n')
            f.write('\n')
            f.write('0523456789abcdef,\n')
        public_ids = ['0223456789abcdef']
        private_ids = ['0323456789abcdef', '0423456789abcdef']
        ignored_ids = ['0123456789abcdef', '0523456789abcdef']
        (public_predictions, private_predictions) = dataset_file_io.ReadPredictions(file_path, public_ids, private_ids, ignored_ids, dataset_file_io.RETRIEVAL_TASK_ID)
        expected_public_predictions = {'0223456789abcdef': ['fedcba9876543210', 'fedcba9876543200', 'fedcba9876543220']}
        expected_private_predictions = {'0423456789abcdef': ['fedcba9876543260']}
        self.assertEqual(public_predictions, expected_public_predictions)
        self.assertEqual(private_predictions, expected_private_predictions)
if __name__ == '__main__':
    tf.test.main()