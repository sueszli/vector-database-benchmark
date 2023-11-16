"""Tests for Google Landmarks dataset metric computation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from delf.python.google_landmarks_dataset import metrics

def _CreateRecognitionSolution():
    if False:
        print('Hello World!')
    'Creates recognition solution to be used in tests.\n\n  Returns:\n    solution: Dict mapping test image ID to list of ground-truth landmark IDs.\n  '
    return {'0123456789abcdef': [0, 12], '0223456789abcdef': [100, 200, 300], '0323456789abcdef': [1], '0423456789abcdef': [], '0523456789abcdef': []}

def _CreateRecognitionPredictions():
    if False:
        while True:
            i = 10
    "Creates recognition predictions to be used in tests.\n\n  Returns:\n    predictions: Dict mapping test image ID to a dict with keys 'class'\n      (integer) and 'score' (float).\n  "
    return {'0223456789abcdef': {'class': 0, 'score': 0.01}, '0323456789abcdef': {'class': 1, 'score': 10.0}, '0423456789abcdef': {'class': 150, 'score': 15.0}}

def _CreateRetrievalSolution():
    if False:
        for i in range(10):
            print('nop')
    'Creates retrieval solution to be used in tests.\n\n  Returns:\n    solution: Dict mapping test image ID to list of ground-truth image IDs.\n  '
    return {'0123456789abcdef': ['fedcba9876543210', 'fedcba9876543220'], '0223456789abcdef': ['fedcba9876543210'], '0323456789abcdef': ['fedcba9876543230', 'fedcba9876543240', 'fedcba9876543250'], '0423456789abcdef': ['fedcba9876543230']}

def _CreateRetrievalPredictions():
    if False:
        return 10
    'Creates retrieval predictions to be used in tests.\n\n  Returns:\n    predictions: Dict mapping test image ID to a list with predicted index image\n    ids.\n  '
    return {'0223456789abcdef': ['fedcba9876543200', 'fedcba9876543210'], '0323456789abcdef': ['fedcba9876543240'], '0423456789abcdef': ['fedcba9876543230', 'fedcba9876543240']}

class MetricsTest(tf.test.TestCase):

    def testGlobalAveragePrecisionWorks(self):
        if False:
            print('Hello World!')
        predictions = _CreateRecognitionPredictions()
        solution = _CreateRecognitionSolution()
        gap = metrics.GlobalAveragePrecision(predictions, solution)
        expected_gap = 0.166667
        self.assertAllClose(gap, expected_gap)

    def testGlobalAveragePrecisionIgnoreNonGroundTruthWorks(self):
        if False:
            for i in range(10):
                print('nop')
        predictions = _CreateRecognitionPredictions()
        solution = _CreateRecognitionSolution()
        gap = metrics.GlobalAveragePrecision(predictions, solution, ignore_non_gt_test_images=True)
        expected_gap = 0.333333
        self.assertAllClose(gap, expected_gap)

    def testTop1AccuracyWorks(self):
        if False:
            for i in range(10):
                print('nop')
        predictions = _CreateRecognitionPredictions()
        solution = _CreateRecognitionSolution()
        accuracy = metrics.Top1Accuracy(predictions, solution)
        expected_accuracy = 0.333333
        self.assertAllClose(accuracy, expected_accuracy)

    def testMeanAveragePrecisionWorks(self):
        if False:
            for i in range(10):
                print('nop')
        predictions = _CreateRetrievalPredictions()
        solution = _CreateRetrievalSolution()
        mean_ap = metrics.MeanAveragePrecision(predictions, solution)
        expected_mean_ap = 0.458333
        self.assertAllClose(mean_ap, expected_mean_ap)

    def testMeanAveragePrecisionMaxPredictionsWorks(self):
        if False:
            return 10
        predictions = _CreateRetrievalPredictions()
        solution = _CreateRetrievalSolution()
        mean_ap = metrics.MeanAveragePrecision(predictions, solution, max_predictions=1)
        expected_mean_ap = 0.5
        self.assertAllClose(mean_ap, expected_mean_ap)

    def testMeanPrecisionsWorks(self):
        if False:
            print('Hello World!')
        predictions = _CreateRetrievalPredictions()
        solution = _CreateRetrievalSolution()
        mean_precisions = metrics.MeanPrecisions(predictions, solution, max_predictions=2)
        expected_mean_precisions = [0.5, 0.375]
        self.assertAllClose(mean_precisions, expected_mean_precisions)

    def testMeanMedianPositionWorks(self):
        if False:
            i = 10
            return i + 15
        predictions = _CreateRetrievalPredictions()
        solution = _CreateRetrievalSolution()
        (mean_position, median_position) = metrics.MeanMedianPosition(predictions, solution)
        expected_mean_position = 26.25
        expected_median_position = 1.5
        self.assertAllClose(mean_position, expected_mean_position)
        self.assertAllClose(median_position, expected_median_position)

    def testMeanMedianPositionMaxPredictionsWorks(self):
        if False:
            return 10
        predictions = _CreateRetrievalPredictions()
        solution = _CreateRetrievalSolution()
        (mean_position, median_position) = metrics.MeanMedianPosition(predictions, solution, max_predictions=1)
        expected_mean_position = 1.5
        expected_median_position = 1.5
        self.assertAllClose(mean_position, expected_mean_position)
        self.assertAllClose(median_position, expected_median_position)
if __name__ == '__main__':
    tf.test.main()