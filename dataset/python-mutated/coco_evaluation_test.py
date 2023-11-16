"""Tests for tensorflow_models.object_detection.metrics.coco_evaluation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from object_detection.core import standard_fields
from object_detection.metrics import coco_evaluation

def _get_categories_list():
    if False:
        while True:
            i = 10
    return [{'id': 1, 'name': 'person'}, {'id': 2, 'name': 'dog'}, {'id': 3, 'name': 'cat'}]

class CocoDetectionEvaluationTest(tf.test.TestCase):

    def testGetOneMAPWithMatchingGroundtruthAndDetections(self):
        if False:
            i = 10
            return i + 15
        'Tests that mAP is calculated correctly on GT and Detections.'
        coco_evaluator = coco_evaluation.CocoDetectionEvaluator(_get_categories_list())
        coco_evaluator.add_single_ground_truth_image_info(image_id='image1', groundtruth_dict={standard_fields.InputDataFields.groundtruth_boxes: np.array([[100.0, 100.0, 200.0, 200.0]]), standard_fields.InputDataFields.groundtruth_classes: np.array([1])})
        coco_evaluator.add_single_detected_image_info(image_id='image1', detections_dict={standard_fields.DetectionResultFields.detection_boxes: np.array([[100.0, 100.0, 200.0, 200.0]]), standard_fields.DetectionResultFields.detection_scores: np.array([0.8]), standard_fields.DetectionResultFields.detection_classes: np.array([1])})
        coco_evaluator.add_single_ground_truth_image_info(image_id='image2', groundtruth_dict={standard_fields.InputDataFields.groundtruth_boxes: np.array([[50.0, 50.0, 100.0, 100.0]]), standard_fields.InputDataFields.groundtruth_classes: np.array([1])})
        coco_evaluator.add_single_detected_image_info(image_id='image2', detections_dict={standard_fields.DetectionResultFields.detection_boxes: np.array([[50.0, 50.0, 100.0, 100.0]]), standard_fields.DetectionResultFields.detection_scores: np.array([0.8]), standard_fields.DetectionResultFields.detection_classes: np.array([1])})
        coco_evaluator.add_single_ground_truth_image_info(image_id='image3', groundtruth_dict={standard_fields.InputDataFields.groundtruth_boxes: np.array([[25.0, 25.0, 50.0, 50.0]]), standard_fields.InputDataFields.groundtruth_classes: np.array([1])})
        coco_evaluator.add_single_detected_image_info(image_id='image3', detections_dict={standard_fields.DetectionResultFields.detection_boxes: np.array([[25.0, 25.0, 50.0, 50.0]]), standard_fields.DetectionResultFields.detection_scores: np.array([0.8]), standard_fields.DetectionResultFields.detection_classes: np.array([1])})
        metrics = coco_evaluator.evaluate()
        self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP'], 1.0)

    def testGetOneMAPWithMatchingGroundtruthAndDetectionsSkipCrowd(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests computing mAP with is_crowd GT boxes skipped.'
        coco_evaluator = coco_evaluation.CocoDetectionEvaluator(_get_categories_list())
        coco_evaluator.add_single_ground_truth_image_info(image_id='image1', groundtruth_dict={standard_fields.InputDataFields.groundtruth_boxes: np.array([[100.0, 100.0, 200.0, 200.0], [99.0, 99.0, 200.0, 200.0]]), standard_fields.InputDataFields.groundtruth_classes: np.array([1, 2]), standard_fields.InputDataFields.groundtruth_is_crowd: np.array([0, 1])})
        coco_evaluator.add_single_detected_image_info(image_id='image1', detections_dict={standard_fields.DetectionResultFields.detection_boxes: np.array([[100.0, 100.0, 200.0, 200.0]]), standard_fields.DetectionResultFields.detection_scores: np.array([0.8]), standard_fields.DetectionResultFields.detection_classes: np.array([1])})
        metrics = coco_evaluator.evaluate()
        self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP'], 1.0)

    def testGetOneMAPWithMatchingGroundtruthAndDetectionsEmptyCrowd(self):
        if False:
            i = 10
            return i + 15
        'Tests computing mAP with empty is_crowd array passed in.'
        coco_evaluator = coco_evaluation.CocoDetectionEvaluator(_get_categories_list())
        coco_evaluator.add_single_ground_truth_image_info(image_id='image1', groundtruth_dict={standard_fields.InputDataFields.groundtruth_boxes: np.array([[100.0, 100.0, 200.0, 200.0]]), standard_fields.InputDataFields.groundtruth_classes: np.array([1]), standard_fields.InputDataFields.groundtruth_is_crowd: np.array([])})
        coco_evaluator.add_single_detected_image_info(image_id='image1', detections_dict={standard_fields.DetectionResultFields.detection_boxes: np.array([[100.0, 100.0, 200.0, 200.0]]), standard_fields.DetectionResultFields.detection_scores: np.array([0.8]), standard_fields.DetectionResultFields.detection_classes: np.array([1])})
        metrics = coco_evaluator.evaluate()
        self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP'], 1.0)

    def testRejectionOnDuplicateGroundtruth(self):
        if False:
            print('Hello World!')
        'Tests that groundtruth cannot be added more than once for an image.'
        coco_evaluator = coco_evaluation.CocoDetectionEvaluator(_get_categories_list())
        image_key1 = 'img1'
        groundtruth_boxes1 = np.array([[0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 3, 3]], dtype=float)
        groundtruth_class_labels1 = np.array([1, 3, 1], dtype=int)
        coco_evaluator.add_single_ground_truth_image_info(image_key1, {standard_fields.InputDataFields.groundtruth_boxes: groundtruth_boxes1, standard_fields.InputDataFields.groundtruth_classes: groundtruth_class_labels1})
        groundtruth_lists_len = len(coco_evaluator._groundtruth_list)
        coco_evaluator.add_single_ground_truth_image_info(image_key1, {standard_fields.InputDataFields.groundtruth_boxes: groundtruth_boxes1, standard_fields.InputDataFields.groundtruth_classes: groundtruth_class_labels1})
        self.assertEqual(groundtruth_lists_len, len(coco_evaluator._groundtruth_list))

    def testRejectionOnDuplicateDetections(self):
        if False:
            print('Hello World!')
        'Tests that detections cannot be added more than once for an image.'
        coco_evaluator = coco_evaluation.CocoDetectionEvaluator(_get_categories_list())
        coco_evaluator.add_single_ground_truth_image_info(image_id='image1', groundtruth_dict={standard_fields.InputDataFields.groundtruth_boxes: np.array([[99.0, 100.0, 200.0, 200.0]]), standard_fields.InputDataFields.groundtruth_classes: np.array([1])})
        coco_evaluator.add_single_detected_image_info(image_id='image1', detections_dict={standard_fields.DetectionResultFields.detection_boxes: np.array([[100.0, 100.0, 200.0, 200.0]]), standard_fields.DetectionResultFields.detection_scores: np.array([0.8]), standard_fields.DetectionResultFields.detection_classes: np.array([1])})
        detections_lists_len = len(coco_evaluator._detection_boxes_list)
        coco_evaluator.add_single_detected_image_info(image_id='image1', detections_dict={standard_fields.DetectionResultFields.detection_boxes: np.array([[100.0, 100.0, 200.0, 200.0]]), standard_fields.DetectionResultFields.detection_scores: np.array([0.8]), standard_fields.DetectionResultFields.detection_classes: np.array([1])})
        self.assertEqual(detections_lists_len, len(coco_evaluator._detection_boxes_list))

    def testExceptionRaisedWithMissingGroundtruth(self):
        if False:
            while True:
                i = 10
        'Tests that exception is raised for detection with missing groundtruth.'
        coco_evaluator = coco_evaluation.CocoDetectionEvaluator(_get_categories_list())
        with self.assertRaises(ValueError):
            coco_evaluator.add_single_detected_image_info(image_id='image1', detections_dict={standard_fields.DetectionResultFields.detection_boxes: np.array([[100.0, 100.0, 200.0, 200.0]]), standard_fields.DetectionResultFields.detection_scores: np.array([0.8]), standard_fields.DetectionResultFields.detection_classes: np.array([1])})

class CocoEvaluationPyFuncTest(tf.test.TestCase):

    def testGetOneMAPWithMatchingGroundtruthAndDetections(self):
        if False:
            print('Hello World!')
        coco_evaluator = coco_evaluation.CocoDetectionEvaluator(_get_categories_list())
        image_id = tf.placeholder(tf.string, shape=())
        groundtruth_boxes = tf.placeholder(tf.float32, shape=(None, 4))
        groundtruth_classes = tf.placeholder(tf.float32, shape=None)
        detection_boxes = tf.placeholder(tf.float32, shape=(None, 4))
        detection_scores = tf.placeholder(tf.float32, shape=None)
        detection_classes = tf.placeholder(tf.float32, shape=None)
        input_data_fields = standard_fields.InputDataFields
        detection_fields = standard_fields.DetectionResultFields
        eval_dict = {input_data_fields.key: image_id, input_data_fields.groundtruth_boxes: groundtruth_boxes, input_data_fields.groundtruth_classes: groundtruth_classes, detection_fields.detection_boxes: detection_boxes, detection_fields.detection_scores: detection_scores, detection_fields.detection_classes: detection_classes}
        eval_metric_ops = coco_evaluator.get_estimator_eval_metric_ops(eval_dict)
        (_, update_op) = eval_metric_ops['DetectionBoxes_Precision/mAP']
        with self.test_session() as sess:
            sess.run(update_op, feed_dict={image_id: 'image1', groundtruth_boxes: np.array([[100.0, 100.0, 200.0, 200.0]]), groundtruth_classes: np.array([1]), detection_boxes: np.array([[100.0, 100.0, 200.0, 200.0]]), detection_scores: np.array([0.8]), detection_classes: np.array([1])})
            sess.run(update_op, feed_dict={image_id: 'image2', groundtruth_boxes: np.array([[50.0, 50.0, 100.0, 100.0]]), groundtruth_classes: np.array([3]), detection_boxes: np.array([[50.0, 50.0, 100.0, 100.0]]), detection_scores: np.array([0.7]), detection_classes: np.array([3])})
            sess.run(update_op, feed_dict={image_id: 'image3', groundtruth_boxes: np.array([[25.0, 25.0, 50.0, 50.0]]), groundtruth_classes: np.array([2]), detection_boxes: np.array([[25.0, 25.0, 50.0, 50.0]]), detection_scores: np.array([0.9]), detection_classes: np.array([2])})
        metrics = {}
        for (key, (value_op, _)) in eval_metric_ops.iteritems():
            metrics[key] = value_op
        metrics = sess.run(metrics)
        self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP@.50IOU'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP@.75IOU'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP (large)'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP (medium)'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP (small)'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@1'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@10'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@100'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@100 (large)'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@100 (medium)'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@100 (small)'], 1.0)
        self.assertFalse(coco_evaluator._groundtruth_list)
        self.assertFalse(coco_evaluator._detection_boxes_list)
        self.assertFalse(coco_evaluator._image_ids)

    def testGetOneMAPWithMatchingGroundtruthAndDetectionsIsAnnotated(self):
        if False:
            for i in range(10):
                print('nop')
        coco_evaluator = coco_evaluation.CocoDetectionEvaluator(_get_categories_list())
        image_id = tf.placeholder(tf.string, shape=())
        groundtruth_boxes = tf.placeholder(tf.float32, shape=(None, 4))
        groundtruth_classes = tf.placeholder(tf.float32, shape=None)
        is_annotated = tf.placeholder(tf.bool, shape=())
        detection_boxes = tf.placeholder(tf.float32, shape=(None, 4))
        detection_scores = tf.placeholder(tf.float32, shape=None)
        detection_classes = tf.placeholder(tf.float32, shape=None)
        input_data_fields = standard_fields.InputDataFields
        detection_fields = standard_fields.DetectionResultFields
        eval_dict = {input_data_fields.key: image_id, input_data_fields.groundtruth_boxes: groundtruth_boxes, input_data_fields.groundtruth_classes: groundtruth_classes, 'is_annotated': is_annotated, detection_fields.detection_boxes: detection_boxes, detection_fields.detection_scores: detection_scores, detection_fields.detection_classes: detection_classes}
        eval_metric_ops = coco_evaluator.get_estimator_eval_metric_ops(eval_dict)
        (_, update_op) = eval_metric_ops['DetectionBoxes_Precision/mAP']
        with self.test_session() as sess:
            sess.run(update_op, feed_dict={image_id: 'image1', groundtruth_boxes: np.array([[100.0, 100.0, 200.0, 200.0]]), groundtruth_classes: np.array([1]), is_annotated: True, detection_boxes: np.array([[100.0, 100.0, 200.0, 200.0]]), detection_scores: np.array([0.8]), detection_classes: np.array([1])})
            sess.run(update_op, feed_dict={image_id: 'image2', groundtruth_boxes: np.array([[50.0, 50.0, 100.0, 100.0]]), groundtruth_classes: np.array([3]), is_annotated: True, detection_boxes: np.array([[50.0, 50.0, 100.0, 100.0]]), detection_scores: np.array([0.7]), detection_classes: np.array([3])})
            sess.run(update_op, feed_dict={image_id: 'image3', groundtruth_boxes: np.array([[25.0, 25.0, 50.0, 50.0]]), groundtruth_classes: np.array([2]), is_annotated: True, detection_boxes: np.array([[25.0, 25.0, 50.0, 50.0]]), detection_scores: np.array([0.9]), detection_classes: np.array([2])})
            sess.run(update_op, feed_dict={image_id: 'image4', groundtruth_boxes: np.zeros((0, 4)), groundtruth_classes: np.zeros(0), is_annotated: False, detection_boxes: np.array([[25.0, 25.0, 50.0, 50.0], [25.0, 25.0, 70.0, 50.0], [25.0, 25.0, 80.0, 50.0], [25.0, 25.0, 90.0, 50.0]]), detection_scores: np.array([0.6, 0.7, 0.8, 0.9]), detection_classes: np.array([1, 2, 2, 3])})
        metrics = {}
        for (key, (value_op, _)) in eval_metric_ops.iteritems():
            metrics[key] = value_op
        metrics = sess.run(metrics)
        self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP@.50IOU'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP@.75IOU'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP (large)'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP (medium)'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP (small)'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@1'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@10'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@100'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@100 (large)'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@100 (medium)'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@100 (small)'], 1.0)
        self.assertFalse(coco_evaluator._groundtruth_list)
        self.assertFalse(coco_evaluator._detection_boxes_list)
        self.assertFalse(coco_evaluator._image_ids)

    def testGetOneMAPWithMatchingGroundtruthAndDetectionsPadded(self):
        if False:
            i = 10
            return i + 15
        coco_evaluator = coco_evaluation.CocoDetectionEvaluator(_get_categories_list())
        image_id = tf.placeholder(tf.string, shape=())
        groundtruth_boxes = tf.placeholder(tf.float32, shape=(None, 4))
        groundtruth_classes = tf.placeholder(tf.float32, shape=None)
        detection_boxes = tf.placeholder(tf.float32, shape=(None, 4))
        detection_scores = tf.placeholder(tf.float32, shape=None)
        detection_classes = tf.placeholder(tf.float32, shape=None)
        input_data_fields = standard_fields.InputDataFields
        detection_fields = standard_fields.DetectionResultFields
        eval_dict = {input_data_fields.key: image_id, input_data_fields.groundtruth_boxes: groundtruth_boxes, input_data_fields.groundtruth_classes: groundtruth_classes, detection_fields.detection_boxes: detection_boxes, detection_fields.detection_scores: detection_scores, detection_fields.detection_classes: detection_classes}
        eval_metric_ops = coco_evaluator.get_estimator_eval_metric_ops(eval_dict)
        (_, update_op) = eval_metric_ops['DetectionBoxes_Precision/mAP']
        with self.test_session() as sess:
            sess.run(update_op, feed_dict={image_id: 'image1', groundtruth_boxes: np.array([[100.0, 100.0, 200.0, 200.0], [-1, -1, -1, -1]]), groundtruth_classes: np.array([1, -1]), detection_boxes: np.array([[100.0, 100.0, 200.0, 200.0], [0.0, 0.0, 0.0, 0.0]]), detection_scores: np.array([0.8, 0.0]), detection_classes: np.array([1, -1])})
            sess.run(update_op, feed_dict={image_id: 'image2', groundtruth_boxes: np.array([[50.0, 50.0, 100.0, 100.0], [-1, -1, -1, -1]]), groundtruth_classes: np.array([3, -1]), detection_boxes: np.array([[50.0, 50.0, 100.0, 100.0], [0.0, 0.0, 0.0, 0.0]]), detection_scores: np.array([0.7, 0.0]), detection_classes: np.array([3, -1])})
            sess.run(update_op, feed_dict={image_id: 'image3', groundtruth_boxes: np.array([[25.0, 25.0, 50.0, 50.0], [10.0, 10.0, 15.0, 15.0]]), groundtruth_classes: np.array([2, 2]), detection_boxes: np.array([[25.0, 25.0, 50.0, 50.0], [10.0, 10.0, 15.0, 15.0]]), detection_scores: np.array([0.95, 0.9]), detection_classes: np.array([2, 2])})
        metrics = {}
        for (key, (value_op, _)) in eval_metric_ops.iteritems():
            metrics[key] = value_op
        metrics = sess.run(metrics)
        self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP@.50IOU'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP@.75IOU'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP (large)'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP (medium)'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP (small)'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@1'], 0.83333331)
        self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@10'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@100'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@100 (large)'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@100 (medium)'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@100 (small)'], 1.0)
        self.assertFalse(coco_evaluator._groundtruth_list)
        self.assertFalse(coco_evaluator._detection_boxes_list)
        self.assertFalse(coco_evaluator._image_ids)

    def testGetOneMAPWithMatchingGroundtruthAndDetectionsBatched(self):
        if False:
            while True:
                i = 10
        coco_evaluator = coco_evaluation.CocoDetectionEvaluator(_get_categories_list())
        batch_size = 3
        image_id = tf.placeholder(tf.string, shape=batch_size)
        groundtruth_boxes = tf.placeholder(tf.float32, shape=(batch_size, None, 4))
        groundtruth_classes = tf.placeholder(tf.float32, shape=(batch_size, None))
        detection_boxes = tf.placeholder(tf.float32, shape=(batch_size, None, 4))
        detection_scores = tf.placeholder(tf.float32, shape=(batch_size, None))
        detection_classes = tf.placeholder(tf.float32, shape=(batch_size, None))
        input_data_fields = standard_fields.InputDataFields
        detection_fields = standard_fields.DetectionResultFields
        eval_dict = {input_data_fields.key: image_id, input_data_fields.groundtruth_boxes: groundtruth_boxes, input_data_fields.groundtruth_classes: groundtruth_classes, detection_fields.detection_boxes: detection_boxes, detection_fields.detection_scores: detection_scores, detection_fields.detection_classes: detection_classes}
        eval_metric_ops = coco_evaluator.get_estimator_eval_metric_ops(eval_dict)
        (_, update_op) = eval_metric_ops['DetectionBoxes_Precision/mAP']
        with self.test_session() as sess:
            sess.run(update_op, feed_dict={image_id: ['image1', 'image2', 'image3'], groundtruth_boxes: np.array([[[100.0, 100.0, 200.0, 200.0]], [[50.0, 50.0, 100.0, 100.0]], [[25.0, 25.0, 50.0, 50.0]]]), groundtruth_classes: np.array([[1], [3], [2]]), detection_boxes: np.array([[[100.0, 100.0, 200.0, 200.0]], [[50.0, 50.0, 100.0, 100.0]], [[25.0, 25.0, 50.0, 50.0]]]), detection_scores: np.array([[0.8], [0.7], [0.9]]), detection_classes: np.array([[1], [3], [2]])})
        metrics = {}
        for (key, (value_op, _)) in eval_metric_ops.iteritems():
            metrics[key] = value_op
        metrics = sess.run(metrics)
        self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP@.50IOU'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP@.75IOU'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP (large)'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP (medium)'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP (small)'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@1'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@10'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@100'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@100 (large)'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@100 (medium)'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@100 (small)'], 1.0)
        self.assertFalse(coco_evaluator._groundtruth_list)
        self.assertFalse(coco_evaluator._detection_boxes_list)
        self.assertFalse(coco_evaluator._image_ids)

    def testGetOneMAPWithMatchingGroundtruthAndDetectionsPaddedBatches(self):
        if False:
            for i in range(10):
                print('nop')
        coco_evaluator = coco_evaluation.CocoDetectionEvaluator(_get_categories_list())
        batch_size = 3
        image_id = tf.placeholder(tf.string, shape=batch_size)
        groundtruth_boxes = tf.placeholder(tf.float32, shape=(batch_size, None, 4))
        groundtruth_classes = tf.placeholder(tf.float32, shape=(batch_size, None))
        num_gt_boxes_per_image = tf.placeholder(tf.int32, shape=None)
        detection_boxes = tf.placeholder(tf.float32, shape=(batch_size, None, 4))
        detection_scores = tf.placeholder(tf.float32, shape=(batch_size, None))
        detection_classes = tf.placeholder(tf.float32, shape=(batch_size, None))
        num_det_boxes_per_image = tf.placeholder(tf.int32, shape=None)
        input_data_fields = standard_fields.InputDataFields
        detection_fields = standard_fields.DetectionResultFields
        eval_dict = {input_data_fields.key: image_id, input_data_fields.groundtruth_boxes: groundtruth_boxes, input_data_fields.groundtruth_classes: groundtruth_classes, detection_fields.detection_boxes: detection_boxes, detection_fields.detection_scores: detection_scores, detection_fields.detection_classes: detection_classes, 'num_groundtruth_boxes_per_image': num_gt_boxes_per_image, 'num_det_boxes_per_image': num_det_boxes_per_image}
        eval_metric_ops = coco_evaluator.get_estimator_eval_metric_ops(eval_dict)
        (_, update_op) = eval_metric_ops['DetectionBoxes_Precision/mAP']
        with self.test_session() as sess:
            sess.run(update_op, feed_dict={image_id: ['image1', 'image2', 'image3'], groundtruth_boxes: np.array([[[100.0, 100.0, 200.0, 200.0], [-1, -1, -1, -1]], [[50.0, 50.0, 100.0, 100.0], [-1, -1, -1, -1]], [[25.0, 25.0, 50.0, 50.0], [10.0, 10.0, 15.0, 15.0]]]), groundtruth_classes: np.array([[1, -1], [3, -1], [2, 2]]), num_gt_boxes_per_image: np.array([1, 1, 2]), detection_boxes: np.array([[[100.0, 100.0, 200.0, 200.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[50.0, 50.0, 100.0, 100.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[25.0, 25.0, 50.0, 50.0], [10.0, 10.0, 15.0, 15.0], [10.0, 10.0, 15.0, 15.0]]]), detection_scores: np.array([[0.8, 0.0, 0.0], [0.7, 0.0, 0.0], [0.95, 0.9, 0.9]]), detection_classes: np.array([[1, -1, -1], [3, -1, -1], [2, 2, 2]]), num_det_boxes_per_image: np.array([1, 1, 3])})
        self.assertEqual(len(coco_evaluator._groundtruth_list), 4)
        self.assertEqual(len(coco_evaluator._detection_boxes_list), 5)
        metrics = {}
        for (key, (value_op, _)) in eval_metric_ops.iteritems():
            metrics[key] = value_op
        metrics = sess.run(metrics)
        self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP@.50IOU'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP@.75IOU'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP (large)'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP (medium)'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP (small)'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@1'], 0.83333331)
        self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@10'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@100'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@100 (large)'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@100 (medium)'], 1.0)
        self.assertAlmostEqual(metrics['DetectionBoxes_Recall/AR@100 (small)'], 1.0)
        self.assertFalse(coco_evaluator._groundtruth_list)
        self.assertFalse(coco_evaluator._detection_boxes_list)
        self.assertFalse(coco_evaluator._image_ids)

class CocoMaskEvaluationTest(tf.test.TestCase):

    def testGetOneMAPWithMatchingGroundtruthAndDetections(self):
        if False:
            i = 10
            return i + 15
        coco_evaluator = coco_evaluation.CocoMaskEvaluator(_get_categories_list())
        coco_evaluator.add_single_ground_truth_image_info(image_id='image1', groundtruth_dict={standard_fields.InputDataFields.groundtruth_boxes: np.array([[100.0, 100.0, 200.0, 200.0]]), standard_fields.InputDataFields.groundtruth_classes: np.array([1]), standard_fields.InputDataFields.groundtruth_instance_masks: np.pad(np.ones([1, 100, 100], dtype=np.uint8), ((0, 0), (10, 10), (10, 10)), mode='constant')})
        coco_evaluator.add_single_detected_image_info(image_id='image1', detections_dict={standard_fields.DetectionResultFields.detection_boxes: np.array([[100.0, 100.0, 200.0, 200.0]]), standard_fields.DetectionResultFields.detection_scores: np.array([0.8]), standard_fields.DetectionResultFields.detection_classes: np.array([1]), standard_fields.DetectionResultFields.detection_masks: np.pad(np.ones([1, 100, 100], dtype=np.uint8), ((0, 0), (10, 10), (10, 10)), mode='constant')})
        coco_evaluator.add_single_ground_truth_image_info(image_id='image2', groundtruth_dict={standard_fields.InputDataFields.groundtruth_boxes: np.array([[50.0, 50.0, 100.0, 100.0]]), standard_fields.InputDataFields.groundtruth_classes: np.array([1]), standard_fields.InputDataFields.groundtruth_instance_masks: np.pad(np.ones([1, 50, 50], dtype=np.uint8), ((0, 0), (10, 10), (10, 10)), mode='constant')})
        coco_evaluator.add_single_detected_image_info(image_id='image2', detections_dict={standard_fields.DetectionResultFields.detection_boxes: np.array([[50.0, 50.0, 100.0, 100.0]]), standard_fields.DetectionResultFields.detection_scores: np.array([0.8]), standard_fields.DetectionResultFields.detection_classes: np.array([1]), standard_fields.DetectionResultFields.detection_masks: np.pad(np.ones([1, 50, 50], dtype=np.uint8), ((0, 0), (10, 10), (10, 10)), mode='constant')})
        coco_evaluator.add_single_ground_truth_image_info(image_id='image3', groundtruth_dict={standard_fields.InputDataFields.groundtruth_boxes: np.array([[25.0, 25.0, 50.0, 50.0]]), standard_fields.InputDataFields.groundtruth_classes: np.array([1]), standard_fields.InputDataFields.groundtruth_instance_masks: np.pad(np.ones([1, 25, 25], dtype=np.uint8), ((0, 0), (10, 10), (10, 10)), mode='constant')})
        coco_evaluator.add_single_detected_image_info(image_id='image3', detections_dict={standard_fields.DetectionResultFields.detection_boxes: np.array([[25.0, 25.0, 50.0, 50.0]]), standard_fields.DetectionResultFields.detection_scores: np.array([0.8]), standard_fields.DetectionResultFields.detection_classes: np.array([1]), standard_fields.DetectionResultFields.detection_masks: np.pad(np.ones([1, 25, 25], dtype=np.uint8), ((0, 0), (10, 10), (10, 10)), mode='constant')})
        metrics = coco_evaluator.evaluate()
        self.assertAlmostEqual(metrics['DetectionMasks_Precision/mAP'], 1.0)
        coco_evaluator.clear()
        self.assertFalse(coco_evaluator._image_id_to_mask_shape_map)
        self.assertFalse(coco_evaluator._image_ids_with_detections)
        self.assertFalse(coco_evaluator._groundtruth_list)
        self.assertFalse(coco_evaluator._detection_masks_list)

class CocoMaskEvaluationPyFuncTest(tf.test.TestCase):

    def testGetOneMAPWithMatchingGroundtruthAndDetections(self):
        if False:
            i = 10
            return i + 15
        coco_evaluator = coco_evaluation.CocoMaskEvaluator(_get_categories_list())
        image_id = tf.placeholder(tf.string, shape=())
        groundtruth_boxes = tf.placeholder(tf.float32, shape=(None, 4))
        groundtruth_classes = tf.placeholder(tf.float32, shape=None)
        groundtruth_masks = tf.placeholder(tf.uint8, shape=(None, None, None))
        detection_scores = tf.placeholder(tf.float32, shape=None)
        detection_classes = tf.placeholder(tf.float32, shape=None)
        detection_masks = tf.placeholder(tf.uint8, shape=(None, None, None))
        input_data_fields = standard_fields.InputDataFields
        detection_fields = standard_fields.DetectionResultFields
        eval_dict = {input_data_fields.key: image_id, input_data_fields.groundtruth_boxes: groundtruth_boxes, input_data_fields.groundtruth_classes: groundtruth_classes, input_data_fields.groundtruth_instance_masks: groundtruth_masks, detection_fields.detection_scores: detection_scores, detection_fields.detection_classes: detection_classes, detection_fields.detection_masks: detection_masks}
        eval_metric_ops = coco_evaluator.get_estimator_eval_metric_ops(eval_dict)
        (_, update_op) = eval_metric_ops['DetectionMasks_Precision/mAP']
        with self.test_session() as sess:
            sess.run(update_op, feed_dict={image_id: 'image1', groundtruth_boxes: np.array([[100.0, 100.0, 200.0, 200.0], [50.0, 50.0, 100.0, 100.0]]), groundtruth_classes: np.array([1, 2]), groundtruth_masks: np.stack([np.pad(np.ones([100, 100], dtype=np.uint8), ((10, 10), (10, 10)), mode='constant'), np.pad(np.ones([50, 50], dtype=np.uint8), ((0, 70), (0, 70)), mode='constant')]), detection_scores: np.array([0.9, 0.8]), detection_classes: np.array([2, 1]), detection_masks: np.stack([np.pad(np.ones([50, 50], dtype=np.uint8), ((0, 70), (0, 70)), mode='constant'), np.pad(np.ones([100, 100], dtype=np.uint8), ((10, 10), (10, 10)), mode='constant')])})
            sess.run(update_op, feed_dict={image_id: 'image2', groundtruth_boxes: np.array([[50.0, 50.0, 100.0, 100.0]]), groundtruth_classes: np.array([1]), groundtruth_masks: np.pad(np.ones([1, 50, 50], dtype=np.uint8), ((0, 0), (10, 10), (10, 10)), mode='constant'), detection_scores: np.array([0.8]), detection_classes: np.array([1]), detection_masks: np.pad(np.ones([1, 50, 50], dtype=np.uint8), ((0, 0), (10, 10), (10, 10)), mode='constant')})
            sess.run(update_op, feed_dict={image_id: 'image3', groundtruth_boxes: np.array([[25.0, 25.0, 50.0, 50.0]]), groundtruth_classes: np.array([1]), groundtruth_masks: np.pad(np.ones([1, 25, 25], dtype=np.uint8), ((0, 0), (10, 10), (10, 10)), mode='constant'), detection_scores: np.array([0.8]), detection_classes: np.array([1]), detection_masks: np.pad(np.ones([1, 25, 25], dtype=np.uint8), ((0, 0), (10, 10), (10, 10)), mode='constant')})
        metrics = {}
        for (key, (value_op, _)) in eval_metric_ops.iteritems():
            metrics[key] = value_op
        metrics = sess.run(metrics)
        self.assertAlmostEqual(metrics['DetectionMasks_Precision/mAP'], 1.0)
        self.assertAlmostEqual(metrics['DetectionMasks_Precision/mAP@.50IOU'], 1.0)
        self.assertAlmostEqual(metrics['DetectionMasks_Precision/mAP@.75IOU'], 1.0)
        self.assertAlmostEqual(metrics['DetectionMasks_Precision/mAP (large)'], 1.0)
        self.assertAlmostEqual(metrics['DetectionMasks_Precision/mAP (medium)'], 1.0)
        self.assertAlmostEqual(metrics['DetectionMasks_Precision/mAP (small)'], 1.0)
        self.assertAlmostEqual(metrics['DetectionMasks_Recall/AR@1'], 1.0)
        self.assertAlmostEqual(metrics['DetectionMasks_Recall/AR@10'], 1.0)
        self.assertAlmostEqual(metrics['DetectionMasks_Recall/AR@100'], 1.0)
        self.assertAlmostEqual(metrics['DetectionMasks_Recall/AR@100 (large)'], 1.0)
        self.assertAlmostEqual(metrics['DetectionMasks_Recall/AR@100 (medium)'], 1.0)
        self.assertAlmostEqual(metrics['DetectionMasks_Recall/AR@100 (small)'], 1.0)
        self.assertFalse(coco_evaluator._groundtruth_list)
        self.assertFalse(coco_evaluator._image_ids_with_detections)
        self.assertFalse(coco_evaluator._image_id_to_mask_shape_map)
        self.assertFalse(coco_evaluator._detection_masks_list)

    def testGetOneMAPWithMatchingGroundtruthAndDetectionsBatched(self):
        if False:
            print('Hello World!')
        coco_evaluator = coco_evaluation.CocoMaskEvaluator(_get_categories_list())
        batch_size = 3
        image_id = tf.placeholder(tf.string, shape=batch_size)
        groundtruth_boxes = tf.placeholder(tf.float32, shape=(batch_size, None, 4))
        groundtruth_classes = tf.placeholder(tf.float32, shape=(batch_size, None))
        groundtruth_masks = tf.placeholder(tf.uint8, shape=(batch_size, None, None, None))
        detection_scores = tf.placeholder(tf.float32, shape=(batch_size, None))
        detection_classes = tf.placeholder(tf.float32, shape=(batch_size, None))
        detection_masks = tf.placeholder(tf.uint8, shape=(batch_size, None, None, None))
        input_data_fields = standard_fields.InputDataFields
        detection_fields = standard_fields.DetectionResultFields
        eval_dict = {input_data_fields.key: image_id, input_data_fields.groundtruth_boxes: groundtruth_boxes, input_data_fields.groundtruth_classes: groundtruth_classes, input_data_fields.groundtruth_instance_masks: groundtruth_masks, detection_fields.detection_scores: detection_scores, detection_fields.detection_classes: detection_classes, detection_fields.detection_masks: detection_masks}
        eval_metric_ops = coco_evaluator.get_estimator_eval_metric_ops(eval_dict)
        (_, update_op) = eval_metric_ops['DetectionMasks_Precision/mAP']
        with self.test_session() as sess:
            sess.run(update_op, feed_dict={image_id: ['image1', 'image2', 'image3'], groundtruth_boxes: np.array([[[100.0, 100.0, 200.0, 200.0]], [[50.0, 50.0, 100.0, 100.0]], [[25.0, 25.0, 50.0, 50.0]]]), groundtruth_classes: np.array([[1], [1], [1]]), groundtruth_masks: np.stack([np.pad(np.ones([1, 100, 100], dtype=np.uint8), ((0, 0), (0, 0), (0, 0)), mode='constant'), np.pad(np.ones([1, 50, 50], dtype=np.uint8), ((0, 0), (25, 25), (25, 25)), mode='constant'), np.pad(np.ones([1, 25, 25], dtype=np.uint8), ((0, 0), (37, 38), (37, 38)), mode='constant')], axis=0), detection_scores: np.array([[0.8], [0.8], [0.8]]), detection_classes: np.array([[1], [1], [1]]), detection_masks: np.stack([np.pad(np.ones([1, 100, 100], dtype=np.uint8), ((0, 0), (0, 0), (0, 0)), mode='constant'), np.pad(np.ones([1, 50, 50], dtype=np.uint8), ((0, 0), (25, 25), (25, 25)), mode='constant'), np.pad(np.ones([1, 25, 25], dtype=np.uint8), ((0, 0), (37, 38), (37, 38)), mode='constant')], axis=0)})
        metrics = {}
        for (key, (value_op, _)) in eval_metric_ops.iteritems():
            metrics[key] = value_op
        metrics = sess.run(metrics)
        self.assertAlmostEqual(metrics['DetectionMasks_Precision/mAP'], 1.0)
        self.assertAlmostEqual(metrics['DetectionMasks_Precision/mAP@.50IOU'], 1.0)
        self.assertAlmostEqual(metrics['DetectionMasks_Precision/mAP@.75IOU'], 1.0)
        self.assertAlmostEqual(metrics['DetectionMasks_Precision/mAP (large)'], 1.0)
        self.assertAlmostEqual(metrics['DetectionMasks_Precision/mAP (medium)'], 1.0)
        self.assertAlmostEqual(metrics['DetectionMasks_Precision/mAP (small)'], 1.0)
        self.assertAlmostEqual(metrics['DetectionMasks_Recall/AR@1'], 1.0)
        self.assertAlmostEqual(metrics['DetectionMasks_Recall/AR@10'], 1.0)
        self.assertAlmostEqual(metrics['DetectionMasks_Recall/AR@100'], 1.0)
        self.assertAlmostEqual(metrics['DetectionMasks_Recall/AR@100 (large)'], 1.0)
        self.assertAlmostEqual(metrics['DetectionMasks_Recall/AR@100 (medium)'], 1.0)
        self.assertAlmostEqual(metrics['DetectionMasks_Recall/AR@100 (small)'], 1.0)
        self.assertFalse(coco_evaluator._groundtruth_list)
        self.assertFalse(coco_evaluator._image_ids_with_detections)
        self.assertFalse(coco_evaluator._image_id_to_mask_shape_map)
        self.assertFalse(coco_evaluator._detection_masks_list)
if __name__ == '__main__':
    tf.test.main()