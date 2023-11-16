"""Tests for video_object_detection.metrics.coco_video_evaluation."""
import numpy as np
import tensorflow as tf
from lstm_object_detection.metrics import coco_evaluation_all_frames
from object_detection.core import standard_fields

class CocoEvaluationAllFramesTest(tf.test.TestCase):

    def testGroundtruthAndDetectionsDisagreeOnAllFrames(self):
        if False:
            print('Hello World!')
        'Tests that mAP is calculated on several different frame results.'
        category_list = [{'id': 0, 'name': 'dog'}, {'id': 1, 'name': 'cat'}]
        video_evaluator = coco_evaluation_all_frames.CocoEvaluationAllFrames(category_list)
        video_evaluator.add_single_ground_truth_image_info(image_id='image1', groundtruth_dict=[{standard_fields.InputDataFields.groundtruth_boxes: np.array([[50.0, 50.0, 200.0, 200.0]]), standard_fields.InputDataFields.groundtruth_classes: np.array([1])}, {standard_fields.InputDataFields.groundtruth_boxes: np.array([[50.0, 50.0, 100.0, 100.0]]), standard_fields.InputDataFields.groundtruth_classes: np.array([1])}])
        video_evaluator.add_single_detected_image_info(image_id='image1', detections_dict=[{standard_fields.DetectionResultFields.detection_boxes: np.array([[100.0, 100.0, 200.0, 200.0]]), standard_fields.DetectionResultFields.detection_scores: np.array([0.8]), standard_fields.DetectionResultFields.detection_classes: np.array([1])}, {standard_fields.DetectionResultFields.detection_boxes: np.array([[50.0, 50.0, 100.0, 100.0]]), standard_fields.DetectionResultFields.detection_scores: np.array([0.8]), standard_fields.DetectionResultFields.detection_classes: np.array([1])}])
        metrics = video_evaluator.evaluate()
        self.assertNotEqual(metrics['DetectionBoxes_Precision/mAP'], 1.0)

    def testGroundtruthAndDetections(self):
        if False:
            while True:
                i = 10
        'Tests that mAP is calculated correctly on GT and Detections.'
        category_list = [{'id': 0, 'name': 'dog'}, {'id': 1, 'name': 'cat'}]
        video_evaluator = coco_evaluation_all_frames.CocoEvaluationAllFrames(category_list)
        video_evaluator.add_single_ground_truth_image_info(image_id='image1', groundtruth_dict=[{standard_fields.InputDataFields.groundtruth_boxes: np.array([[100.0, 100.0, 200.0, 200.0]]), standard_fields.InputDataFields.groundtruth_classes: np.array([1])}])
        video_evaluator.add_single_ground_truth_image_info(image_id='image2', groundtruth_dict=[{standard_fields.InputDataFields.groundtruth_boxes: np.array([[50.0, 50.0, 100.0, 100.0]]), standard_fields.InputDataFields.groundtruth_classes: np.array([1])}])
        video_evaluator.add_single_ground_truth_image_info(image_id='image3', groundtruth_dict=[{standard_fields.InputDataFields.groundtruth_boxes: np.array([[50.0, 100.0, 100.0, 120.0]]), standard_fields.InputDataFields.groundtruth_classes: np.array([1])}])
        video_evaluator.add_single_detected_image_info(image_id='image1', detections_dict=[{standard_fields.DetectionResultFields.detection_boxes: np.array([[100.0, 100.0, 200.0, 200.0]]), standard_fields.DetectionResultFields.detection_scores: np.array([0.8]), standard_fields.DetectionResultFields.detection_classes: np.array([1])}])
        video_evaluator.add_single_detected_image_info(image_id='image2', detections_dict=[{standard_fields.DetectionResultFields.detection_boxes: np.array([[50.0, 50.0, 100.0, 100.0]]), standard_fields.DetectionResultFields.detection_scores: np.array([0.8]), standard_fields.DetectionResultFields.detection_classes: np.array([1])}])
        video_evaluator.add_single_detected_image_info(image_id='image3', detections_dict=[{standard_fields.DetectionResultFields.detection_boxes: np.array([[50.0, 100.0, 100.0, 120.0]]), standard_fields.DetectionResultFields.detection_scores: np.array([0.8]), standard_fields.DetectionResultFields.detection_classes: np.array([1])}])
        metrics = video_evaluator.evaluate()
        self.assertAlmostEqual(metrics['DetectionBoxes_Precision/mAP'], 1.0)

    def testMissingDetectionResults(self):
        if False:
            i = 10
            return i + 15
        'Tests if groundtrue is missing, raises ValueError.'
        category_list = [{'id': 0, 'name': 'dog'}]
        video_evaluator = coco_evaluation_all_frames.CocoEvaluationAllFrames(category_list)
        video_evaluator.add_single_ground_truth_image_info(image_id='image1', groundtruth_dict=[{standard_fields.InputDataFields.groundtruth_boxes: np.array([[100.0, 100.0, 200.0, 200.0]]), standard_fields.InputDataFields.groundtruth_classes: np.array([1])}])
        with self.assertRaisesRegexp(ValueError, 'Missing groundtruth for image-frame id:.*'):
            video_evaluator.add_single_detected_image_info(image_id='image3', detections_dict=[{standard_fields.DetectionResultFields.detection_boxes: np.array([[100.0, 100.0, 200.0, 200.0]]), standard_fields.DetectionResultFields.detection_scores: np.array([0.8]), standard_fields.DetectionResultFields.detection_classes: np.array([1])}])
if __name__ == '__main__':
    tf.test.main()