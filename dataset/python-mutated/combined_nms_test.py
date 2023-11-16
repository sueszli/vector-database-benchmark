"""Script to test TF-TensorRT conversion of CombinedNMS op."""
import os
from tensorflow.python.compiler.tensorrt import utils as trt_utils
from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import image_ops_impl
from tensorflow.python.platform import test

class CombinedNmsTest(trt_test.TfTrtIntegrationTestBase):
    """Test for CombinedNMS op in TF-TRT."""

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.num_boxes = 200

    def GraphFn(self, boxes, scores):
        if False:
            for i in range(10):
                print('nop')
        max_total_size = 3
        score_threshold = 0.1
        iou_threshold = 0.5
        max_total_size_tensor = constant_op.constant(max_total_size, dtype=dtypes.int32, name='max_total_size')
        iou_threshold_tensor = constant_op.constant(iou_threshold, dtype=dtypes.float32, name='iou_threshold')
        score_threshold_tensor = constant_op.constant(score_threshold, dtype=dtypes.float32, name='score_threshold')
        nms_output = image_ops_impl.combined_non_max_suppression(boxes, scores, max_total_size_tensor, max_total_size_tensor, iou_threshold_tensor, score_threshold_tensor, name='combined_nms')
        return [array_ops.identity(output, name='output_%d' % i) for (i, output) in enumerate(nms_output)]

    def GetParams(self):
        if False:
            while True:
                i = 10
        q = 1
        batch_size = 2
        num_classes = 2
        max_total_size = 3
        boxes_shape = [batch_size, self.num_boxes, q, 4]
        scores_shape = [batch_size, self.num_boxes, num_classes]
        nmsed_boxes_shape = [batch_size, max_total_size, 4]
        nmsed_scores_shape = [batch_size, max_total_size]
        nmsed_classes_shape = [batch_size, max_total_size]
        valid_detections_shape = [batch_size]
        return self.BuildParams(self.GraphFn, dtypes.float32, [boxes_shape, scores_shape], [nmsed_boxes_shape, nmsed_scores_shape, nmsed_classes_shape, valid_detections_shape])

    def ExpectedEnginesToBuild(self, run_params):
        if False:
            print('Hello World!')
        'Return the expected engines to build.'
        if not run_params.dynamic_shape:
            return {'TRTEngineOp_000': ['combined_nms/CombinedNonMaxSuppression', 'max_total_size', 'iou_threshold', 'score_threshold']}
        else:
            return dict()

    def ShouldRunTest(self, run_params):
        if False:
            i = 10
            return i + 15
        (should_run, reason) = super().ShouldRunTest(run_params)
        should_run = should_run and (not trt_test.IsQuantizationMode(run_params.precision_mode))
        reason += ' and precision != INT8'
        return (should_run and trt_utils.is_linked_tensorrt_version_greater_equal(7, 1, 3), reason + ' and >= TRT 7.1.3')

class CombinedNmsExecuteNativeSegmentTest(CombinedNmsTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        os.environ['TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT_EXECUTION'] = 'True'

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        super().tearDown()
        os.environ['TF_TRT_ALLOW_ENGINE_NATIVE_SEGMENT_EXECUTION'] = 'False'

    def GetMaxBatchSize(self, run_params):
        if False:
            i = 10
            return i + 15
        'Returns the max_batch_size that the converter should use for tests.'
        if run_params.dynamic_engine:
            return None
        return super().GetMaxBatchSize(run_params) - 1

    def ShouldRunTest(self, run_params):
        if False:
            for i in range(10):
                print('nop')
        (should_run, reason) = super().ShouldRunTest(run_params)
        return (should_run and (not run_params.dynamic_engine), reason + ' and static engines')

class CombinedNmsTestTopK(CombinedNmsTest):
    """Test for CombinedNMS TopK op in TF-TRT."""

    def GraphFn(self, pre_nms_boxes, pre_nms_scores, max_boxes_to_draw, max_detetion_points):
        if False:
            while True:
                i = 10
        iou_threshold = 0.1
        score_threshold = 0.001
        max_output_size_per_class_tensor = constant_op.constant(max_detetion_points, dtype=dtypes.int32, name='max_output_size_per_class')
        max_total_size_tensor = constant_op.constant(max_boxes_to_draw, dtype=dtypes.int32, name='max_total_size')
        iou_threshold_tensor = constant_op.constant(iou_threshold, dtype=dtypes.float32, name='iou_threshold')
        score_threshold_tensor = constant_op.constant(score_threshold, dtype=dtypes.float32, name='score_threshold')
        nms_output = image_ops_impl.combined_non_max_suppression(pre_nms_boxes, pre_nms_scores, max_output_size_per_class=max_output_size_per_class_tensor, max_total_size=max_total_size_tensor, iou_threshold=iou_threshold_tensor, score_threshold=score_threshold_tensor, pad_per_class=False, name='combined_nms')
        return [array_ops.identity(output, name='output_%d' % i) for (i, output) in enumerate(nms_output)]

    def GetParams(self):
        if False:
            print('Hello World!')
        batch_size = 1
        max_detetion_points = 2048
        num_classes = 90
        max_boxes_to_draw = 30
        pre_nms_boxes_shape = [batch_size, max_detetion_points, 1, 4]
        pre_nms_scores_shape = [batch_size, max_detetion_points, num_classes]
        nmsed_boxes_shape = [batch_size, max_boxes_to_draw, 4]
        nmsed_scores_shape = [batch_size, max_boxes_to_draw]
        nmsed_classes_shape = [batch_size, max_boxes_to_draw]
        valid_detections_shape = [batch_size]

        def _get_graph_fn(x, y):
            if False:
                return 10
            return self.GraphFn(x, y, max_boxes_to_draw=max_boxes_to_draw, max_detetion_points=max_detetion_points)
        return self.BuildParams(_get_graph_fn, dtypes.float32, [pre_nms_boxes_shape, pre_nms_scores_shape], [nmsed_boxes_shape, nmsed_scores_shape, nmsed_classes_shape, valid_detections_shape])

class CombinedNmsTopKOverride(CombinedNmsTest):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.num_boxes = 5000
        os.environ['TF_TRT_ALLOW_NMS_TOPK_OVERRIDE'] = '1'

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        super().tearDown()
        os.environ['TF_TRT_ALLOW_NMS_TOPK_OVERRIDE'] = '0'

    def GetMaxBatchSize(self, run_params):
        if False:
            for i in range(10):
                print('nop')
        'Returns the max_batch_size that the converter should use for tests.'
        if run_params.dynamic_engine:
            return None
        return super().GetMaxBatchSize(run_params)
if __name__ == '__main__':
    test.main()