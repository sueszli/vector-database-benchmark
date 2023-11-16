import itertools
import unittest
import numpy as np
from inference_pass_test import InferencePassTest
import paddle
from paddle import base
from paddle.base import core
from paddle.base.core import AnalysisConfig, PassVersionChecker
from paddle.base.layer_helper import LayerHelper
from paddle.framework import in_dynamic_mode
from paddle.static import nn

def multiclass_nms(bboxes, scores, score_threshold, nms_top_k, keep_top_k, nms_threshold=0.3, normalized=True, nms_eta=1.0, background_label=-1, return_index=False, return_rois_num=True, rois_num=None, name=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    This operator is to do multi-class non maximum suppression (NMS) on\n    boxes and scores.\n    In the NMS step, this operator greedily selects a subset of detection bounding\n    boxes that have high scores larger than score_threshold, if providing this\n    threshold, then selects the largest nms_top_k confidences scores if nms_top_k\n    is larger than -1. Then this operator pruns away boxes that have high IOU\n    (intersection over union) overlap with already selected boxes by adaptive\n    threshold NMS based on parameters of nms_threshold and nms_eta.\n    Aftern NMS step, at most keep_top_k number of total bboxes are to be kept\n    per image if keep_top_k is larger than -1.\n    Args:\n        bboxes (Tensor): Two types of bboxes are supported:\n                           1. (Tensor) A 3-D Tensor with shape\n                           [N, M, 4 or 8 16 24 32] represents the\n                           predicted locations of M bounding bboxes,\n                           N is the batch size. Each bounding box has four\n                           coordinate values and the layout is\n                           [xmin, ymin, xmax, ymax], when box size equals to 4.\n                           2. (LoDTensor) A 3-D Tensor with shape [M, C, 4]\n                           M is the number of bounding boxes, C is the\n                           class number\n        scores (Tensor): Two types of scores are supported:\n                           1. (Tensor) A 3-D Tensor with shape [N, C, M]\n                           represents the predicted confidence predictions.\n                           N is the batch size, C is the class number, M is\n                           number of bounding boxes. For each category there\n                           are total M scores which corresponding M bounding\n                           boxes. Please note, M is equal to the 2nd dimension\n                           of BBoxes.\n                           2. (LoDTensor) A 2-D LoDTensor with shape [M, C].\n                           M is the number of bbox, C is the class number.\n                           In this case, input BBoxes should be the second\n                           case with shape [M, C, 4].\n        background_label (int): The index of background label, the background\n                                label will be ignored. If set to -1, then all\n                                categories will be considered. Default: 0\n        score_threshold (float): Threshold to filter out bounding boxes with\n                                 low confidence score. If not provided,\n                                 consider all boxes.\n        nms_top_k (int): Maximum number of detections to be kept according to\n                         the confidences after the filtering detections based\n                         on score_threshold.\n        nms_threshold (float): The threshold to be used in NMS. Default: 0.3\n        nms_eta (float): The threshold to be used in NMS. Default: 1.0\n        keep_top_k (int): Number of total bboxes to be kept per image after NMS\n                          step. -1 means keeping all bboxes after NMS step.\n        normalized (bool): Whether detections are normalized. Default: True\n        return_index(bool): Whether return selected index. Default: False\n        rois_num(Tensor): 1-D Tensor contains the number of RoIs in each image.\n            The shape is [B] and data type is int32. B is the number of images.\n            If it is not None then return a list of 1-D Tensor. Each element\n            is the output RoIs' number of each image on the corresponding level\n            and the shape is [B]. None by default.\n        name(str): Name of the multiclass nms op. Default: None.\n    Returns:\n        A tuple with two Variables: (Out, Index) if return_index is True,\n        otherwise, a tuple with one Variable(Out) is returned.\n        Out: A 2-D LoDTensor with shape [No, 6] represents the detections.\n        Each row has 6 values: [label, confidence, xmin, ymin, xmax, ymax]\n        or A 2-D LoDTensor with shape [No, 10] represents the detections.\n        Each row has 10 values: [label, confidence, x1, y1, x2, y2, x3, y3,\n        x4, y4]. No is the total number of detections.\n        If all images have not detected results, all elements in LoD will be\n        0, and output tensor is empty (None).\n        Index: Only return when return_index is True. A 2-D LoDTensor with\n        shape [No, 1] represents the selected index which type is Integer.\n        The index is the absolute value cross batches. No is the same number\n        as Out. If the index is used to gather other attribute such as age,\n        one needs to reshape the input(N, M, 1) to (N * M, 1) as first, where\n        N is the batch size and M is the number of boxes.\n    Examples:\n        .. code-block:: python\n            import paddle\n            from ppdet.modeling import ops\n            boxes = paddle.static.data(name='bboxes', shape=[81, 4],\n                                      dtype='float32', lod_level=1)\n            scores = paddle.static.data(name='scores', shape=[81],\n                                      dtype='float32', lod_level=1)\n            out, index = ops.multiclass_nms(bboxes=boxes,\n                                            scores=scores,\n                                            background_label=0,\n                                            score_threshold=0.5,\n                                            nms_top_k=400,\n                                            nms_threshold=0.3,\n                                            keep_top_k=200,\n                                            normalized=False,\n                                            return_index=True)\n    "
    if in_dynamic_mode():
        attrs = ('background_label', background_label, 'score_threshold', score_threshold, 'nms_top_k', nms_top_k, 'nms_threshold', nms_threshold, 'keep_top_k', keep_top_k, 'nms_eta', nms_eta, 'normalized', normalized)
        (output, index, nms_rois_num) = core.eager.ops.legacy.multiclass_nms3(bboxes, scores, rois_num, *attrs)
        if not return_index:
            index = None
        return (output, nms_rois_num, index)
    else:
        helper = LayerHelper('multiclass_nms3', **locals())
        output = helper.create_variable_for_type_inference(dtype=bboxes.dtype)
        index = helper.create_variable_for_type_inference(dtype='int32')
        inputs = {'BBoxes': bboxes, 'Scores': scores}
        outputs = {'Out': output, 'Index': index}
        if rois_num is not None:
            inputs['RoisNum'] = rois_num
        if return_rois_num:
            nms_rois_num = helper.create_variable_for_type_inference(dtype='int32')
            outputs['NmsRoisNum'] = nms_rois_num
        helper.append_op(type='multiclass_nms3', inputs=inputs, attrs={'background_label': background_label, 'score_threshold': score_threshold, 'nms_top_k': nms_top_k, 'nms_threshold': nms_threshold, 'keep_top_k': keep_top_k, 'nms_eta': nms_eta, 'normalized': normalized}, outputs=outputs)
        output.stop_gradient = True
        index.stop_gradient = True
        if not return_index:
            index = None
        if not return_rois_num:
            nms_rois_num = None
        return (output, nms_rois_num, index)

class TensorRTMultiClassNMS3Test(InferencePassTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.enable_trt = True
        self.enable_tensorrt_varseqlen = True
        self.precision = AnalysisConfig.Precision.Float32
        self.serialize = False
        self.bs = 1
        self.background_label = -1
        self.score_threshold = 0.5
        self.nms_top_k = 8
        self.nms_threshold = 0.3
        self.keep_top_k = 8
        self.normalized = False
        self.num_classes = 8
        self.num_boxes = 8
        self.nms_eta = 1.1
        self.trt_parameters = InferencePassTest.TensorRTParam(1 << 30, self.bs, 2, self.precision, self.serialize, False)

    def build(self):
        if False:
            i = 10
            return i + 15
        with base.program_guard(self.main_program, self.startup_program):
            boxes = paddle.static.data(name='bboxes', shape=[-1, self.num_boxes, 4], dtype='float32')
            scores = paddle.static.data(name='scores', shape=[-1, self.num_classes, self.num_boxes], dtype='float32')
            (multiclass_nms_out, _, _) = multiclass_nms(bboxes=boxes, scores=scores, background_label=self.background_label, score_threshold=self.score_threshold, nms_top_k=self.nms_top_k, nms_threshold=self.nms_threshold, keep_top_k=self.keep_top_k, normalized=self.normalized, nms_eta=self.nms_eta)
            mutliclass_nms_out = multiclass_nms_out + 1.0
            multiclass_nms_out = paddle.reshape(multiclass_nms_out, [self.bs, 1, self.keep_top_k, 6], name='reshape')
            out = nn.batch_norm(multiclass_nms_out, is_test=True)
        boxes_data = np.arange(self.num_boxes * 4).reshape([self.bs, self.num_boxes, 4]).astype('float32')
        scores_data = np.arange(1 * self.num_classes * self.num_boxes).reshape([self.bs, self.num_classes, self.num_boxes]).astype('float32')
        self.feeds = {'bboxes': boxes_data, 'scores': scores_data}
        self.fetch_list = [out]

    def run_test(self):
        if False:
            print('Hello World!')
        self.build()
        self.check_output()

    def run_test_all(self):
        if False:
            for i in range(10):
                print('nop')
        precision_opt = [AnalysisConfig.Precision.Float32, AnalysisConfig.Precision.Half]
        serialize_opt = [False, True]
        max_shape = {'bboxes': [self.bs, self.num_boxes, 4], 'scores': [self.bs, self.num_classes, self.num_boxes]}
        opt_shape = max_shape
        dynamic_shape_opt = [None, InferencePassTest.DynamicShapeParam({'bboxes': [1, 1, 4], 'scores': [1, 1, 1]}, max_shape, opt_shape, False)]
        for (precision, serialize, dynamic_shape) in itertools.product(precision_opt, serialize_opt, dynamic_shape_opt):
            self.precision = precision
            self.serialize = serialize
            self.dynamic_shape_params = dynamic_shape
            self.build()
            self.check_output()

    def check_output(self):
        if False:
            while True:
                i = 10
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))

    def test_base(self):
        if False:
            i = 10
            return i + 15
        self.run_test()

    def test_fp16(self):
        if False:
            return 10
        self.precision = AnalysisConfig.Precision.Half
        self.run_test()

    def test_serialize(self):
        if False:
            while True:
                i = 10
        self.serialize = True
        self.run_test()

    def test_dynamic(self):
        if False:
            for i in range(10):
                print('nop')
        max_shape = {'bboxes': [self.bs, self.num_boxes, 4], 'scores': [self.bs, self.num_classes, self.num_boxes]}
        opt_shape = max_shape
        self.dynamic_shape_params = InferencePassTest.DynamicShapeParam({'bboxes': [1, 1, 4], 'scores': [1, 1, 1]}, max_shape, opt_shape, False)
        self.run_test()

    def test_background(self):
        if False:
            i = 10
            return i + 15
        self.background = 7
        self.run_test()

    def test_disable_varseqlen(self):
        if False:
            while True:
                i = 10
        self.diable_tensorrt_varseqlen = False
        self.run_test()
if __name__ == '__main__':
    unittest.main()