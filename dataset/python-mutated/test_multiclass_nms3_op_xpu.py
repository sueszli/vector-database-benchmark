import copy
import unittest
import numpy as np
from get_test_cover_info import XPUOpTestWrapper, create_test_class, get_xpu_op_support_types
from op_test_xpu import XPUOpTest
import paddle
paddle.enable_static()

def softmax(x):
    if False:
        print('Hello World!')
    shiftx = (x - np.max(x)).clip(-64.0)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)

def iou(box_a, box_b, norm):
    if False:
        print('Hello World!')
    'Apply intersection-over-union overlap between box_a and box_b.'
    xmin_a = min(box_a[0], box_a[2])
    ymin_a = min(box_a[1], box_a[3])
    xmax_a = max(box_a[0], box_a[2])
    ymax_a = max(box_a[1], box_a[3])
    xmin_b = min(box_b[0], box_b[2])
    ymin_b = min(box_b[1], box_b[3])
    xmax_b = max(box_b[0], box_b[2])
    ymax_b = max(box_b[1], box_b[3])
    area_a = (ymax_a - ymin_a + (not norm)) * (xmax_a - xmin_a + (not norm))
    area_b = (ymax_b - ymin_b + (not norm)) * (xmax_b - xmin_b + (not norm))
    if area_a <= 0 and area_b <= 0:
        return 0.0
    xa = max(xmin_a, xmin_b)
    ya = max(ymin_a, ymin_b)
    xb = min(xmax_a, xmax_b)
    yb = min(ymax_a, ymax_b)
    inter_area = max(xb - xa + (not norm), 0.0) * max(yb - ya + (not norm), 0.0)
    iou_ratio = inter_area / (area_a + area_b - inter_area)
    return iou_ratio

def nms(boxes, scores, score_threshold, nms_threshold, top_k=200, normalized=True, eta=1.0):
    if False:
        print('Hello World!')
    'Apply non-maximum suppression at test time to avoid detecting too many\n    overlapping bounding boxes for a given object.\n    Args:\n        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].\n        scores: (tensor) The class predscores for the img, Shape:[num_priors].\n        score_threshold: (float) The confidence thresh for filtering low\n            confidence boxes.\n        nms_threshold: (float) The overlap thresh for suppressing unnecessary\n            boxes.\n        top_k: (int) The maximum number of box preds to consider.\n        eta: (float) The parameter for adaptive NMS.\n    Return:\n        The indices of the kept boxes with respect to num_priors.\n    '
    all_scores = copy.deepcopy(scores)
    all_scores = all_scores.flatten()
    selected_indices = np.argwhere(all_scores > score_threshold)
    selected_indices = selected_indices.flatten()
    all_scores = all_scores[selected_indices]
    sorted_indices = np.argsort(-all_scores, axis=0, kind='mergesort')
    sorted_scores = all_scores[sorted_indices]
    sorted_indices = selected_indices[sorted_indices]
    if top_k > -1 and top_k < sorted_indices.shape[0]:
        sorted_indices = sorted_indices[:top_k]
        sorted_scores = sorted_scores[:top_k]
    selected_indices = []
    adaptive_threshold = nms_threshold
    for i in range(sorted_scores.shape[0]):
        idx = sorted_indices[i]
        keep = True
        for k in range(len(selected_indices)):
            if keep:
                kept_idx = selected_indices[k]
                overlap = iou(boxes[idx], boxes[kept_idx], normalized)
                keep = True if overlap <= adaptive_threshold else False
            else:
                break
        if keep:
            selected_indices.append(idx)
        if keep and eta < 1 and (adaptive_threshold > 0.5):
            adaptive_threshold *= eta
    return selected_indices

def multiclass_nms(boxes, scores, background, score_threshold, nms_threshold, nms_top_k, keep_top_k, normalized, shared):
    if False:
        return 10
    if shared:
        class_num = scores.shape[0]
        priorbox_num = scores.shape[1]
    else:
        box_num = scores.shape[0]
        class_num = scores.shape[1]
    selected_indices = {}
    num_det = 0
    for c in range(class_num):
        if c == background:
            continue
        if shared:
            indices = nms(boxes, scores[c], score_threshold, nms_threshold, nms_top_k, normalized)
        else:
            indices = nms(boxes[:, c, :], scores[:, c], score_threshold, nms_threshold, nms_top_k, normalized)
        selected_indices[c] = indices
        num_det += len(indices)
    if keep_top_k > -1 and num_det > keep_top_k:
        score_index = []
        for (c, indices) in selected_indices.items():
            for idx in indices:
                if shared:
                    score_index.append((scores[c][idx], c, idx))
                else:
                    score_index.append((scores[idx][c], c, idx))
        sorted_score_index = sorted(score_index, key=lambda tup: tup[0], reverse=True)
        sorted_score_index = sorted_score_index[:keep_top_k]
        selected_indices = {}
        for (_, c, _) in sorted_score_index:
            selected_indices[c] = []
        for (s, c, idx) in sorted_score_index:
            selected_indices[c].append(idx)
        if not shared:
            for labels in selected_indices:
                selected_indices[labels].sort()
        num_det = keep_top_k
    return (selected_indices, num_det)

def batched_multiclass_nms(boxes, scores, background, score_threshold, nms_threshold, nms_top_k, keep_top_k, normalized=True, gpu_logic=False):
    if False:
        return 10
    batch_size = scores.shape[0]
    num_boxes = scores.shape[2]
    det_outs = []
    index_outs = []
    lod = []
    for n in range(batch_size):
        (nmsed_outs, nmsed_num) = multiclass_nms(boxes[n], scores[n], background, score_threshold, nms_threshold, nms_top_k, keep_top_k, normalized, shared=True)
        lod.append(nmsed_num)
        if nmsed_num == 0:
            continue
        tmp_det_out = []
        for (c, indices) in nmsed_outs.items():
            for idx in indices:
                (xmin, ymin, xmax, ymax) = boxes[n][idx][:]
                tmp_det_out.append([c, scores[n][c][idx], xmin, ymin, xmax, ymax, idx + n * num_boxes])
        if gpu_logic:
            sorted_det_out = sorted(tmp_det_out, key=lambda tup: tup[1], reverse=True)
        else:
            sorted_det_out = sorted(tmp_det_out, key=lambda tup: tup[0], reverse=False)
        det_outs.extend(sorted_det_out)
    return (det_outs, lod)

class TestIOU(unittest.TestCase):

    def test_iou(self):
        if False:
            for i in range(10):
                print('nop')
        box1 = np.array([4.0, 3.0, 7.0, 5.0]).astype('float32')
        box2 = np.array([3.0, 4.0, 6.0, 8.0]).astype('float32')
        expt_output = np.array([2.0 / 16.0]).astype('float32')
        calc_output = np.array([iou(box1, box2, True)]).astype('float32')
        np.testing.assert_allclose(calc_output, expt_output, rtol=1e-05)

class XPUTestMulticlassNMS3Op(XPUOpTestWrapper):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.op_name = 'multiclass_nms3'
        self.use_dynamic_create_class = False

    class TestXpuMulticlassNMS3Op(XPUOpTest):

        def set_argument(self):
            if False:
                while True:
                    i = 10
            self.score_threshold = 0.01

        def setUp(self):
            if False:
                for i in range(10):
                    print('nop')
            self.op_type = 'multiclass_nms3'
            self.dtype = self.in_type
            self.set_argument()
            N = 7
            M = 1200
            C = 21
            BOX_SIZE = 4
            background = 0
            nms_threshold = 0.3
            nms_top_k = 400
            keep_top_k = 200 if not hasattr(self, 'keep_top_k') else self.keep_top_k
            score_threshold = self.score_threshold
            scores = np.random.random((N * M, C)).astype(self.dtype)
            scores = np.apply_along_axis(softmax, 1, scores)
            scores = np.reshape(scores, (N, M, C))
            scores = np.transpose(scores, (0, 2, 1))
            boxes = np.random.random((N, M, BOX_SIZE)).astype(self.dtype)
            boxes[:, :, 0:2] = boxes[:, :, 0:2] * 0.5
            boxes[:, :, 2:4] = boxes[:, :, 2:4] * 0.5 + 0.5
            (det_outs, lod) = batched_multiclass_nms(boxes, scores, background, score_threshold, nms_threshold, nms_top_k, keep_top_k, gpu_logic=self.gpu_logic if hasattr(self, 'gpu_logic') else None)
            det_outs = np.array(det_outs)
            nmsed_outs = det_outs[:, :-1].astype(self.dtype) if len(det_outs) else np.array([], dtype=np.float32).reshape([0, BOX_SIZE + 2])
            index_outs = det_outs[:, -1:].astype('int') if len(det_outs) else np.array([], dtype='int').reshape([0, 1])
            self.inputs = {'BBoxes': boxes, 'Scores': scores}
            self.outputs = {'Out': nmsed_outs, 'Index': index_outs, 'NmsRoisNum': np.array(lod).astype('int32')}
            self.attrs = {'background_label': 0, 'nms_threshold': nms_threshold, 'nms_top_k': nms_top_k, 'keep_top_k': keep_top_k, 'score_threshold': score_threshold, 'nms_eta': 1.0, 'normalized': True}

        def test_check_output(self):
            if False:
                i = 10
                return i + 15
            self.check_output_with_place(paddle.XPUPlace(0))
support_types = get_xpu_op_support_types('multiclass_nms3')
for stype in support_types:
    create_test_class(globals(), XPUTestMulticlassNMS3Op, stype)
if __name__ == '__main__':
    unittest.main()