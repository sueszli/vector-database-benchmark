"""
The implementation here is modified based on insightface, originally MIT license and publicly available at
https://github.com/deepinsight/insightface/tree/master/detection/scrfd/mmdet/core/post_processing/bbox_nms.py
"""
import torch

def multiclass_nms(multi_bboxes, multi_scores, score_thr, nms_cfg, max_num=-1, score_factors=None, return_inds=False, multi_kps=None):
    if False:
        while True:
            i = 10
    'NMS for multi-class bboxes.\n\n    Args:\n        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)\n        multi_kps (Tensor): shape (n, #class*num_kps*2) or (n, num_kps*2)\n        multi_scores (Tensor): shape (n, #class), where the last column\n            contains scores of the background class, but this will be ignored.\n        score_thr (float): bbox threshold, bboxes with scores lower than it\n            will not be considered.\n        nms_thr (float): NMS IoU threshold\n        max_num (int, optional): if there are more than max_num bboxes after\n            NMS, only top max_num will be kept. Default to -1.\n        score_factors (Tensor, optional): The factors multiplied to scores\n            before applying NMS. Default to None.\n        return_inds (bool, optional): Whether return the indices of kept\n            bboxes. Default to False.\n\n    Returns:\n        tuple: (bboxes, labels, indices (optional)), tensors of shape (k, 5),\n            (k), and (k). Labels are 0-based.\n    '
    num_classes = multi_scores.size(1) - 1
    kps = None
    if multi_kps is not None:
        num_kps = int(multi_kps.shape[1] / num_classes / 2)
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
        if multi_kps is not None:
            kps = multi_kps.view(multi_scores.size(0), -1, num_kps * 2)
    else:
        bboxes = multi_bboxes[:, None].expand(multi_scores.size(0), num_classes, 4)
        if multi_kps is not None:
            kps = multi_kps[:, None].expand(multi_scores.size(0), num_classes, num_kps * 2)
    scores = multi_scores[:, :-1]
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    labels = torch.arange(num_classes, dtype=torch.long, device=scores.device)
    labels = labels.view(1, -1).expand_as(scores)
    bboxes = bboxes.reshape(-1, 4)
    if kps is not None:
        kps = kps.reshape(-1, num_kps * 2)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)
    valid_mask = scores > score_thr
    inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
    (bboxes, scores, labels) = (bboxes[inds], scores[inds], labels[inds])
    if kps is not None:
        kps = kps[inds]
    if inds.numel() == 0:
        if torch.onnx.is_in_onnx_export():
            raise RuntimeError('[ONNX Error] Can not record NMS as it has not been executed this time')
        return (bboxes, labels, kps)
    from mmcv.ops.nms import batched_nms
    (dets, keep) = batched_nms(bboxes, scores, labels, nms_cfg)
    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]
    if return_inds:
        return (dets, labels[keep], kps[keep], keep)
    else:
        return (dets, labels[keep], kps[keep])