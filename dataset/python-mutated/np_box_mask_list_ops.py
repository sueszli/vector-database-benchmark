"""Operations for np_box_mask_list.BoxMaskList.

Example box operations that are supported:
  * Areas: compute bounding box areas
  * IOU: pairwise intersection-over-union scores
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from six.moves import range
from object_detection.utils import np_box_list_ops
from object_detection.utils import np_box_mask_list
from object_detection.utils import np_mask_ops

def box_list_to_box_mask_list(boxlist):
    if False:
        i = 10
        return i + 15
    "Converts a BoxList containing 'masks' into a BoxMaskList.\n\n  Args:\n    boxlist: An np_box_list.BoxList object.\n\n  Returns:\n    An np_box_mask_list.BoxMaskList object.\n\n  Raises:\n    ValueError: If boxlist does not contain `masks` as a field.\n  "
    if not boxlist.has_field('masks'):
        raise ValueError('boxlist does not contain mask field.')
    box_mask_list = np_box_mask_list.BoxMaskList(box_data=boxlist.get(), mask_data=boxlist.get_field('masks'))
    extra_fields = boxlist.get_extra_fields()
    for key in extra_fields:
        if key != 'masks':
            box_mask_list.data[key] = boxlist.get_field(key)
    return box_mask_list

def area(box_mask_list):
    if False:
        while True:
            i = 10
    'Computes area of masks.\n\n  Args:\n    box_mask_list: np_box_mask_list.BoxMaskList holding N boxes and masks\n\n  Returns:\n    a numpy array with shape [N*1] representing mask areas\n  '
    return np_mask_ops.area(box_mask_list.get_masks())

def intersection(box_mask_list1, box_mask_list2):
    if False:
        for i in range(10):
            print('nop')
    'Compute pairwise intersection areas between masks.\n\n  Args:\n    box_mask_list1: BoxMaskList holding N boxes and masks\n    box_mask_list2: BoxMaskList holding M boxes and masks\n\n  Returns:\n    a numpy array with shape [N*M] representing pairwise intersection area\n  '
    return np_mask_ops.intersection(box_mask_list1.get_masks(), box_mask_list2.get_masks())

def iou(box_mask_list1, box_mask_list2):
    if False:
        return 10
    'Computes pairwise intersection-over-union between box and mask collections.\n\n  Args:\n    box_mask_list1: BoxMaskList holding N boxes and masks\n    box_mask_list2: BoxMaskList holding M boxes and masks\n\n  Returns:\n    a numpy array with shape [N, M] representing pairwise iou scores.\n  '
    return np_mask_ops.iou(box_mask_list1.get_masks(), box_mask_list2.get_masks())

def ioa(box_mask_list1, box_mask_list2):
    if False:
        for i in range(10):
            print('nop')
    "Computes pairwise intersection-over-area between box and mask collections.\n\n  Intersection-over-area (ioa) between two masks mask1 and mask2 is defined as\n  their intersection area over mask2's area. Note that ioa is not symmetric,\n  that is, IOA(mask1, mask2) != IOA(mask2, mask1).\n\n  Args:\n    box_mask_list1: np_box_mask_list.BoxMaskList holding N boxes and masks\n    box_mask_list2: np_box_mask_list.BoxMaskList holding M boxes and masks\n\n  Returns:\n    a numpy array with shape [N, M] representing pairwise ioa scores.\n  "
    return np_mask_ops.ioa(box_mask_list1.get_masks(), box_mask_list2.get_masks())

def gather(box_mask_list, indices, fields=None):
    if False:
        return 10
    'Gather boxes from np_box_mask_list.BoxMaskList according to indices.\n\n  By default, gather returns boxes corresponding to the input index list, as\n  well as all additional fields stored in the box_mask_list (indexing into the\n  first dimension).  However one can optionally only gather from a\n  subset of fields.\n\n  Args:\n    box_mask_list: np_box_mask_list.BoxMaskList holding N boxes\n    indices: a 1-d numpy array of type int_\n    fields: (optional) list of fields to also gather from.  If None (default),\n        all fields are gathered from.  Pass an empty fields list to only gather\n        the box coordinates.\n\n  Returns:\n    subbox_mask_list: a np_box_mask_list.BoxMaskList corresponding to the subset\n        of the input box_mask_list specified by indices\n\n  Raises:\n    ValueError: if specified field is not contained in box_mask_list or if the\n        indices are not of type int_\n  '
    if fields is not None:
        if 'masks' not in fields:
            fields.append('masks')
    return box_list_to_box_mask_list(np_box_list_ops.gather(boxlist=box_mask_list, indices=indices, fields=fields))

def sort_by_field(box_mask_list, field, order=np_box_list_ops.SortOrder.DESCEND):
    if False:
        while True:
            i = 10
    "Sort boxes and associated fields according to a scalar field.\n\n  A common use case is reordering the boxes according to descending scores.\n\n  Args:\n    box_mask_list: BoxMaskList holding N boxes.\n    field: A BoxMaskList field for sorting and reordering the BoxMaskList.\n    order: (Optional) 'descend' or 'ascend'. Default is descend.\n\n  Returns:\n    sorted_box_mask_list: A sorted BoxMaskList with the field in the specified\n      order.\n  "
    return box_list_to_box_mask_list(np_box_list_ops.sort_by_field(boxlist=box_mask_list, field=field, order=order))

def non_max_suppression(box_mask_list, max_output_size=10000, iou_threshold=1.0, score_threshold=-10.0):
    if False:
        print('Hello World!')
    "Non maximum suppression.\n\n  This op greedily selects a subset of detection bounding boxes, pruning\n  away boxes that have high IOU (intersection over union) overlap (> thresh)\n  with already selected boxes. In each iteration, the detected bounding box with\n  highest score in the available pool is selected.\n\n  Args:\n    box_mask_list: np_box_mask_list.BoxMaskList holding N boxes.  Must contain\n      a 'scores' field representing detection scores. All scores belong to the\n      same class.\n    max_output_size: maximum number of retained boxes\n    iou_threshold: intersection over union threshold.\n    score_threshold: minimum score threshold. Remove the boxes with scores\n                     less than this value. Default value is set to -10. A very\n                     low threshold to pass pretty much all the boxes, unless\n                     the user sets a different score threshold.\n\n  Returns:\n    an np_box_mask_list.BoxMaskList holding M boxes where M <= max_output_size\n\n  Raises:\n    ValueError: if 'scores' field does not exist\n    ValueError: if threshold is not in [0, 1]\n    ValueError: if max_output_size < 0\n  "
    if not box_mask_list.has_field('scores'):
        raise ValueError('Field scores does not exist')
    if iou_threshold < 0.0 or iou_threshold > 1.0:
        raise ValueError('IOU threshold must be in [0, 1]')
    if max_output_size < 0:
        raise ValueError('max_output_size must be bigger than 0.')
    box_mask_list = filter_scores_greater_than(box_mask_list, score_threshold)
    if box_mask_list.num_boxes() == 0:
        return box_mask_list
    box_mask_list = sort_by_field(box_mask_list, 'scores')
    if iou_threshold == 1.0:
        if box_mask_list.num_boxes() > max_output_size:
            selected_indices = np.arange(max_output_size)
            return gather(box_mask_list, selected_indices)
        else:
            return box_mask_list
    masks = box_mask_list.get_masks()
    num_masks = box_mask_list.num_boxes()
    is_index_valid = np.full(num_masks, 1, dtype=bool)
    selected_indices = []
    num_output = 0
    for i in range(num_masks):
        if num_output < max_output_size:
            if is_index_valid[i]:
                num_output += 1
                selected_indices.append(i)
                is_index_valid[i] = False
                valid_indices = np.where(is_index_valid)[0]
                if valid_indices.size == 0:
                    break
                intersect_over_union = np_mask_ops.iou(np.expand_dims(masks[i], axis=0), masks[valid_indices])
                intersect_over_union = np.squeeze(intersect_over_union, axis=0)
                is_index_valid[valid_indices] = np.logical_and(is_index_valid[valid_indices], intersect_over_union <= iou_threshold)
    return gather(box_mask_list, np.array(selected_indices))

def multi_class_non_max_suppression(box_mask_list, score_thresh, iou_thresh, max_output_size):
    if False:
        for i in range(10):
            print('nop')
    "Multi-class version of non maximum suppression.\n\n  This op greedily selects a subset of detection bounding boxes, pruning\n  away boxes that have high IOU (intersection over union) overlap (> thresh)\n  with already selected boxes.  It operates independently for each class for\n  which scores are provided (via the scores field of the input box_list),\n  pruning boxes with score less than a provided threshold prior to\n  applying NMS.\n\n  Args:\n    box_mask_list: np_box_mask_list.BoxMaskList holding N boxes.  Must contain a\n      'scores' field representing detection scores.  This scores field is a\n      tensor that can be 1 dimensional (in the case of a single class) or\n      2-dimensional, in which case we assume that it takes the\n      shape [num_boxes, num_classes]. We further assume that this rank is known\n      statically and that scores.shape[1] is also known (i.e., the number of\n      classes is fixed and known at graph construction time).\n    score_thresh: scalar threshold for score (low scoring boxes are removed).\n    iou_thresh: scalar threshold for IOU (boxes that that high IOU overlap\n      with previously selected boxes are removed).\n    max_output_size: maximum number of retained boxes per class.\n\n  Returns:\n    a box_mask_list holding M boxes with a rank-1 scores field representing\n      corresponding scores for each box with scores sorted in decreasing order\n      and a rank-1 classes field representing a class label for each box.\n  Raises:\n    ValueError: if iou_thresh is not in [0, 1] or if input box_mask_list does\n      not have a valid scores field.\n  "
    if not 0 <= iou_thresh <= 1.0:
        raise ValueError('thresh must be between 0 and 1')
    if not isinstance(box_mask_list, np_box_mask_list.BoxMaskList):
        raise ValueError('box_mask_list must be a box_mask_list')
    if not box_mask_list.has_field('scores'):
        raise ValueError("input box_mask_list must have 'scores' field")
    scores = box_mask_list.get_field('scores')
    if len(scores.shape) == 1:
        scores = np.reshape(scores, [-1, 1])
    elif len(scores.shape) == 2:
        if scores.shape[1] is None:
            raise ValueError('scores field must have statically defined second dimension')
    else:
        raise ValueError('scores field must be of rank 1 or 2')
    num_boxes = box_mask_list.num_boxes()
    num_scores = scores.shape[0]
    num_classes = scores.shape[1]
    if num_boxes != num_scores:
        raise ValueError('Incorrect scores field length: actual vs expected.')
    selected_boxes_list = []
    for class_idx in range(num_classes):
        box_mask_list_and_class_scores = np_box_mask_list.BoxMaskList(box_data=box_mask_list.get(), mask_data=box_mask_list.get_masks())
        class_scores = np.reshape(scores[0:num_scores, class_idx], [-1])
        box_mask_list_and_class_scores.add_field('scores', class_scores)
        box_mask_list_filt = filter_scores_greater_than(box_mask_list_and_class_scores, score_thresh)
        nms_result = non_max_suppression(box_mask_list_filt, max_output_size=max_output_size, iou_threshold=iou_thresh, score_threshold=score_thresh)
        nms_result.add_field('classes', np.zeros_like(nms_result.get_field('scores')) + class_idx)
        selected_boxes_list.append(nms_result)
    selected_boxes = np_box_list_ops.concatenate(selected_boxes_list)
    sorted_boxes = np_box_list_ops.sort_by_field(selected_boxes, 'scores')
    return box_list_to_box_mask_list(boxlist=sorted_boxes)

def prune_non_overlapping_masks(box_mask_list1, box_mask_list2, minoverlap=0.0):
    if False:
        print('Hello World!')
    "Prunes the boxes in list1 that overlap less than thresh with list2.\n\n  For each mask in box_mask_list1, we want its IOA to be more than minoverlap\n  with at least one of the masks in box_mask_list2. If it does not, we remove\n  it. If the masks are not full size image, we do the pruning based on boxes.\n\n  Args:\n    box_mask_list1: np_box_mask_list.BoxMaskList holding N boxes and masks.\n    box_mask_list2: np_box_mask_list.BoxMaskList holding M boxes and masks.\n    minoverlap: Minimum required overlap between boxes, to count them as\n                overlapping.\n\n  Returns:\n    A pruned box_mask_list with size [N', 4].\n  "
    intersection_over_area = ioa(box_mask_list2, box_mask_list1)
    intersection_over_area = np.amax(intersection_over_area, axis=0)
    keep_bool = np.greater_equal(intersection_over_area, np.array(minoverlap))
    keep_inds = np.nonzero(keep_bool)[0]
    new_box_mask_list1 = gather(box_mask_list1, keep_inds)
    return new_box_mask_list1

def concatenate(box_mask_lists, fields=None):
    if False:
        print('Hello World!')
    'Concatenate list of box_mask_lists.\n\n  This op concatenates a list of input box_mask_lists into a larger\n  box_mask_list.  It also\n  handles concatenation of box_mask_list fields as long as the field tensor\n  shapes are equal except for the first dimension.\n\n  Args:\n    box_mask_lists: list of np_box_mask_list.BoxMaskList objects\n    fields: optional list of fields to also concatenate.  By default, all\n      fields from the first BoxMaskList in the list are included in the\n      concatenation.\n\n  Returns:\n    a box_mask_list with number of boxes equal to\n      sum([box_mask_list.num_boxes() for box_mask_list in box_mask_list])\n  Raises:\n    ValueError: if box_mask_lists is invalid (i.e., is not a list, is empty, or\n      contains non box_mask_list objects), or if requested fields are not\n      contained in all box_mask_lists\n  '
    if fields is not None:
        if 'masks' not in fields:
            fields.append('masks')
    return box_list_to_box_mask_list(np_box_list_ops.concatenate(boxlists=box_mask_lists, fields=fields))

def filter_scores_greater_than(box_mask_list, thresh):
    if False:
        while True:
            i = 10
    "Filter to keep only boxes and masks with score exceeding a given threshold.\n\n  This op keeps the collection of boxes and masks whose corresponding scores are\n  greater than the input threshold.\n\n  Args:\n    box_mask_list: BoxMaskList holding N boxes and masks.  Must contain a\n      'scores' field representing detection scores.\n    thresh: scalar threshold\n\n  Returns:\n    a BoxMaskList holding M boxes and masks where M <= N\n\n  Raises:\n    ValueError: if box_mask_list not a np_box_mask_list.BoxMaskList object or\n      if it does not have a scores field\n  "
    if not isinstance(box_mask_list, np_box_mask_list.BoxMaskList):
        raise ValueError('box_mask_list must be a BoxMaskList')
    if not box_mask_list.has_field('scores'):
        raise ValueError("input box_mask_list must have 'scores' field")
    scores = box_mask_list.get_field('scores')
    if len(scores.shape) > 2:
        raise ValueError('Scores should have rank 1 or 2')
    if len(scores.shape) == 2 and scores.shape[1] != 1:
        raise ValueError('Scores should have rank 1 or have shape consistent with [None, 1]')
    high_score_indices = np.reshape(np.where(np.greater(scores, thresh)), [-1]).astype(np.int32)
    return gather(box_mask_list, high_score_indices)