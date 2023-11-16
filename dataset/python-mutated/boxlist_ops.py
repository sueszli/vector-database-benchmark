import torch
from .bounding_box import BoxList

def remove_small_boxes(boxlist, min_size):
    if False:
        i = 10
        return i + 15
    '\n    Only keep boxes with both sides >= min_size\n    Arguments:\n        boxlist (Boxlist)\n        min_size (int)\n    '
    xywh_boxes = boxlist.convert('xywh').bbox
    (_, _, ws, hs) = xywh_boxes.unbind(dim=1)
    keep = ((ws >= min_size) & (hs >= min_size)).nonzero().squeeze(1)
    return boxlist[keep]

def boxlist_iou(boxlist1, boxlist2):
    if False:
        while True:
            i = 10
    'Compute the intersection over union of two set of boxes.\n    The box order must be (xmin, ymin, xmax, ymax).\n    Arguments:\n      box1: (BoxList) bounding boxes, sized [N,4].\n      box2: (BoxList) bounding boxes, sized [M,4].\n    Returns:\n      (tensor) iou, sized [N,M].\n    Reference:\n      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py\n    '
    if boxlist1.size != boxlist2.size:
        raise RuntimeError('boxlists should have same image size, got {}, {}'.format(boxlist1, boxlist2))
    area1 = boxlist1.area()
    area2 = boxlist2.area()
    (box1, box2) = (boxlist1.bbox, boxlist2.bbox)
    lt = torch.max(box1[:, None, :2], box2[:, :2])
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])
    TO_REMOVE = 1
    wh = (rb - lt + TO_REMOVE).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    iou = inter / (area1[:, None] + area2 - inter)
    return iou

def _cat(tensors, dim=0):
    if False:
        return 10
    '\n    Efficient version of torch.cat that avoids a copy if there is only\n    a single element in a list\n    '
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)

def cat_boxlist(bboxes):
    if False:
        i = 10
        return i + 15
    '\n    Concatenates a list of BoxList (having the same image size) into a\n    single BoxList\n    Arguments:\n        bboxes (list[BoxList])\n    '
    assert isinstance(bboxes, (list, tuple))
    assert all((isinstance(bbox, BoxList) for bbox in bboxes))
    size = bboxes[0].size
    assert all((bbox.size == size for bbox in bboxes))
    mode = bboxes[0].mode
    assert all((bbox.mode == mode for bbox in bboxes))
    fields = set(bboxes[0].fields())
    assert all((set(bbox.fields()) == fields for bbox in bboxes))
    cat_boxes = BoxList(_cat([bbox.bbox for bbox in bboxes], dim=0), size, mode)
    for field in fields:
        data = _cat([bbox.get_field(field) for bbox in bboxes], dim=0)
        cat_boxes.add_field(field, data)
    return cat_boxes