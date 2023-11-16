import torch
from mmdet.core.bbox import bbox_overlaps
from mmdet.core.bbox.iou_calculators.builder import IOU_CALCULATORS
from ..structures import get_box_type

@IOU_CALCULATORS.register_module()
class BboxOverlapsNearest3D(object):
    """Nearest 3D IoU Calculator.

    Note:
        This IoU calculator first finds the nearest 2D boxes in bird eye view
        (BEV), and then calculates the 2D IoU using :meth:`bbox_overlaps`.

    Args:
        coordinate (str): 'camera', 'lidar', or 'depth' coordinate system.
    """

    def __init__(self, coordinate='lidar'):
        if False:
            while True:
                i = 10
        assert coordinate in ['camera', 'lidar', 'depth']
        self.coordinate = coordinate

    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False):
        if False:
            for i in range(10):
                print('nop')
        'Calculate nearest 3D IoU.\n\n        Note:\n            If ``is_aligned`` is ``False``, then it calculates the ious between\n            each bbox of bboxes1 and bboxes2, otherwise it calculates the ious\n            between each aligned pair of bboxes1 and bboxes2.\n\n        Args:\n            bboxes1 (torch.Tensor): shape (N, 7+N)\n                [x, y, z, x_size, y_size, z_size, ry, v].\n            bboxes2 (torch.Tensor): shape (M, 7+N)\n                [x, y, z, x_size, y_size, z_size, ry, v].\n            mode (str): "iou" (intersection over union) or iof\n                (intersection over foreground).\n            is_aligned (bool): Whether the calculation is aligned.\n\n        Return:\n            torch.Tensor: If ``is_aligned`` is ``True``, return ious between\n                bboxes1 and bboxes2 with shape (M, N). If ``is_aligned`` is\n                ``False``, return shape is M.\n        '
        return bbox_overlaps_nearest_3d(bboxes1, bboxes2, mode, is_aligned, self.coordinate)

    def __repr__(self):
        if False:
            return 10
        'str: Return a string that describes the module.'
        repr_str = self.__class__.__name__
        repr_str += f'(coordinate={self.coordinate}'
        return repr_str

@IOU_CALCULATORS.register_module()
class BboxOverlaps3D(object):
    """3D IoU Calculator.

    Args:
        coordinate (str): The coordinate system, valid options are
            'camera', 'lidar', and 'depth'.
    """

    def __init__(self, coordinate):
        if False:
            for i in range(10):
                print('nop')
        assert coordinate in ['camera', 'lidar', 'depth']
        self.coordinate = coordinate

    def __call__(self, bboxes1, bboxes2, mode='iou'):
        if False:
            for i in range(10):
                print('nop')
        'Calculate 3D IoU using cuda implementation.\n\n        Note:\n            This function calculate the IoU of 3D boxes based on their volumes.\n            IoU calculator ``:class:BboxOverlaps3D`` uses this function to\n            calculate the actual 3D IoUs of boxes.\n\n        Args:\n            bboxes1 (torch.Tensor): with shape (N, 7+C),\n                (x, y, z, x_size, y_size, z_size, ry, v*).\n            bboxes2 (torch.Tensor): with shape (M, 7+C),\n                (x, y, z, x_size, y_size, z_size, ry, v*).\n            mode (str): "iou" (intersection over union) or\n                iof (intersection over foreground).\n\n        Return:\n            torch.Tensor: Bbox overlaps results of bboxes1 and bboxes2\n                with shape (M, N) (aligned mode is not supported currently).\n        '
        return bbox_overlaps_3d(bboxes1, bboxes2, mode, self.coordinate)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        'str: return a string that describes the module'
        repr_str = self.__class__.__name__
        repr_str += f'(coordinate={self.coordinate}'
        return repr_str

def bbox_overlaps_nearest_3d(bboxes1, bboxes2, mode='iou', is_aligned=False, coordinate='lidar'):
    if False:
        i = 10
        return i + 15
    'Calculate nearest 3D IoU.\n\n    Note:\n        This function first finds the nearest 2D boxes in bird eye view\n        (BEV), and then calculates the 2D IoU using :meth:`bbox_overlaps`.\n        This IoU calculator :class:`BboxOverlapsNearest3D` uses this\n        function to calculate IoUs of boxes.\n\n        If ``is_aligned`` is ``False``, then it calculates the ious between\n        each bbox of bboxes1 and bboxes2, otherwise the ious between each\n        aligned pair of bboxes1 and bboxes2.\n\n    Args:\n        bboxes1 (torch.Tensor): with shape (N, 7+C),\n            (x, y, z, x_size, y_size, z_size, ry, v*).\n        bboxes2 (torch.Tensor): with shape (M, 7+C),\n            (x, y, z, x_size, y_size, z_size, ry, v*).\n        mode (str): "iou" (intersection over union) or iof\n            (intersection over foreground).\n        is_aligned (bool): Whether the calculation is aligned\n\n    Return:\n        torch.Tensor: If ``is_aligned`` is ``True``, return ious between\n            bboxes1 and bboxes2 with shape (M, N). If ``is_aligned`` is\n            ``False``, return shape is M.\n    '
    assert bboxes1.size(-1) == bboxes2.size(-1) >= 7
    (box_type, _) = get_box_type(coordinate)
    bboxes1 = box_type(bboxes1, box_dim=bboxes1.shape[-1])
    bboxes2 = box_type(bboxes2, box_dim=bboxes2.shape[-1])
    bboxes1_bev = bboxes1.nearest_bev
    bboxes2_bev = bboxes2.nearest_bev
    ret = bbox_overlaps(bboxes1_bev, bboxes2_bev, mode=mode, is_aligned=is_aligned)
    return ret

def bbox_overlaps_3d(bboxes1, bboxes2, mode='iou', coordinate='camera'):
    if False:
        for i in range(10):
            print('nop')
    'Calculate 3D IoU using cuda implementation.\n\n    Note:\n        This function calculates the IoU of 3D boxes based on their volumes.\n        IoU calculator :class:`BboxOverlaps3D` uses this function to\n        calculate the actual IoUs of boxes.\n\n    Args:\n        bboxes1 (torch.Tensor): with shape (N, 7+C),\n            (x, y, z, x_size, y_size, z_size, ry, v*).\n        bboxes2 (torch.Tensor): with shape (M, 7+C),\n            (x, y, z, x_size, y_size, z_size, ry, v*).\n        mode (str): "iou" (intersection over union) or\n            iof (intersection over foreground).\n        coordinate (str): \'camera\' or \'lidar\' coordinate system.\n\n    Return:\n        torch.Tensor: Bbox overlaps results of bboxes1 and bboxes2\n            with shape (M, N) (aligned mode is not supported currently).\n    '
    assert bboxes1.size(-1) == bboxes2.size(-1) >= 7
    (box_type, _) = get_box_type(coordinate)
    bboxes1 = box_type(bboxes1, box_dim=bboxes1.shape[-1])
    bboxes2 = box_type(bboxes2, box_dim=bboxes2.shape[-1])
    return bboxes1.overlaps(bboxes1, bboxes2, mode=mode)

@IOU_CALCULATORS.register_module()
class AxisAlignedBboxOverlaps3D(object):
    """Axis-aligned 3D Overlaps (IoU) Calculator."""

    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False):
        if False:
            return 10
        'Calculate IoU between 2D bboxes.\n\n        Args:\n            bboxes1 (Tensor): shape (B, m, 6) in <x1, y1, z1, x2, y2, z2>\n                format or empty.\n            bboxes2 (Tensor): shape (B, n, 6) in <x1, y1, z1, x2, y2, z2>\n                format or empty.\n                B indicates the batch dim, in shape (B1, B2, ..., Bn).\n                If ``is_aligned`` is ``True``, then m and n must be equal.\n            mode (str): "iou" (intersection over union) or "giou" (generalized\n                intersection over union).\n            is_aligned (bool, optional): If True, then m and n must be equal.\n                Defaults to False.\n        Returns:\n            Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)\n        '
        assert bboxes1.size(-1) == bboxes2.size(-1) == 6
        return axis_aligned_bbox_overlaps_3d(bboxes1, bboxes2, mode, is_aligned)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        'str: a string describing the module'
        repr_str = self.__class__.__name__ + '()'
        return repr_str

def axis_aligned_bbox_overlaps_3d(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-06):
    if False:
        return 10
    'Calculate overlap between two set of axis aligned 3D bboxes. If\n    ``is_aligned`` is ``False``, then calculate the overlaps between each bbox\n    of bboxes1 and bboxes2, otherwise the overlaps between each aligned pair of\n    bboxes1 and bboxes2.\n\n    Args:\n        bboxes1 (Tensor): shape (B, m, 6) in <x1, y1, z1, x2, y2, z2>\n            format or empty.\n        bboxes2 (Tensor): shape (B, n, 6) in <x1, y1, z1, x2, y2, z2>\n            format or empty.\n            B indicates the batch dim, in shape (B1, B2, ..., Bn).\n            If ``is_aligned`` is ``True``, then m and n must be equal.\n        mode (str): "iou" (intersection over union) or "giou" (generalized\n            intersection over union).\n        is_aligned (bool, optional): If True, then m and n must be equal.\n            Defaults to False.\n        eps (float, optional): A value added to the denominator for numerical\n            stability. Defaults to 1e-6.\n\n    Returns:\n        Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)\n\n    Example:\n        >>> bboxes1 = torch.FloatTensor([\n        >>>     [0, 0, 0, 10, 10, 10],\n        >>>     [10, 10, 10, 20, 20, 20],\n        >>>     [32, 32, 32, 38, 40, 42],\n        >>> ])\n        >>> bboxes2 = torch.FloatTensor([\n        >>>     [0, 0, 0, 10, 20, 20],\n        >>>     [0, 10, 10, 10, 19, 20],\n        >>>     [10, 10, 10, 20, 20, 20],\n        >>> ])\n        >>> overlaps = axis_aligned_bbox_overlaps_3d(bboxes1, bboxes2)\n        >>> assert overlaps.shape == (3, 3)\n        >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)\n        >>> assert overlaps.shape == (3, )\n    Example:\n        >>> empty = torch.empty(0, 6)\n        >>> nonempty = torch.FloatTensor([[0, 0, 0, 10, 9, 10]])\n        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)\n        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)\n        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)\n    '
    assert mode in ['iou', 'giou'], f'Unsupported mode {mode}'
    assert bboxes1.size(-1) == 6 or bboxes1.size(0) == 0
    assert bboxes2.size(-1) == 6 or bboxes2.size(0) == 0
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]
    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols
    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows,))
        else:
            return bboxes1.new(batch_shape + (rows, cols))
    area1 = (bboxes1[..., 3] - bboxes1[..., 0]) * (bboxes1[..., 4] - bboxes1[..., 1]) * (bboxes1[..., 5] - bboxes1[..., 2])
    area2 = (bboxes2[..., 3] - bboxes2[..., 0]) * (bboxes2[..., 4] - bboxes2[..., 1]) * (bboxes2[..., 5] - bboxes2[..., 2])
    if is_aligned:
        lt = torch.max(bboxes1[..., :3], bboxes2[..., :3])
        rb = torch.min(bboxes1[..., 3:], bboxes2[..., 3:])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[..., 0] * wh[..., 1] * wh[..., 2]
        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :3], bboxes2[..., :3])
            enclosed_rb = torch.max(bboxes1[..., 3:], bboxes2[..., 3:])
    else:
        lt = torch.max(bboxes1[..., :, None, :3], bboxes2[..., None, :, :3])
        rb = torch.min(bboxes1[..., :, None, 3:], bboxes2[..., None, :, 3:])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[..., 0] * wh[..., 1] * wh[..., 2]
        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :3], bboxes2[..., None, :, :3])
            enclosed_rb = torch.max(bboxes1[..., :, None, 3:], bboxes2[..., None, :, 3:])
    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou']:
        return ious
    enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1] * enclose_wh[..., 2]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious