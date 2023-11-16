import torch
from catalyst.metrics.functional import iou

def test_iou():
    if False:
        i = 10
        return i + 15
    '\n    Tests for catalyst.metrics.iou metric.\n    '
    size = 4
    half_size = size // 2
    shape = (1, 1, size, size)
    empty = torch.zeros(shape)
    full = torch.ones(shape)
    assert iou(empty, full, class_dim=1, mode='per-class').item() == 0
    left = torch.ones(shape)
    left[:, :, :, half_size:] = 0
    right = torch.ones(shape)
    right[:, :, :, :half_size] = 0
    assert iou(left, right, class_dim=1, mode='per-class').item() == 0
    assert iou(empty, empty, class_dim=1, mode='per-class').item() == 1
    assert iou(full, full, class_dim=1, mode='per-class').item() == 1
    assert iou(left, left, class_dim=1, mode='per-class').item() == 1
    top_left = torch.zeros(shape)
    top_left[:, :, :half_size, :half_size] = 1
    assert torch.isclose(iou(top_left, left, class_dim=1, mode='per-class'), torch.Tensor([[0.5]]))
    assert torch.isclose(iou(top_left, left, class_dim=1, mode='micro'), torch.Tensor([[0.5]]))
    assert torch.isclose(iou(top_left, left, class_dim=1, mode='macro'), torch.Tensor([[0.5]]))
    a = torch.cat([empty, left, empty, full, left, top_left], dim=1)
    b = torch.cat([full, right, empty, full, left, left], dim=1)
    ans = torch.Tensor([0, 0, 1, 1, 1, 0.5])
    ans_micro = torch.tensor(0.4375)
    assert torch.allclose(iou(a, b, class_dim=1, mode='per-class'), ans)
    assert torch.allclose(iou(a, b, class_dim=1, mode='micro'), ans_micro)
    aaa = torch.cat([a, a, a], dim=0)
    bbb = torch.cat([b, b, b], dim=0)
    assert torch.allclose(iou(aaa, bbb, class_dim=1, mode='per-class'), ans)
    assert torch.allclose(iou(aaa, bbb, class_dim=1, mode='micro'), ans_micro)