import math
import numpy as np
import torch
from catalyst.metrics.functional._average_precision import average_precision, binary_average_precision, mean_average_precision

def test_binary_average_precision_base():
    if False:
        return 10
    '\n    Tests for catalyst.binary_average_precision metric.\n    '
    outputs = torch.Tensor([0.1, 0.4, 0.35, 0.8])
    targets = torch.Tensor([0, 0, 1, 1])
    assert torch.isclose(binary_average_precision(outputs, targets), torch.tensor(0.8333), atol=0.001)

def test_binary_average_precision_weighted():
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests for catalyst.binary_average_precision metric.\n    '
    target = torch.Tensor([0, 1, 0, 1])
    output = torch.Tensor([0.1, 0.2, 0.3, 4])
    weight = torch.Tensor([0.5, 1.0, 2.0, 0.1])
    ap = binary_average_precision(outputs=output, targets=target, weights=weight)
    val = (1 * 0.1 / 0.1 + 0 * 2.0 / 2.1 + 1.1 * 1 / 3.1 + 0 * 1 / 4) / 2.0
    assert math.fabs(ap - val) < 0.01, 'ap test1 failed'
    ap = binary_average_precision(outputs=output, targets=target, weights=None)
    val = (1 * 1.0 / 1.0 + 0 * 1.0 / 2.0 + 2 * 1.0 / 3.0 + 0 * 1.0 / 4.0) / 2.0
    assert math.fabs(ap - val) < 0.01, 'ap test2 failed'
    target = torch.Tensor([0, 1, 0, 1])
    output = torch.Tensor([4, 3, 2, 1])
    weight = torch.Tensor([1, 2, 3, 4])
    ap = binary_average_precision(outputs=output, targets=target, weights=weight)
    val = (0 * 1.0 / 1.0 + 1.0 * 2.0 / 3.0 + 2.0 * 0 / 6.0 + 6.0 * 1.0 / 10.0) / 2.0
    assert math.fabs(ap - val) < 0.01, 'ap test3 failed'
    ap = binary_average_precision(outputs=output, targets=target, weights=None)
    val = (0 * 1.0 + 1 * 1.0 / 2.0 + 0 * 1.0 / 3.0 + 2 * 1.0 / 4.0) / 2.0
    assert math.fabs(ap - val) < 0.01, 'ap test4 failed'
    target = torch.Tensor([0, 1, 0, 1])
    output = torch.Tensor([1, 4, 2, 3])
    weight = torch.Tensor([1, 2, 3, 4])
    ap = binary_average_precision(outputs=output, targets=target, weights=weight)
    val = (4 * 1.0 / 4.0 + 6 * 1.0 / 6.0 + 0 * 6.0 / 9.0 + 0 * 6.0 / 10.0) / 2.0
    assert math.fabs(ap - val) < 0.01, 'ap test5 failed'
    ap = binary_average_precision(outputs=output, targets=target, weights=None)
    val = (1 * 1.0 + 2 * 1.0 / 2.0 + 0 * 1.0 / 3.0 + 0 * 1.0 / 4.0) / 2.0
    assert math.fabs(ap - val) < 0.01, 'ap test6 failed'
    target = torch.Tensor([0, 0, 0, 0])
    output = torch.Tensor([1, 4, 2, 3])
    weight = torch.Tensor([1.0, 0.1, 0.0, 0.5])
    ap = binary_average_precision(outputs=output, targets=target, weights=weight)
    val = 0.0
    assert math.fabs(ap - val) < 0.01, 'ap test7 failed'
    ap = binary_average_precision(outputs=output, targets=target, weights=None)
    val = 0.0
    assert math.fabs(ap - val) < 0.01, 'ap test8 failed'
    target = torch.Tensor([1, 1, 0])
    output = torch.Tensor([3, 1, 2])
    weight = torch.Tensor([1, 0.1, 3])
    ap = binary_average_precision(outputs=output, targets=target, weights=weight)
    val = (1 * 1.0 / 1.0 + 1 * 0.0 / 4.0 + 1.1 / 4.1) / 2.0
    assert math.fabs(ap - val) < 0.01, 'ap test9 failed'
    ap = binary_average_precision(outputs=output, targets=target, weights=None)
    val = (1 * 1.0 + 0 * 1.0 / 2.0 + 2 * 1.0 / 3.0) / 2.0
    assert math.fabs(ap - val) < 0.01, 'ap test10 failed'
    target = torch.Tensor([[0, 1, 0, 1], [0, 1, 0, 1]]).transpose(0, 1)
    output = torch.Tensor([[0.1, 0.2, 0.3, 4], [4, 3, 2, 1]]).transpose(0, 1)
    weight = torch.Tensor([[1.0, 0.5, 2.0, 3.0]]).transpose(0, 1)
    ap = binary_average_precision(outputs=output, targets=target, weights=weight)
    assert math.fabs(ap.sum() - torch.Tensor([(1 * 3.0 / 3.0 + 0 * 3.0 / 5.0 + 3.5 * 1 / 5.5 + 0 * 3.5 / 6.5) / 2.0, (0 * 1.0 / 1.0 + 1 * 0.5 / 1.5 + 0 * 0.5 / 3.5 + 1 * 3.5 / 6.5) / 2.0]).sum()) < 0.01, 'ap test11 failed'
    ap = binary_average_precision(outputs=output, targets=target, weights=None)
    assert math.fabs(ap.sum() - torch.Tensor([(1 * 1.0 + 0 * 1.0 / 2.0 + 2 * 1.0 / 3 + 0 * 1.0 / 4.0) / 2.0, (0 * 1.0 + 1 * 1.0 / 2.0 + 0 * 1.0 / 3.0 + 2.0 * 1.0 / 4.0) / 2.0]).sum()) < 0.01, 'ap test12 failed'

def test_average_precision():
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests for catalyst.metrics.average_precision metric.\n    '
    y_pred = [0.5, 0.2, 0.3, 0.8]
    y_true = [1.0, 1.0, 1.0, 1.0]
    k = 4
    avg_precision = average_precision(torch.Tensor([y_pred]), torch.Tensor([y_true]), k)
    assert avg_precision[0] == 1
    y_pred = [0.5, 0.2, 0.3, 0.8]
    y_true = [1.0, 1.0, 1.0, 1.0]
    k = 4
    avg_precision = average_precision(torch.Tensor([y_pred, y_pred, y_pred]), torch.Tensor([y_true, y_true, y_true]), k)
    assert torch.equal(avg_precision, torch.ones(3))
    y_pred = [0.5, 0.2, 0.3, 0.8]
    y_true = [0.0, 0.0, 0.0, 0.0]
    k = 4
    avg_precision = average_precision(torch.Tensor([y_pred]), torch.Tensor([y_true]), k)
    assert avg_precision[0] == 0
    y_pred = [0.5, 0.2, 0.3, 0.8]
    y_true = [0.0, 0.0, 0.0, 0.0]
    k = 4
    avg_precision = average_precision(torch.Tensor([y_pred, y_pred, y_pred]), torch.Tensor([y_true, y_true, y_true]), k)
    assert torch.equal(avg_precision, torch.zeros(3))
    y_pred1 = [4.0, 2.0, 3.0, 1.0]
    y_pred2 = [1.0, 2.0, 3.0, 4.0]
    y_true1 = [0.0, 1.0, 1.0, 1.0]
    y_true2 = [0.0, 1.0, 0.0, 0.0]
    k = 4
    y_pred_torch = torch.Tensor([y_pred1, y_pred2])
    y_true_torch = torch.Tensor([y_true1, y_true2])
    avg_precision = average_precision(y_pred_torch, y_true_torch, k)
    assert np.isclose(avg_precision[0], 0.6389, atol=0.001)
    assert np.isclose(avg_precision[1], 0.333, atol=0.001)
    y_pred1 = np.arange(9, -1, -1)
    y_true1 = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    y_pred2 = np.arange(9, -1, -1)
    y_true2 = [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    k = 10
    y_pred_torch = torch.Tensor([y_pred1, y_pred2])
    y_true_torch = torch.Tensor([y_true1, y_true2])
    avg_precision = average_precision(y_pred_torch, y_true_torch, k)
    assert np.isclose(avg_precision[0], 0.6222, atol=0.001)
    assert np.isclose(avg_precision[1], 0.4429, atol=0.001)

def test_mean_avg_precision():
    if False:
        print('Hello World!')
    '\n    Tests for catalyst.mean_avg_precision metric.\n    '
    y_pred1 = np.arange(9, -1, -1)
    y_true1 = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    y_pred2 = np.arange(9, -1, -1)
    y_true2 = [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    y_pred_torch = torch.Tensor([y_pred1, y_pred2])
    y_true_torch = torch.Tensor([y_true1, y_true2])
    top_k = [10]
    map_at10 = mean_average_precision(y_pred_torch, y_true_torch, top_k)[0]
    assert np.allclose(map_at10, 0.5325, atol=0.001)
    y_pred1 = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    y_pred2 = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    y_true1 = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    y_true2 = [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    y_pred_torch = torch.Tensor([y_pred1, y_pred2])
    y_true_torch = torch.Tensor([y_true1, y_true2])
    top_k = [1, 3, 5, 10]
    map_k = mean_average_precision(y_pred_torch, y_true_torch, top_k)
    map_at1 = map_k[0]
    map_at3 = map_k[1]
    map_at5 = map_k[2]
    map_at10 = map_k[3]
    assert np.allclose(map_at1, 0.5, atol=0.001)
    assert np.allclose(map_at3, 0.6675, atol=0.001)
    assert np.allclose(map_at5, 0.6425, atol=0.001)
    assert np.allclose(map_at10, 0.5325, atol=0.001)