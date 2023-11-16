import numpy as np
import pytest
import torch
from sklearn.metrics import multilabel_confusion_matrix
import ignite.distributed as idist
from ignite.exceptions import NotComputableError
from ignite.metrics.multilabel_confusion_matrix import MultiLabelConfusionMatrix
torch.manual_seed(12)

def test_no_update():
    if False:
        while True:
            i = 10
    cm = MultiLabelConfusionMatrix(10)
    with pytest.raises(NotComputableError, match='Confusion matrix must have at least one example before it can be computed'):
        cm.compute()

def test_num_classes_wrong_input():
    if False:
        print('Hello World!')
    with pytest.raises(ValueError, match='Argument num_classes needs to be > 1'):
        MultiLabelConfusionMatrix(num_classes=1)

def test_multiclass_wrong_inputs():
    if False:
        while True:
            i = 10
    cm = MultiLabelConfusionMatrix(10)
    with pytest.raises(ValueError, match='y_pred must at least have shape \\(batch_size, num_classes \\(currently set to 10\\), ...\\)'):
        cm.update((torch.rand(10), torch.randint(0, 2, size=(10, 10)).long()))
    with pytest.raises(ValueError, match='y must at least have shape \\(batch_size, num_classes \\(currently set to 10\\), ...\\)'):
        cm.update((torch.rand(10, 10), torch.randint(0, 2, size=(10,)).long()))
    with pytest.raises(ValueError, match='y_pred and y have different batch size: 10 vs 8'):
        cm.update((torch.rand(10, 10), torch.randint(0, 2, size=(8, 10)).long()))
    with pytest.raises(ValueError, match='y does not have correct number of classes: 9 vs 10'):
        cm.update((torch.rand(10, 10), torch.randint(0, 2, size=(10, 9)).long()))
    with pytest.raises(ValueError, match='y_pred does not have correct number of classes: 3 vs 10'):
        cm.update((torch.rand(10, 3), torch.randint(0, 2, size=(10, 10)).long()))
    with pytest.raises(ValueError, match='y and y_pred shapes must match.'):
        cm.update((torch.rand(10, 10, 2), torch.randint(0, 2, size=(10, 10)).long()))
    with pytest.raises(ValueError, match='y_pred must be of any type: \\(torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64\\)'):
        cm.update((torch.rand(10, 10), torch.rand(10, 10)))
    with pytest.raises(ValueError, match='y must be of any type: \\(torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64\\)'):
        cm.update((torch.rand(10, 10).type(torch.int32), torch.rand(10, 10)))
    with pytest.raises(ValueError, match='y_pred must be a binary tensor'):
        y = torch.randint(0, 2, size=(10, 10)).long()
        y_pred = torch.randint(0, 2, size=(10, 10)).long()
        y_pred[0, 0] = 2
        cm.update((y_pred, y))
    with pytest.raises(ValueError, match='y must be a binary tensor'):
        y = torch.randint(0, 2, size=(10, 10)).long()
        y_pred = torch.randint(0, 2, size=(10, 10)).long()
        y[0, 0] = 2
        cm.update((y_pred, y))

def get_y_true_y_pred():
    if False:
        return 10
    y_true = np.zeros((1, 3, 30, 30), dtype=np.int64)
    y_true[0, 0, 5:17, 7:11] = 1
    y_true[0, 1, 1:11, 1:11] = 1
    y_true[0, 2, 15:25, 15:25] = 1
    y_pred = np.zeros((1, 3, 30, 30), dtype=np.int64)
    y_pred[0, 0, 0:7, 8:15] = 1
    y_pred[0, 1, 5:15, 1:11] = 1
    y_pred[0, 2, 20:30, 20:30] = 1
    return (y_true, y_pred)

def test_multiclass_images():
    if False:
        for i in range(10):
            print('nop')
    num_classes = 3
    cm = MultiLabelConfusionMatrix(num_classes=num_classes)
    (y_true, y_pred) = get_y_true_y_pred()
    sklearn_CM = multilabel_confusion_matrix(y_true.transpose((0, 2, 3, 1)).reshape(-1, 3), y_pred.transpose((0, 2, 3, 1)).reshape(-1, 3))
    output = (torch.tensor(y_pred), torch.tensor(y_true))
    cm.update(output)
    ignite_CM = cm.compute().cpu().numpy()
    assert np.all(ignite_CM == sklearn_CM)
    cm = MultiLabelConfusionMatrix(num_classes=num_classes)
    th_y_true1 = torch.tensor(y_true)
    th_y_true2 = torch.tensor(y_true.transpose(0, 1, 3, 2))
    th_y_true = torch.cat([th_y_true1, th_y_true2], dim=0)
    th_y_pred1 = torch.tensor(y_pred)
    th_y_pred2 = torch.tensor(y_pred.transpose(0, 1, 3, 2))
    th_y_pred = torch.cat([th_y_pred1, th_y_pred2], dim=0)
    output = (th_y_pred, th_y_true)
    cm.update(output)
    ignite_CM = cm.compute().cpu().numpy()
    th_y_true = idist.all_gather(th_y_true)
    th_y_pred = idist.all_gather(th_y_pred)
    np_y_true = th_y_true.cpu().numpy().transpose((0, 2, 3, 1)).reshape(-1, 3)
    np_y_pred = th_y_pred.cpu().numpy().transpose((0, 2, 3, 1)).reshape(-1, 3)
    sklearn_CM = multilabel_confusion_matrix(np_y_true, np_y_pred)
    assert np.all(ignite_CM == sklearn_CM)

def _test_distrib_multiclass_images(device):
    if False:
        i = 10
        return i + 15

    def _test(metric_device):
        if False:
            while True:
                i = 10
        num_classes = 3
        cm = MultiLabelConfusionMatrix(num_classes=num_classes, device=metric_device)
        (y_true, y_pred) = get_y_true_y_pred()
        sklearn_CM = multilabel_confusion_matrix(y_true.transpose((0, 2, 3, 1)).reshape(-1, 3), y_pred.transpose((0, 2, 3, 1)).reshape(-1, 3))
        output = (torch.tensor(y_pred).to(device), torch.tensor(y_true).to(device))
        cm.update(output)
        ignite_CM = cm.compute().cpu().numpy()
        assert np.all(ignite_CM == sklearn_CM)
        num_classes = 3
        cm = MultiLabelConfusionMatrix(num_classes=num_classes, device=metric_device)
        th_y_true1 = torch.tensor(y_true)
        th_y_true2 = torch.tensor(y_true.transpose(0, 1, 3, 2))
        th_y_true = torch.cat([th_y_true1, th_y_true2], dim=0)
        th_y_true = th_y_true.to(device)
        th_y_pred1 = torch.tensor(y_pred)
        th_y_pred2 = torch.tensor(y_pred.transpose(0, 1, 3, 2))
        th_y_pred = torch.cat([th_y_pred1, th_y_pred2], dim=0)
        th_y_pred = th_y_pred.to(device)
        output = (th_y_pred, th_y_true)
        cm.update(output)
        ignite_CM = cm.compute().cpu().numpy()
        th_y_true = idist.all_gather(th_y_true)
        th_y_pred = idist.all_gather(th_y_pred)
        np_y_true = th_y_true.cpu().numpy().transpose((0, 2, 3, 1)).reshape(-1, 3)
        np_y_pred = th_y_pred.cpu().numpy().transpose((0, 2, 3, 1)).reshape(-1, 3)
        sklearn_CM = multilabel_confusion_matrix(np_y_true, np_y_pred)
        assert np.all(ignite_CM == sklearn_CM)
    _test('cpu')
    if device.type != 'xla':
        _test(idist.device())

def _test_distrib_accumulator_device(device):
    if False:
        i = 10
        return i + 15
    metric_devices = [torch.device('cpu')]
    if device.type != 'xla':
        metric_devices.append(idist.device())
    for metric_device in metric_devices:
        cm = MultiLabelConfusionMatrix(num_classes=3, device=metric_device)
        assert cm._device == metric_device
        assert cm.confusion_matrix.device == metric_device, f'{type(cm.confusion_matrix.device)}:{cm._num_correct.device} vs {type(metric_device)}:{metric_device}'
        (y_true, y_pred) = get_y_true_y_pred()
        cm.update((torch.tensor(y_pred), torch.tensor(y_true)))
        assert cm.confusion_matrix.device == metric_device, f'{type(cm.confusion_matrix.device)}:{cm._num_correct.device} vs {type(metric_device)}:{metric_device}'

def test_simple_2D_input():
    if False:
        i = 10
        return i + 15
    num_iters = 5
    num_samples = 100
    num_classes = 10
    torch.manual_seed(0)
    for _ in range(num_iters):
        target = torch.randint(0, 2, size=(num_samples, num_classes))
        prediction = torch.randint(0, 2, size=(num_samples, num_classes))
        sklearn_CM = multilabel_confusion_matrix(target.numpy(), prediction.numpy())
        mlcm = MultiLabelConfusionMatrix(num_classes, normalized=False)
        mlcm.update([prediction, target])
        ignite_CM = mlcm.compute().numpy()
        assert np.all(sklearn_CM.astype(np.int64) == ignite_CM.astype(np.int64))
        mlcm = MultiLabelConfusionMatrix(num_classes, normalized=True)
        mlcm.update([prediction, target])
        ignite_CM_normalized = mlcm.compute().numpy()
        sklearn_CM_normalized = sklearn_CM / sklearn_CM.sum(axis=(1, 2))[:, None, None]
        assert np.allclose(sklearn_CM_normalized, ignite_CM_normalized)

def test_simple_ND_input():
    if False:
        i = 10
        return i + 15
    num_iters = 5
    num_samples = 100
    num_classes = 10
    torch.manual_seed(0)
    size_3d = 4
    for _ in range(num_iters):
        target = torch.randint(0, 2, size=(num_samples, num_classes, size_3d))
        prediction = torch.randint(0, 2, size=(num_samples, num_classes, size_3d))
        mlcm = MultiLabelConfusionMatrix(num_classes, normalized=False)
        mlcm.update([prediction, target])
        ignite_CM = mlcm.compute().numpy()
        target_reshaped = target.permute(0, 2, 1).reshape(size_3d * num_samples, num_classes)
        prediction_reshaped = prediction.permute(0, 2, 1).reshape(size_3d * num_samples, num_classes)
        sklearn_CM = multilabel_confusion_matrix(target_reshaped.numpy(), prediction_reshaped.numpy())
        assert np.all(sklearn_CM.astype(np.int64) == ignite_CM.astype(np.int64))
    size_4d = 4
    for _ in range(num_iters):
        target = torch.randint(0, 2, size=(num_samples, num_classes, size_3d, size_4d))
        prediction = torch.randint(0, 2, size=(num_samples, num_classes, size_3d, size_4d))
        mlcm = MultiLabelConfusionMatrix(num_classes, normalized=False)
        mlcm.update([prediction, target])
        ignite_CM = mlcm.compute().numpy()
        target_reshaped = target.permute(0, 2, 3, 1).reshape(size_3d * size_4d * num_samples, num_classes)
        prediction_reshaped = prediction.permute(0, 2, 3, 1).reshape(size_3d * size_4d * num_samples, num_classes)
        sklearn_CM = multilabel_confusion_matrix(target_reshaped.numpy(), prediction_reshaped.numpy())
        assert np.all(sklearn_CM.astype(np.int64) == ignite_CM.astype(np.int64))
    size_5d = 4
    for _ in range(num_iters):
        target = torch.randint(0, 2, size=(num_samples, num_classes, size_3d, size_4d, size_5d))
        prediction = torch.randint(0, 2, size=(num_samples, num_classes, size_3d, size_4d, size_5d))
        mlcm = MultiLabelConfusionMatrix(num_classes, normalized=False)
        mlcm.update([prediction, target])
        ignite_CM = mlcm.compute().numpy()
        target_reshaped = target.permute(0, 2, 3, 4, 1).reshape(size_3d * size_4d * size_5d * num_samples, num_classes)
        prediction_reshaped = prediction.permute(0, 2, 3, 4, 1).reshape(size_3d * size_4d * size_5d * num_samples, num_classes)
        sklearn_CM = multilabel_confusion_matrix(target_reshaped.numpy(), prediction_reshaped.numpy())
        assert np.all(sklearn_CM.astype(np.int64) == ignite_CM.astype(np.int64))

def test_simple_batched():
    if False:
        return 10
    num_iters = 5
    num_samples = 100
    num_classes = 10
    batch_size = 1
    torch.manual_seed(0)
    for _ in range(num_iters):
        mlcm = MultiLabelConfusionMatrix(num_classes, normalized=False)
        targets = torch.randint(0, 2, size=(int(num_samples / batch_size), batch_size, num_classes))
        predictions = torch.randint(0, 2, size=(int(num_samples / batch_size), batch_size, num_classes))
        for i in range(int(num_samples / batch_size)):
            target_sample = targets[i]
            prediction_sample = predictions[i]
            mlcm.update([prediction_sample, target_sample])
        ignite_CM = mlcm.compute().numpy()
        targets_reshaped = targets.reshape(-1, num_classes)
        predictions_reshaped = predictions.reshape(-1, num_classes)
        sklearn_CM = multilabel_confusion_matrix(targets_reshaped.numpy(), predictions_reshaped.numpy())
        assert np.all(sklearn_CM.astype(np.int64) == ignite_CM.astype(np.int64))
    size_3d = 4
    for _ in range(num_iters):
        mlcm = MultiLabelConfusionMatrix(num_classes, normalized=False)
        targets = torch.randint(0, 2, size=(int(num_samples / batch_size), batch_size, num_classes, size_3d))
        predictions = torch.randint(0, 2, size=(int(num_samples / batch_size), batch_size, num_classes, size_3d))
        for i in range(int(num_samples / batch_size)):
            target_sample = targets[i]
            prediction_sample = predictions[i]
            mlcm.update([prediction_sample, target_sample])
        ignite_CM = mlcm.compute().numpy()
        targets_reshaped = targets.permute(0, 1, 3, 2).reshape(-1, num_classes)
        predictions_reshaped = predictions.permute(0, 1, 3, 2).reshape(-1, num_classes)
        sklearn_CM = multilabel_confusion_matrix(targets_reshaped.numpy(), predictions_reshaped.numpy())
        assert np.all(sklearn_CM.astype(np.int64) == ignite_CM.astype(np.int64))
    size_4d = 4
    for _ in range(num_iters):
        mlcm = MultiLabelConfusionMatrix(num_classes, normalized=False)
        targets = torch.randint(0, 2, size=(int(num_samples / batch_size), batch_size, num_classes, size_3d, size_4d))
        predictions = torch.randint(0, 2, size=(int(num_samples / batch_size), batch_size, num_classes, size_3d, size_4d))
        for i in range(int(num_samples / batch_size)):
            target_sample = targets[i]
            prediction_sample = predictions[i]
            mlcm.update([prediction_sample, target_sample])
        ignite_CM = mlcm.compute().numpy()
        targets_reshaped = targets.permute(0, 1, 3, 4, 2).reshape(-1, num_classes)
        predictions_reshaped = predictions.permute(0, 1, 3, 4, 2).reshape(-1, num_classes)
        sklearn_CM = multilabel_confusion_matrix(targets_reshaped.numpy(), predictions_reshaped.numpy())
        assert np.all(sklearn_CM.astype(np.int64) == ignite_CM.astype(np.int64))
    size_5d = 4
    for _ in range(num_iters):
        mlcm = MultiLabelConfusionMatrix(num_classes, normalized=False)
        targets = torch.randint(0, 2, size=(int(num_samples / batch_size), batch_size, num_classes, size_3d, size_4d, size_5d))
        predictions = torch.randint(0, 2, size=(int(num_samples / batch_size), batch_size, num_classes, size_3d, size_4d, size_5d))
        for i in range(int(num_samples / batch_size)):
            target_sample = targets[i]
            prediction_sample = predictions[i]
            mlcm.update([prediction_sample, target_sample])
        ignite_CM = mlcm.compute().numpy()
        targets_reshaped = targets.permute(0, 1, 3, 4, 5, 2).reshape(-1, num_classes)
        predictions_reshaped = predictions.permute(0, 1, 3, 4, 5, 2).reshape(-1, num_classes)
        sklearn_CM = multilabel_confusion_matrix(targets_reshaped.numpy(), predictions_reshaped.numpy())
        assert np.all(sklearn_CM.astype(np.int64) == ignite_CM.astype(np.int64))