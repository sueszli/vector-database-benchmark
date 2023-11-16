from unittest.mock import patch
import numpy as np
import pytest
import sklearn
import torch
from sklearn.metrics import roc_curve
from ignite import distributed as idist
from ignite.contrib.metrics.roc_auc import RocCurve
from ignite.engine import Engine
from ignite.exceptions import NotComputableError
from ignite.metrics.epoch_metric import EpochMetricWarning

def test_wrong_setup():
    if False:
        print('Hello World!')

    def compute_fn(y_preds, y_targets):
        if False:
            for i in range(10):
                print('nop')
        return 0.0
    with pytest.raises(NotComputableError, match='RocCurve must have at least one example before it can be computed'):
        metric = RocCurve(compute_fn)
        metric.compute()

@pytest.fixture()
def mock_no_sklearn():
    if False:
        i = 10
        return i + 15
    with patch.dict('sys.modules', {'sklearn.metrics': None}):
        yield sklearn

def test_no_sklearn(mock_no_sklearn):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ModuleNotFoundError, match='This contrib module requires scikit-learn to be installed'):
        RocCurve()

def test_roc_curve():
    if False:
        while True:
            i = 10
    size = 100
    np_y_pred = np.random.rand(size, 1)
    np_y = np.zeros((size,))
    np_y[size // 2:] = 1
    (sk_fpr, sk_tpr, sk_thresholds) = roc_curve(np_y, np_y_pred)
    roc_curve_metric = RocCurve()
    y_pred = torch.from_numpy(np_y_pred)
    y = torch.from_numpy(np_y)
    roc_curve_metric.update((y_pred, y))
    (fpr, tpr, thresholds) = roc_curve_metric.compute()
    assert np.array_equal(fpr, sk_fpr)
    assert np.array_equal(tpr, sk_tpr)
    np.testing.assert_array_almost_equal(thresholds, sk_thresholds)

def test_integration_roc_curve_with_output_transform():
    if False:
        i = 10
        return i + 15
    np.random.seed(1)
    size = 100
    np_y_pred = np.random.rand(size, 1)
    np_y = np.zeros((size,))
    np_y[size // 2:] = 1
    np.random.shuffle(np_y)
    (sk_fpr, sk_tpr, sk_thresholds) = roc_curve(np_y, np_y_pred)
    batch_size = 10

    def update_fn(engine, batch):
        if False:
            i = 10
            return i + 15
        idx = (engine.state.iteration - 1) * batch_size
        y_true_batch = np_y[idx:idx + batch_size]
        y_pred_batch = np_y_pred[idx:idx + batch_size]
        return (idx, torch.from_numpy(y_pred_batch), torch.from_numpy(y_true_batch))
    engine = Engine(update_fn)
    roc_curve_metric = RocCurve(output_transform=lambda x: (x[1], x[2]))
    roc_curve_metric.attach(engine, 'roc_curve')
    data = list(range(size // batch_size))
    (fpr, tpr, thresholds) = engine.run(data, max_epochs=1).metrics['roc_curve']
    assert np.array_equal(fpr, sk_fpr)
    assert np.array_equal(tpr, sk_tpr)
    np.testing.assert_array_almost_equal(thresholds, sk_thresholds)

def test_integration_roc_curve_with_activated_output_transform():
    if False:
        i = 10
        return i + 15
    np.random.seed(1)
    size = 100
    np_y_pred = np.random.rand(size, 1)
    np_y_pred_sigmoid = torch.sigmoid(torch.from_numpy(np_y_pred)).numpy()
    np_y = np.zeros((size,))
    np_y[size // 2:] = 1
    np.random.shuffle(np_y)
    (sk_fpr, sk_tpr, sk_thresholds) = roc_curve(np_y, np_y_pred_sigmoid)
    batch_size = 10

    def update_fn(engine, batch):
        if False:
            print('Hello World!')
        idx = (engine.state.iteration - 1) * batch_size
        y_true_batch = np_y[idx:idx + batch_size]
        y_pred_batch = np_y_pred[idx:idx + batch_size]
        return (idx, torch.from_numpy(y_pred_batch), torch.from_numpy(y_true_batch))
    engine = Engine(update_fn)
    roc_curve_metric = RocCurve(output_transform=lambda x: (torch.sigmoid(x[1]), x[2]))
    roc_curve_metric.attach(engine, 'roc_curve')
    data = list(range(size // batch_size))
    (fpr, tpr, thresholds) = engine.run(data, max_epochs=1).metrics['roc_curve']
    assert np.array_equal(fpr, sk_fpr)
    assert np.array_equal(tpr, sk_tpr)
    np.testing.assert_array_almost_equal(thresholds, sk_thresholds)

def test_check_compute_fn():
    if False:
        print('Hello World!')
    y_pred = torch.zeros((8, 13))
    y_pred[:, 1] = 1
    y_true = torch.zeros_like(y_pred)
    output = (y_pred, y_true)
    em = RocCurve(check_compute_fn=True)
    em.reset()
    with pytest.warns(EpochMetricWarning, match='Probably, there can be a problem with `compute_fn`'):
        em.update(output)
    em = RocCurve(check_compute_fn=False)
    em.update(output)

def test_distrib_integration(distributed):
    if False:
        return 10
    rank = idist.get_rank()
    torch.manual_seed(41 + rank)
    (n_batches, batch_size) = (5, 10)
    y = torch.randint(0, 2, size=(n_batches * batch_size,))
    y_pred = torch.rand((n_batches * batch_size,))

    def update(engine, i):
        if False:
            for i in range(10):
                print('nop')
        return (y_pred[i * batch_size:(i + 1) * batch_size], y[i * batch_size:(i + 1) * batch_size])
    engine = Engine(update)
    device = torch.device('cpu') if idist.device().type == 'xla' else idist.device()
    metric = RocCurve(device=device)
    metric.attach(engine, 'roc_curve')
    data = list(range(n_batches))
    engine.run(data=data, max_epochs=1)
    (fpr, tpr, thresholds) = engine.state.metrics['roc_curve']
    assert isinstance(fpr, torch.Tensor) and fpr.device == device
    assert isinstance(tpr, torch.Tensor) and tpr.device == device
    assert isinstance(thresholds, torch.Tensor) and thresholds.device == device
    y = idist.all_gather(y)
    y_pred = idist.all_gather(y_pred)
    (sk_fpr, sk_tpr, sk_thresholds) = roc_curve(y.cpu().numpy(), y_pred.cpu().numpy())
    np.testing.assert_array_almost_equal(fpr.cpu().numpy(), sk_fpr)
    np.testing.assert_array_almost_equal(tpr.cpu().numpy(), sk_tpr)
    np.testing.assert_array_almost_equal(thresholds.cpu().numpy(), sk_thresholds)