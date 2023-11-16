import pytest
import torch
from torch.autograd.functional import jacobian
from pyro.distributions import Normal, OrderedLogistic
from pyro.distributions.transforms import OrderedTransform

@pytest.mark.parametrize('n_cutpoints', [1, 5, 100])
@pytest.mark.parametrize('pred_shape', [(1,), (5,), (5, 5), (1, 2, 3)])
def test_sample(n_cutpoints, pred_shape):
    if False:
        i = 10
        return i + 15
    predictor = torch.randn(pred_shape)
    cutpoints = torch.sort(torch.randn(n_cutpoints)).values
    dist = OrderedLogistic(predictor, cutpoints, validate_args=True)
    sample = dist.sample([100])
    assert sample.shape[1:] == pred_shape
    assert sample.min().item() >= 0
    assert sample.max().item() <= n_cutpoints

def test_constraints():
    if False:
        i = 10
        return i + 15
    predictor = torch.randn(5)
    for cp in (torch.tensor([1, 2, 3, 4, 0]), torch.tensor([1, 2, 4, 3, 5]), torch.tensor([1, 2, 3, 4, 4])):
        with pytest.raises(ValueError):
            OrderedLogistic(predictor, cp)

def test_broadcast():
    if False:
        i = 10
        return i + 15
    predictor = torch.randn(2, 3, 4)
    for cp in (torch.arange(5), torch.arange(5).view(1, -1), torch.stack(4 * [torch.arange(5)]), torch.sort(torch.randn(3, 4, 5), dim=-1).values, torch.sort(torch.randn(predictor.shape + (100,)), dim=-1).values):
        dist = OrderedLogistic(predictor, cp, validate_args=True)
        assert dist.batch_shape == predictor.shape
        assert dist.sample().shape == predictor.shape
    cp = torch.sort(torch.randn(10, 2, 3, 4, 5), dim=-1).values
    dist = OrderedLogistic(predictor, cp, validate_args=True)
    assert dist.batch_shape == (10,) + predictor.shape
    assert dist.sample().shape == (10,) + predictor.shape

def test_expand():
    if False:
        i = 10
        return i + 15
    predictor = torch.randn(4, 5)
    cutpoints = torch.sort(torch.randn(5, 6)).values
    dist = OrderedLogistic(predictor, cutpoints, validate_args=True)
    new_batch_shape = (2, 3, 4, 5)
    dist = dist.expand(new_batch_shape)
    assert dist.batch_shape == torch.Size(new_batch_shape)
    assert dist.event_shape == torch.Size(())
    sample = dist.sample([100])
    assert torch.all(sample <= 6).item()

def test_autograd():
    if False:
        print('Hello World!')
    predictor = torch.randn(5, requires_grad=True)
    order = OrderedTransform()
    pre_cutpoints = torch.randn(3, requires_grad=True)
    cutpoints = order(pre_cutpoints)
    data = torch.tensor([0, 1, 2, 3, 0], dtype=float)
    dist = OrderedLogistic(predictor, cutpoints, validate_args=True)
    dist.log_prob(data).sum().backward()
    assert predictor.grad is not None
    assert torch.all(predictor.grad != 0).item()
    assert pre_cutpoints.grad is not None
    assert torch.all(pre_cutpoints.grad != 0).item()

@pytest.mark.parametrize('batch_shape', [(), (1,), (5,), (5, 5), (1, 5), (5, 1)])
@pytest.mark.parametrize('event_shape', [(1,), (5,), (100,)])
def test_transform_bijection(batch_shape, event_shape):
    if False:
        return 10
    tf = OrderedTransform()
    assert tf.inv.inv is tf
    shape = torch.Size(batch_shape + event_shape)
    sample = Normal(0, 1).expand(shape).sample()
    tf_sample = tf(sample)
    inv_tf_sample = tf.inv(tf_sample)
    assert torch.allclose(sample, inv_tf_sample)

def cjald(func, X):
    if False:
        i = 10
        return i + 15
    'cjald = Computes Jacobian Along Last Dimension\n    Recursively splits tensor ``X`` along its leading dimensions until we are\n    left with a vector, computes the jacobian of this vector under the\n    transformation ``func``, then stitches all the results back together using\n    ``torch.stack``.\n    '
    assert X.ndim >= 1
    if X.ndim == 1:
        return jacobian(func, X)
    else:
        return torch.stack([cjald(func, X[i]) for i in range(X.shape[0])], dim=0)

@pytest.mark.parametrize('batch_shape', [(), (1,), (5,), (5, 5), (1, 5), (5, 1)])
@pytest.mark.parametrize('event_shape', [(1,), (5,), (100,)])
def test_transform_log_abs_det(batch_shape, event_shape):
    if False:
        i = 10
        return i + 15
    tf = OrderedTransform()
    shape = torch.Size(batch_shape + event_shape)
    x = torch.randn(shape, requires_grad=True)
    y = tf(x)
    log_det = tf.log_abs_det_jacobian(x, y)
    assert log_det.shape == batch_shape
    log_det_actual = cjald(tf, x).det().abs().log()
    assert torch.allclose(log_det, log_det_actual)