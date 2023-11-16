import argparse
import math
import numpy as np
import torch
from torch.optim import Adam
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer import Trace_ELBO
from pyro.infer.autoguide import AutoDelta, init_to_median
"\nWe demonstrate how to do sparse linear regression using a variant of the\napproach described in [1]. This approach is particularly suitable for situations\nwith many feature dimensions (large P) but not too many datapoints (small N).\nIn particular we consider a quadratic regressor of the form:\n\nf(X) = constant + sum_i theta_i X_i + sum_{i<j} theta_ij X_i X_j + observation noise\n\nNote that in order to keep the set of identified non-negligible weights theta_i\nand theta_ij sparse, the model assumes the weights satisfy a 'strong hierarchy'\ncondition. See reference [1] for details.\n\nNote that in contrast to [1] we do MAP estimation for the kernel hyperparameters\ninstead of HMC. This is not expected to be as robust as doing full Bayesian inference,\nbut in some regimes this works surprisingly well. For the latter HMC approach see\nthe NumPyro version:\n\nhttps://github.com/pyro-ppl/numpyro/blob/master/examples/sparse_regression.py\n\nReferences\n[1] The Kernel Interaction Trick: Fast Bayesian Discovery of Pairwise\n    Interactions in High Dimensions.\n    Raj Agrawal, Jonathan H. Huggins, Brian Trippe, Tamara Broderick\n    https://arxiv.org/abs/1905.06501\n"
torch.set_default_dtype(torch.float32)

def dot(X, Z):
    if False:
        for i in range(10):
            print('nop')
    return torch.mm(X, Z.t())

def kernel(X, Z, eta1, eta2, c):
    if False:
        for i in range(10):
            print('nop')
    (eta1sq, eta2sq) = (eta1.pow(2.0), eta2.pow(2.0))
    k1 = 0.5 * eta2sq * (1.0 + dot(X, Z)).pow(2.0)
    k2 = -0.5 * eta2sq * dot(X.pow(2.0), Z.pow(2.0))
    k3 = (eta1sq - eta2sq) * dot(X, Z)
    k4 = c ** 2 - 0.5 * eta2sq
    return k1 + k2 + k3 + k4

def model(X, Y, hypers, jitter=0.0001):
    if False:
        i = 10
        return i + 15
    (S, P, N) = (hypers['expected_sparsity'], X.size(1), X.size(0))
    sigma = pyro.sample('sigma', dist.HalfNormal(hypers['alpha3']))
    phi = sigma * (S / math.sqrt(N)) / (P - S)
    eta1 = pyro.sample('eta1', dist.HalfCauchy(phi))
    msq = pyro.sample('msq', dist.InverseGamma(hypers['alpha1'], hypers['beta1']))
    xisq = pyro.sample('xisq', dist.InverseGamma(hypers['alpha2'], hypers['beta2']))
    eta2 = eta1.pow(2.0) * xisq.sqrt() / msq
    lam = pyro.sample('lambda', dist.HalfCauchy(torch.ones(P, device=X.device)).to_event(1))
    kappa = msq.sqrt() * lam / (msq + (eta1 * lam).pow(2.0)).sqrt()
    kX = kappa * X
    k = kernel(kX, kX, eta1, eta2, hypers['c']) + (sigma ** 2 + jitter) * torch.eye(N, device=X.device)
    pyro.sample('Y', dist.MultivariateNormal(torch.zeros(N, device=X.device), covariance_matrix=k), obs=Y)
'\nHere we compute the mean and variance of coefficients theta_i (where i = dimension) as well\nas for quadratic coefficients theta_ij for a given (in our case MAP) estimate of the kernel\nhyperparameters (eta1, xisq, ...).\nCompare to theorem 5.1 in reference [1].\n'

@torch.no_grad()
def compute_posterior_stats(X, Y, msq, lam, eta1, xisq, c, sigma, jitter=0.0001):
    if False:
        for i in range(10):
            print('nop')
    (N, P) = X.shape
    probe = torch.zeros((P, 2, P), dtype=X.dtype, device=X.device)
    probe[:, 0, :] = torch.eye(P, dtype=X.dtype, device=X.device)
    probe[:, 1, :] = -torch.eye(P, dtype=X.dtype, device=X.device)
    eta2 = eta1.pow(2.0) * xisq.sqrt() / msq
    kappa = msq.sqrt() * lam / (msq + (eta1 * lam).pow(2.0)).sqrt()
    kX = kappa * X
    kprobe = kappa * probe
    kprobe = kprobe.reshape(-1, P)
    k_xx = kernel(kX, kX, eta1, eta2, c) + (jitter + sigma ** 2) * torch.eye(N, dtype=X.dtype, device=X.device)
    k_xx_inv = torch.inverse(k_xx)
    k_probeX = kernel(kprobe, kX, eta1, eta2, c)
    k_prbprb = kernel(kprobe, kprobe, eta1, eta2, c)
    vec = torch.tensor([0.5, -0.5], dtype=X.dtype, device=X.device)
    mu = torch.matmul(k_probeX, torch.matmul(k_xx_inv, Y).unsqueeze(-1)).squeeze(-1).reshape(P, 2)
    mu = (mu * vec).sum(-1)
    var = k_prbprb - torch.matmul(k_probeX, torch.matmul(k_xx_inv, k_probeX.t()))
    var = var.reshape(P, 2, P, 2).diagonal(dim1=-4, dim2=-2)
    std = ((var * vec.unsqueeze(-1)).sum(-2) * vec.unsqueeze(-1)).sum(-2).clamp(min=0.0).sqrt()
    active_dims = ((mu - 4.0 * std > 0.0) | (mu + 4.0 * std < 0.0)).bool()
    active_dims = active_dims.nonzero(as_tuple=False).squeeze(-1)
    print('Identified the following active dimensions:', active_dims.data.numpy().flatten())
    print('Mean estimate for active singleton weights:\n', mu[active_dims].data.numpy())
    M = len(active_dims)
    if M < 2:
        return (active_dims.data.numpy(), [])
    (left_dims, right_dims) = torch.ones(M, M).triu(1).nonzero(as_tuple=False).t()
    (left_dims, right_dims) = (active_dims[left_dims], active_dims[right_dims])
    probe = torch.zeros(left_dims.size(0), 4, P, dtype=X.dtype, device=X.device)
    left_dims_expand = left_dims.unsqueeze(-1).expand(left_dims.size(0), P)
    right_dims_expand = right_dims.unsqueeze(-1).expand(right_dims.size(0), P)
    for (dim, value) in zip(range(4), [1.0, 1.0, -1.0, -1.0]):
        probe[:, dim, :].scatter_(-1, left_dims_expand, value)
    for (dim, value) in zip(range(4), [1.0, -1.0, 1.0, -1.0]):
        probe[:, dim, :].scatter_(-1, right_dims_expand, value)
    kprobe = kappa * probe
    kprobe = kprobe.reshape(-1, P)
    k_probeX = kernel(kprobe, kX, eta1, eta2, c)
    k_prbprb = kernel(kprobe, kprobe, eta1, eta2, c)
    vec = torch.tensor([0.25, -0.25, -0.25, 0.25], dtype=X.dtype, device=X.device)
    mu = torch.matmul(k_probeX, torch.matmul(k_xx_inv, Y).unsqueeze(-1)).squeeze(-1).reshape(left_dims.size(0), 4)
    mu = (mu * vec).sum(-1)
    var = k_prbprb - torch.matmul(k_probeX, torch.matmul(k_xx_inv, k_probeX.t()))
    var = var.reshape(left_dims.size(0), 4, left_dims.size(0), 4).diagonal(dim1=-4, dim2=-2)
    std = ((var * vec.unsqueeze(-1)).sum(-2) * vec.unsqueeze(-1)).sum(-2).clamp(min=0.0).sqrt()
    active_quad_dims = ((mu - 4.0 * std > 0.0) | (mu + 4.0 * std < 0.0)) & (mu.abs() > 0.0001).bool()
    active_quad_dims = active_quad_dims.nonzero(as_tuple=False)
    active_quadratic_dims = np.stack([left_dims[active_quad_dims].data.numpy().flatten(), right_dims[active_quad_dims].data.numpy().flatten()], axis=1)
    active_quadratic_dims = np.split(active_quadratic_dims, active_quadratic_dims.shape[0])
    active_quadratic_dims = [tuple(a.tolist()[0]) for a in active_quadratic_dims]
    return (active_dims.data.numpy(), active_quadratic_dims)

def get_data(N=20, P=10, S=2, Q=2, sigma_obs=0.15):
    if False:
        print('Hello World!')
    assert S < P and P > 3 and (S > 2) and (Q > 1) and (Q <= S)
    torch.manual_seed(1)
    X = torch.randn(N, P)
    singleton_weights = 2.0 * torch.rand(S) - 1.0
    Y_mean = torch.einsum('ni,i->n', X[:, 0:S], singleton_weights)
    quadratic_weights = []
    expected_quad_dims = []
    for dim1 in range(Q):
        for dim2 in range(Q):
            if dim1 >= dim2:
                continue
            expected_quad_dims.append((dim1, dim2))
            quadratic_weights.append(2.0 * torch.rand(1) - 1.0)
            Y_mean += quadratic_weights[-1] * X[:, dim1] * X[:, dim2]
    quadratic_weights = torch.tensor(quadratic_weights)
    Y = Y_mean
    Y -= Y.mean()
    Y_std1 = Y.std()
    Y /= Y_std1
    Y += sigma_obs * torch.randn(N)
    Y -= Y.mean()
    Y_std2 = Y.std()
    Y /= Y_std2
    assert X.shape == (N, P)
    assert Y.shape == (N,)
    return (X, Y, singleton_weights / (Y_std1 * Y_std2), expected_quad_dims)

def init_loc_fn(site):
    if False:
        while True:
            i = 10
    value = init_to_median(site, num_samples=50)
    if site['name'] == 'sigma':
        value = 0.1 * value
    return value

def main(args):
    if False:
        i = 10
        return i + 15
    hypers = {'expected_sparsity': max(1.0, args.num_dimensions / 10), 'alpha1': 3.0, 'beta1': 1.0, 'alpha2': 3.0, 'beta2': 1.0, 'alpha3': 1.0, 'c': 1.0}
    P = args.num_dimensions
    S = args.active_dimensions
    Q = args.quadratic_dimensions
    (X, Y, expected_thetas, expected_quad_dims) = get_data(N=args.num_data, P=P, S=S, Q=Q, sigma_obs=args.sigma)
    loss_fn = Trace_ELBO().differentiable_loss
    init_losses = []
    for restart in range(args.num_restarts):
        pyro.clear_param_store()
        pyro.set_rng_seed(restart)
        guide = AutoDelta(model, init_loc_fn=init_loc_fn)
        with torch.no_grad():
            init_losses.append(loss_fn(model, guide, X, Y, hypers).item())
    pyro.set_rng_seed(np.argmin(init_losses))
    pyro.clear_param_store()
    guide = AutoDelta(model, init_loc_fn=init_loc_fn)
    with poutine.block(), poutine.trace(param_only=True) as param_capture:
        guide(X, Y, hypers)
    params = list([pyro.param(name).unconstrained() for name in param_capture.trace])
    adam = Adam(params, lr=args.lr)
    report_frequency = 50
    print('Beginning MAP optimization...')
    for step in range(args.num_steps):
        loss = loss_fn(model, guide, X, Y, hypers) / args.num_data
        loss.backward()
        adam.step()
        adam.zero_grad()
        if step in [100, 300, 700, 900]:
            adam.param_groups[0]['lr'] *= 0.2
        if step % report_frequency == 0 or step == args.num_steps - 1:
            print('[step %04d]  loss: %.5f' % (step, loss))
    print('Expected singleton thetas:\n', expected_thetas.data.numpy())
    median = guide.median()
    (active_dims, active_quad_dims) = compute_posterior_stats(X.double(), Y.double(), median['msq'].double(), median['lambda'].double(), median['eta1'].double(), median['xisq'].double(), torch.tensor(hypers['c']).double(), median['sigma'].double())
    expected_active_dims = np.arange(S).tolist()
    tp_singletons = len(set(active_dims) & set(expected_active_dims))
    fp_singletons = len(set(active_dims) - set(expected_active_dims))
    fn_singletons = len(set(expected_active_dims) - set(active_dims))
    singleton_stats = (tp_singletons, fp_singletons, fn_singletons)
    tp_quads = len(set(active_quad_dims) & set(expected_quad_dims))
    fp_quads = len(set(active_quad_dims) - set(expected_quad_dims))
    fn_quads = len(set(expected_quad_dims) - set(active_quad_dims))
    quad_stats = (tp_quads, fp_quads, fn_quads)
    print('[SUMMARY STATS]')
    print('Singletons (true positive, false positive, false negative): ' + '(%d, %d, %d)' % singleton_stats)
    print('Quadratic  (true positive, false positive, false negative): ' + '(%d, %d, %d)' % quad_stats)
if __name__ == '__main__':
    assert pyro.__version__.startswith('1.8.6')
    parser = argparse.ArgumentParser(description='Krylov KIT')
    parser.add_argument('--num-data', type=int, default=750)
    parser.add_argument('--num-steps', type=int, default=1000)
    parser.add_argument('--num-dimensions', type=int, default=100)
    parser.add_argument('--num-restarts', type=int, default=10)
    parser.add_argument('--sigma', type=float, default=0.05)
    parser.add_argument('--active-dimensions', type=int, default=10)
    parser.add_argument('--quadratic-dimensions', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.3)
    args = parser.parse_args()
    main(args)