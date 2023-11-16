import torch
import torch.autograd as autograd
import torch.optim as optim
from torch.distributions import transform_to
import pyro.contrib.gp as gp
import pyro.optim
from pyro.infer import TraceEnum_ELBO

class GPBayesOptimizer(pyro.optim.multi.MultiOptimizer):
    """Performs Bayesian Optimization using a Gaussian Process as an
    emulator for the unknown function.
    """

    def __init__(self, constraints, gpmodel, num_acquisitions, acquisition_func=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param torch.constraint constraints: constraints defining the domain of `f`\n        :param gp.models.GPRegression gpmodel: a (possibly initialized) GP\n            regression model. The kernel, etc is specified via `gpmodel`.\n        :param int num_acquisitions: number of points to acquire at each step\n        :param function acquisition_func: a function to generate acquisitions.\n            It should return a torch.Tensor of new points to query.\n        '
        if acquisition_func is None:
            acquisition_func = self.acquire_thompson
        self.constraints = constraints
        self.gpmodel = gpmodel
        self.num_acquisitions = num_acquisitions
        self.acquisition_func = acquisition_func

    def update_posterior(self, X, y):
        if False:
            while True:
                i = 10
        X = torch.cat([self.gpmodel.X, X])
        y = torch.cat([self.gpmodel.y, y])
        self.gpmodel.set_data(X, y)
        optimizer = torch.optim.Adam(self.gpmodel.parameters(), lr=0.001)
        gp.util.train(self.gpmodel, optimizer, loss_fn=TraceEnum_ELBO(strict_enumeration_warning=False).differentiable_loss, retain_graph=True)

    def find_a_candidate(self, differentiable, x_init):
        if False:
            return 10
        'Given a starting point, `x_init`, takes one LBFGS step\n        to optimize the differentiable function.\n\n        :param function differentiable: a function amenable to torch\n            autograd\n        :param torch.Tensor x_init: the initial point\n\n        '
        unconstrained_x_init = transform_to(self.constraints).inv(x_init)
        unconstrained_x = unconstrained_x_init.detach().clone().requires_grad_(True)
        minimizer = optim.LBFGS([unconstrained_x], max_eval=20)

        def closure():
            if False:
                i = 10
                return i + 15
            minimizer.zero_grad()
            if (torch.log(torch.abs(unconstrained_x)) > 25.0).any():
                return torch.tensor(float('inf'))
            x = transform_to(self.constraints)(unconstrained_x)
            y = differentiable(x)
            autograd.backward(unconstrained_x, autograd.grad(y, unconstrained_x, retain_graph=True))
            return y
        minimizer.step(closure)
        x = transform_to(self.constraints)(unconstrained_x)
        opt_y = differentiable(x)
        return (x.detach(), opt_y.detach())

    def opt_differentiable(self, differentiable, num_candidates=5):
        if False:
            for i in range(10):
                print('nop')
        'Optimizes a differentiable function by choosing `num_candidates`\n        initial points at random and calling :func:`find_a_candidate` on\n        each. The best candidate is returned with its function value.\n\n        :param function differentiable: a function amenable to torch autograd\n        :param int num_candidates: the number of random starting points to\n            use\n        :return: the minimiser and its function value\n        :rtype: tuple\n        '
        candidates = []
        values = []
        for j in range(num_candidates):
            x_init = torch.empty(1, dtype=self.gpmodel.X.dtype, device=self.gpmodel.X.device).uniform_(self.constraints.lower_bound, self.constraints.upper_bound)
            (x, y) = self.find_a_candidate(differentiable, x_init)
            if torch.isnan(y):
                continue
            candidates.append(x)
            values.append(y)
        (mvalue, argmin) = torch.min(torch.cat(values), dim=0)
        return (candidates[argmin.item()], mvalue)

    def acquire_thompson(self, num_acquisitions=1, **opt_params):
        if False:
            while True:
                i = 10
        'Selects `num_acquisitions` query points at which to query the\n        original function by Thompson sampling.\n\n        :param int num_acquisitions: the number of points to generate\n        :param dict opt_params: additional parameters for optimization\n            routines\n        :return: a tensor of points to evaluate `loss` at\n        :rtype: torch.Tensor\n        '
        X = self.gpmodel.X
        X = torch.empty(num_acquisitions, *X.shape[1:], dtype=X.dtype, device=X.device)
        for i in range(num_acquisitions):
            sampler = self.gpmodel.iter_sample(noiseless=False)
            (x, _) = self.opt_differentiable(sampler, **opt_params)
            X[i, ...] = x
        return X

    def get_step(self, loss, params, verbose=False):
        if False:
            return 10
        X = self.acquisition_func(num_acquisitions=self.num_acquisitions)
        y = loss(X)
        if verbose:
            print('Acquire at: X')
            print(X)
            print('y')
            print(y)
        self.update_posterior(X, y)
        return self.opt_differentiable(lambda x: self.gpmodel(x)[0])