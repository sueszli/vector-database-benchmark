import torch
from ._utils import _cast_as_tensor
from ._utils import _update_parameter
from ._utils import _check_parameter
from ._utils import _reshape_weights
from .distributions._distribution import Distribution

class BayesMixin(torch.nn.Module):

    def _reset_cache(self):
        if False:
            return 10
        'Reset the internally stored statistics.\n\n\t\tThis method is meant to only be called internally. It resets the\n\t\tstored statistics used to update the model parameters as well as\n\t\trecalculates the cached values meant to speed up log probability\n\t\tcalculations.\n\t\t'
        if self._initialized == False:
            return
        self.register_buffer('_w_sum', torch.zeros(self.k, device=self.device))
        self.register_buffer('_log_priors', torch.log(self.priors))

    def _emission_matrix(self, X, priors=None):
        if False:
            return 10
        'Return the emission/responsibility matrix.\n\n\t\tThis method returns the log probability of each example under each\n\t\tdistribution contained in the model with the log prior probability\n\t\tof each component added.\n\n\n\t\tParameters\n\t\t----------\n\t\tX: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)\n\t\t\tA set of examples to evaluate. \n\n\t\tpriors: list, numpy.ndarray, torch.Tensor, shape=(-1, self.k)\n\t\t\tPrior probabilities of assigning each symbol to each node. If not\n\t\t\tprovided, do not include in the calculations (conceptually\n\t\t\tequivalent to a uniform probability, but without scaling the\n\t\t\tprobabilities). This can be used to assign labels to observatons\n\t\t\tby setting one of the probabilities for an observation to 1.0.\n\t\t\tNote that this can be used to assign hard labels, but does not\n\t\t\thave the same semantics for soft labels, in that it only\n\t\t\tinfluences the initial estimate of an observation being generated\n\t\t\tby a component, not gives a target. Default is None.\n\n\t\n\t\tReturns\n\t\t-------\n\t\te: torch.Tensor, shape=(-1, self.k)\n\t\t\tA set of log probabilities for each example under each distribution.\n\t\t'
        X = _check_parameter(_cast_as_tensor(X), 'X', ndim=2, shape=(-1, self.d), check_parameter=self.check_data)
        priors = _check_parameter(_cast_as_tensor(priors), 'priors', ndim=2, shape=(X.shape[0], self.k), min_value=0.0, max_value=1.0, value_sum=1.0, value_sum_dim=-1, check_parameter=self.check_data)
        d = X.shape[0]
        e = torch.empty(d, self.k, device=self.device, dtype=self.dtype)
        for (i, d) in enumerate(self.distributions):
            e[:, i] = d.log_probability(X)
        if priors is not None:
            e += torch.log(priors)
        return e + self._log_priors

    def probability(self, X, priors=None):
        if False:
            print('Hello World!')
        'Calculate the probability of each example.\n\n\t\tThis method calculates the probability of each example given the\n\t\tparameters of the distribution. The examples must be given in a 2D\n\t\tformat.\n\n\t\tNote: This differs from some other probability calculation\n\t\tfunctions, like those in torch.distributions, because it is not\n\t\treturning the probability of each feature independently, but rather\n\t\tthe total probability of the entire example.\n\n\n\t\tParameters\n\t\t----------\n\t\tX: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)\n\t\t\tA set of examples to evaluate.\n\n\t\tpriors: list, numpy.ndarray, torch.Tensor, shape=(-1, self.k)\n\t\t\tPrior probabilities of assigning each symbol to each node. If not\n\t\t\tprovided, do not include in the calculations (conceptually\n\t\t\tequivalent to a uniform probability, but without scaling the\n\t\t\tprobabilities). This can be used to assign labels to observatons\n\t\t\tby setting one of the probabilities for an observation to 1.0.\n\t\t\tNote that this can be used to assign hard labels, but does not\n\t\t\thave the same semantics for soft labels, in that it only\n\t\t\tinfluences the initial estimate of an observation being generated\n\t\t\tby a component, not gives a target. Default is None.\n\n\n\t\tReturns\n\t\t-------\n\t\tprob: torch.Tensor, shape=(-1,)\n\t\t\tThe probability of each example.\n\t\t'
        return torch.exp(self.log_probability(X, priors=priors))

    def log_probability(self, X, priors=None):
        if False:
            return 10
        'Calculate the log probability of each example.\n\n\t\tThis method calculates the log probability of each example given the\n\t\tparameters of the distribution. The examples must be given in a 2D\n\t\tformat. For a Bernoulli distribution, each entry in the data must\n\t\tbe either 0 or 1.\n\n\t\tNote: This differs from some other log probability calculation\n\t\tfunctions, like those in torch.distributions, because it is not\n\t\treturning the log probability of each feature independently, but rather\n\t\tthe total log probability of the entire example.\n\n\n\t\tParameters\n\t\t----------\n\t\tX: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)\n\t\t\tA set of examples to evaluate.\n\n\t\tpriors: list, numpy.ndarray, torch.Tensor, shape=(-1, self.k)\n\t\t\tPrior probabilities of assigning each symbol to each node. If not\n\t\t\tprovided, do not include in the calculations (conceptually\n\t\t\tequivalent to a uniform probability, but without scaling the\n\t\t\tprobabilities). This can be used to assign labels to observatons\n\t\t\tby setting one of the probabilities for an observation to 1.0.\n\t\t\tNote that this can be used to assign hard labels, but does not\n\t\t\thave the same semantics for soft labels, in that it only\n\t\t\tinfluences the initial estimate of an observation being generated\n\t\t\tby a component, not gives a target. Default is None.\n\n\n\t\tReturns\n\t\t-------\n\t\tlogp: torch.Tensor, shape=(-1,)\n\t\t\tThe log probability of each example.\n\t\t'
        e = self._emission_matrix(X, priors=priors)
        return torch.logsumexp(e, dim=1)

    def predict(self, X, priors=None):
        if False:
            while True:
                i = 10
        'Calculate the label assignment for each example.\n\n\t\tThis method calculates the label for each example as the most likely\n\t\tcomponent after factoring in the prior probability.\n\n\n\t\tParameters\n\t\t----------\n\t\tX: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)\n\t\t\tA set of examples to summarize.\n\n\t\tpriors: list, numpy.ndarray, torch.Tensor, shape=(-1, self.k)\n\t\t\tPrior probabilities of assigning each symbol to each node. If not\n\t\t\tprovided, do not include in the calculations (conceptually\n\t\t\tequivalent to a uniform probability, but without scaling the\n\t\t\tprobabilities). This can be used to assign labels to observatons\n\t\t\tby setting one of the probabilities for an observation to 1.0.\n\t\t\tNote that this can be used to assign hard labels, but does not\n\t\t\thave the same semantics for soft labels, in that it only\n\t\t\tinfluences the initial estimate of an observation being generated\n\t\t\tby a component, not gives a target. Default is None.\n\n\n\t\tReturns\n\t\t-------\n\t\ty: torch.Tensor, shape=(-1,)\n\t\t\tThe predicted label for each example.\n\t\t'
        e = self._emission_matrix(X, priors=priors)
        return torch.argmax(e, dim=1)

    def predict_proba(self, X, priors=None):
        if False:
            i = 10
            return i + 15
        'Calculate the posterior probabilities for each example.\n\n\t\tThis method calculates the posterior probabilities for each example\n\t\tunder each component of the model after factoring in the prior \n\t\tprobability and normalizing across all the components.\n\n\n\t\tParameters\n\t\t----------\n\t\tX: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)\n\t\t\tA set of examples to summarize.\n\n\t\tpriors: list, numpy.ndarray, torch.Tensor, shape=(-1, self.k)\n\t\t\tPrior probabilities of assigning each symbol to each node. If not\n\t\t\tprovided, do not include in the calculations (conceptually\n\t\t\tequivalent to a uniform probability, but without scaling the\n\t\t\tprobabilities). This can be used to assign labels to observatons\n\t\t\tby setting one of the probabilities for an observation to 1.0.\n\t\t\tNote that this can be used to assign hard labels, but does not\n\t\t\thave the same semantics for soft labels, in that it only\n\t\t\tinfluences the initial estimate of an observation being generated\n\t\t\tby a component, not gives a target. Default is None.\n\n\n\t\tReturns\n\t\t-------\n\t\ty: torch.Tensor, shape=(-1, self.k)\n\t\t\tThe posterior probabilities for each example under each component.\n\t\t'
        e = self._emission_matrix(X, priors=priors)
        return torch.exp(e - torch.logsumexp(e, dim=1, keepdims=True))

    def predict_log_proba(self, X, priors=None):
        if False:
            print('Hello World!')
        'Calculate the log posterior probabilities for each example.\n\n\t\tThis method calculates the log posterior probabilities for each example\n\t\tunder each component of the model after factoring in the prior \n\t\tprobability and normalizing across all the components.\n\n\n\t\tParameters\n\t\t----------\n\t\tX: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)\n\t\t\tA set of examples to summarize.\n\n\t\tpriors: list, numpy.ndarray, torch.Tensor, shape=(-1, self.k)\n\t\t\tPrior probabilities of assigning each symbol to each node. If not\n\t\t\tprovided, do not include in the calculations (conceptually\n\t\t\tequivalent to a uniform probability, but without scaling the\n\t\t\tprobabilities). This can be used to assign labels to observatons\n\t\t\tby setting one of the probabilities for an observation to 1.0.\n\t\t\tNote that this can be used to assign hard labels, but does not\n\t\t\thave the same semantics for soft labels, in that it only\n\t\t\tinfluences the initial estimate of an observation being generated\n\t\t\tby a component, not gives a target. Default is None.\n\n\n\t\tReturns\n\t\t-------\n\t\ty: torch.Tensor, shape=(-1, self.k)\n\t\t\tThe log posterior probabilities for each example under each \n\t\t\tcomponent.\n\t\t'
        e = self._emission_matrix(X, priors=priors)
        return e - torch.logsumexp(e, dim=1, keepdims=True)

    def from_summaries(self):
        if False:
            while True:
                i = 10
        'Update the model parameters given the extracted statistics.\n\n\t\tThis method uses calculated statistics from calls to the `summarize`\n\t\tmethod to update the distribution parameters. Hyperparameters for the\n\t\tupdate are passed in at initialization time.\n\n\t\tNote: Internally, a call to `fit` is just a successive call to the\n\t\t`summarize` method followed by the `from_summaries` method.\n\t\t'
        for d in self.distributions:
            d.from_summaries()
        if self.frozen == True:
            return
        priors = self._w_sum / torch.sum(self._w_sum)
        _update_parameter(self.priors, priors, self.inertia)
        self._reset_cache()