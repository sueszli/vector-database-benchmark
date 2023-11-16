import math
import numpy
import torch
from .._utils import _cast_as_tensor
from .._utils import _cast_as_parameter
from .._utils import _update_parameter
from .._utils import _check_parameter
from ..distributions._distribution import Distribution
from ._base import _BaseHMM
from ._base import _check_inputs
NEGINF = float('-inf')
inf = float('inf')

class DenseHMM(_BaseHMM):
    """A hidden Markov model with a dense transition matrix.

	A hidden Markov model is an extension of a mixture model to sequences by
	including a transition matrix between the elements of the mixture. Each of
	the algorithms for a hidden Markov model are essentially just a revision
	of those algorithms to incorporate this transition matrix.

	This object is a wrapper for a hidden Markov model with a dense transition
	matrix.

	This object is a wrapper for both implementations, which can be specified
	using the `kind` parameter. Choosing the right implementation will not
	effect the accuracy of the results but will change the speed at which they
	are calculated. 	

	Separately, there are two ways to instantiate the hidden Markov model. The
	first is by passing in a set of distributions, a dense transition matrix, 
	and optionally start/end probabilities. The second is to initialize the
	object without these and then to add edges using the `add_edge` method
	and to add distributions using the `add_distributions` method. Importantly, the way that
	you choose to initialize the hidden Markov model is independent of the
	implementation that you end up choosing. If you pass in a dense transition
	matrix, this will be converted to a sparse matrix with all the zeros
	dropped if you choose `kind='sparse'`.


	Parameters
	----------
	distributions: tuple or list
		A set of distribution objects. These objects do not need to be
		initialized, i.e., can be "Normal()". 

	edges: numpy.ndarray, torch.Tensor, or None. shape=(k,k), optional
		A dense transition matrix of probabilities for how each node or
		distribution passed in connects to each other one. This can contain
		many zeroes, and when paired with `kind='sparse'`, will drop those
		elements from the matrix.

	starts: list, numpy.ndarray, torch.Tensor, or None. shape=(k,), optional
		The probability of starting at each node. If not provided, assumes
		these probabilities are uniform.

	ends: list, numpy.ndarray, torch.Tensor, or None. shape=(k,), optional
		The probability of ending at each node. If not provided, assumes
		these probabilities are uniform.

	inertia: float, [0, 1], optional
		Indicates the proportion of the update to apply to the parameters
		during training. When the inertia is 0.0, the update is applied in
		its entirety and the previous parameters are ignored. When the
		inertia is 1.0, the update is entirely ignored and the previous
		parameters are kept, equivalently to if the parameters were frozen.

	frozen: bool, optional
		Whether all the parameters associated with this distribution are frozen.
		If you want to freeze individual pameters, or individual values in those
		parameters, you must modify the `frozen` attribute of the tensor or
		parameter directly. Default is False.

	check_data: bool, optional
		Whether to check properties of the data and potentially recast it to
		torch.tensors. This does not prevent checking of parameters but can
		slightly speed up computation when you know that your inputs are valid.
		Setting this to False is also necessary for compiling. Default is True.
	"""

    def __init__(self, distributions=None, edges=None, starts=None, ends=None, init='random', max_iter=1000, tol=0.1, sample_length=None, return_sample_paths=False, inertia=0.0, frozen=False, check_data=True, random_state=None, verbose=False):
        if False:
            return 10
        super().__init__(distributions=distributions, starts=starts, ends=ends, init=init, max_iter=max_iter, tol=tol, sample_length=sample_length, return_sample_paths=return_sample_paths, inertia=inertia, frozen=frozen, check_data=check_data, random_state=random_state, verbose=verbose)
        self.name = 'DenseHMM'
        n = len(distributions) if distributions is not None else 0
        if edges is not None:
            self.edges = _cast_as_parameter(torch.log(_check_parameter(_cast_as_tensor(edges), 'edges', ndim=2, shape=(n, n), min_value=0.0, max_value=1.0)))
        self._initialized = self.distributions is not None and self.starts is not None and (self.ends is not None) and (self.edges is not None) and all((d._initialized for d in self.distributions))
        if self._initialized:
            self.distributions = torch.nn.ModuleList(self.distributions)
        self._reset_cache()

    def _reset_cache(self):
        if False:
            for i in range(10):
                print('nop')
        'Reset the internally stored statistics.\n\n\t\tThis method is meant to only be called internally. It resets the\n\t\tstored statistics used to update the model parameters as well as\n\t\trecalculates the cached values meant to speed up log probability\n\t\tcalculations.\n\t\t'
        if self._initialized == False:
            return
        for node in self.distributions:
            node._reset_cache()
        self.register_buffer('_xw_sum', torch.zeros(self.n_distributions, self.n_distributions, dtype=self.dtype, requires_grad=False, device=self.device))
        self.register_buffer('_xw_starts_sum', torch.zeros(self.n_distributions, dtype=self.dtype, requires_grad=False, device=self.device))
        self.register_buffer('_xw_ends_sum', torch.zeros(self.n_distributions, dtype=self.dtype, requires_grad=False, device=self.device))

    def _initialize(self, X=None, sample_weight=None):
        if False:
            return 10
        'Initialize the probability distribution.\n\n\t\tThis method is meant to only be called internally. It initializes the\n\t\tparameters of the distribution and stores its dimensionality. For more\n\t\tcomplex methods, this function will do more.\n\n\n\t\tParameters\n\t\t----------\n\t\tX: list, numpy.ndarray, torch.Tensor, shape=(-1, len, self.d), optional\n\t\t\tThe data to use to initialize the model. Default is None.\n\n\t\tsample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional\n\t\t\tA set of weights for the examples. This can be either of shape\n\t\t\t(-1, len) or a vector of shape (-1,). If None, defaults to ones.\n\t\t\tDefault is None.\n\t\t'
        super()._initialize(X, sample_weight=sample_weight)
        n = self.n_distributions
        if self.edges == None:
            self.edges = _cast_as_parameter(torch.log(torch.ones(n, n, dtype=self.dtype, device=self.device) / n))
        self.distributions = torch.nn.ModuleList(self.distributions)

    def add_edge(self, start, end, prob):
        if False:
            print('Hello World!')
        'Add an edge to the model.\n\n\t\tThis method will fill in an entry in the dense transition matrix\n\t\tat row indexed by the start distribution and the column indexed\n\t\tby the end distribution. The value that will be included is the\n\t\tlog of the probability value provided. Note that this will override\n\t\tvalues that already exist, and that this will initialize a new\n\t\tdense transition matrix if none has been passed in so far.\n\n\n\t\tParameters\n\t\t----------\n\t\tstart: torch.distributions.distribution\n\t\t\tThe distribution that the edge starts at.\n\n\t\tend: torch.distributions.distribution\n\t\t\tThe distribution that the edge ends at.\n\n\t\tprob: float, (0.0, 1.0]\n\t\t\tThe probability of that edge.\n\t\t'
        if self.distributions is None:
            raise ValueError('Must add distributions before edges.')
        n = self.n_distributions
        if start == self.start:
            if self.starts is None:
                self.starts = torch.empty(n, dtype=self.dtype, device=self.device) - inf
            idx = self.distributions.index(end)
            self.starts[idx] = math.log(prob)
        elif end == self.end:
            if self.ends is None:
                self.ends = torch.empty(n, dtype=self.dtype, device=self.device) - inf
            idx = self.distributions.index(start)
            self.ends[idx] = math.log(prob)
        else:
            if self.edges is None:
                self.edges = torch.empty((n, n), dtype=self.dtype, device=self.device) - inf
            idx1 = self.distributions.index(start)
            idx2 = self.distributions.index(end)
            self.edges[idx1, idx2] = math.log(prob)

    def sample(self, n):
        if False:
            for i in range(10):
                print('nop')
        'Sample from the probability distribution.\n\n\t\tThis method will return `n` samples generated from the underlying\n\t\tprobability distribution. Because a HMM describes variable length\n\t\tsequences, a list will be returned where each element is one of\n\t\tthe generated sequences.\n\n\n\t\tParameters\n\t\t----------\n\t\tn: int\n\t\t\tThe number of samples to generate.\n\t\t\n\n\t\tReturns\n\t\t-------\n\t\tX: list of torch.tensor, shape=(n,)\n\t\t\tA list of randomly generated samples, where each sample of\n\t\t\tsize (length, self.d).\n\t\t'
        if self.sample_length is None and self.ends is None:
            raise ValueError('Must specify a length or have explicit ' + 'end probabilities.')
        (distributions, emissions) = ([], [])
        edge_probs = torch.hstack([self.edges, self.ends.unsqueeze(1)])
        edge_probs = torch.exp(edge_probs).numpy()
        starts = torch.exp(self.starts).numpy()
        for _ in range(n):
            node_i = self.random_state.choice(self.n_distributions, p=starts)
            emission_i = self.distributions[node_i].sample(n=1)
            (distributions_, emissions_) = ([node_i], [emission_i])
            for i in range(1, self.sample_length or int(100000000.0)):
                node_i = self.random_state.choice(self.n_distributions + 1, p=edge_probs[node_i])
                if node_i == self.n_distributions:
                    break
                emission_i = self.distributions[node_i].sample(n=1)
                distributions_.append(node_i)
                emissions_.append(emission_i)
            distributions.append(distributions_)
            emissions.append(torch.vstack(emissions_))
        if self.return_sample_paths == True:
            return (emissions, distributions)
        return emissions

    def forward(self, X=None, emissions=None, priors=None):
        if False:
            return 10
        'Run the forward algorithm on some data.\n\n\t\tRuns the forward algorithm on a batch of sequences. This is not to be\n\t\tconfused with a "forward pass" when talking about neural networks. The\n\t\tforward algorithm is a dynamic programming algorithm that begins at the\n\t\tstart state and returns the probability, over all paths through the\n\t\tmodel, that result in the alignment of symbol i to node j.\n\n\t\tNote that, as an internal method, this does not take as input the\n\t\tactual sequence of observations but, rather, the emission probabilities\n\t\tcalculated from the sequence given the model.\n\n\t\t\n\t\tParameters\n\t\t----------\n\t\tX: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, d)\n\t\t\tA set of examples to evaluate. Does not need to be passed in if\n\t\t\temissions are. \n\n\t\temissions: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, n_dists)\n\t\t\tPrecalculated emission log probabilities. These are the\n\t\t\tprobabilities of each observation under each probability \n\t\t\tdistribution. When running some algorithms it is more efficient\n\t\t\tto precalculate these and pass them into each call.\n\n\t\tpriors: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, self.k)\n\t\t\tPrior probabilities of assigning each symbol to each node. If not\n\t\t\tprovided, do not include in the calculations (conceptually\n\t\t\tequivalent to a uniform probability, but without scaling the\n\t\t\tprobabilities). This can be used to assign labels to observatons\n\t\t\tby setting one of the probabilities for an observation to 1.0.\n\t\t\tNote that this can be used to assign hard labels, but does not\n\t\t\thave the same semantics for soft labels, in that it only\n\t\t\tinfluences the initial estimate of an observation being generated\n\t\t\tby a component, not gives a target. Default is None.\n\n\n\t\tReturns\n\t\t-------\n\t\tf: torch.Tensor, shape=(-1, -1, self.n_distributions)\n\t\t\tThe log probabilities calculated by the forward algorithm.\n\t\t'
        emissions = _check_inputs(self, X, emissions, priors)
        l = emissions.shape[1]
        t_max = self.edges.max()
        t = torch.exp(self.edges - t_max)
        f = torch.clone(emissions.permute(1, 0, 2)).contiguous()
        f[0] += self.starts
        f[1:] += t_max
        for i in range(1, l):
            p_max = torch.max(f[i - 1], dim=1, keepdims=True).values
            p = torch.exp(f[i - 1] - p_max)
            f[i] += torch.log(torch.matmul(p, t)) + p_max
        f = f.permute(1, 0, 2)
        return f

    def backward(self, X=None, emissions=None, priors=None):
        if False:
            i = 10
            return i + 15
        'Run the backward algorithm on some data.\n\n\t\tRuns the backward algorithm on a batch of sequences. This is not to be\n\t\tconfused with a "backward pass" when talking about neural networks. The\n\t\tbackward algorithm is a dynamic programming algorithm that begins at end\n\t\tof the sequence and returns the probability, over all paths through the\n\t\tmodel, that result in the alignment of symbol i to node j, working\n\t\tbackwards.\n\n\t\tNote that, as an internal method, this does not take as input the\n\t\tactual sequence of observations but, rather, the emission probabilities\n\t\tcalculated from the sequence given the model.\n\n\t\t\n\t\tParameters\n\t\t----------\n\t\tX: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, d)\n\t\t\tA set of examples to evaluate. Does not need to be passed in if\n\t\t\temissions are. \n\n\t\temissions: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, n_distributions)\n\t\t\tPrecalculated emission log probabilities. These are the\n\t\t\tprobabilities of each observation under each probability \n\t\t\tdistribution. When running some algorithms it is more efficient\n\t\t\tto precalculate these and pass them into each call.\n\n\t\tpriors: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, self.k)\n\t\t\tPrior probabilities of assigning each symbol to each node. If not\n\t\t\tprovided, do not include in the calculations (conceptually\n\t\t\tequivalent to a uniform probability, but without scaling the\n\t\t\tprobabilities). This can be used to assign labels to observatons\n\t\t\tby setting one of the probabilities for an observation to 1.0.\n\t\t\tNote that this can be used to assign hard labels, but does not\n\t\t\thave the same semantics for soft labels, in that it only\n\t\t\tinfluences the initial estimate of an observation being generated\n\t\t\tby a component, not gives a target. Default is None.\n\n\n\t\tReturns\n\t\t-------\n\t\tb: torch.Tensor, shape=(-1, length, self.n_distributions)\n\t\t\tThe log probabilities calculated by the backward algorithm.\n\t\t'
        emissions = _check_inputs(self, X, emissions, priors)
        (n, l, _) = emissions.shape
        b = torch.zeros(l, n, self.n_distributions, dtype=self.dtype, device=self.device) + float('-inf')
        b[-1] = self.ends
        t_max = self.edges.max()
        t = torch.exp(self.edges.T - t_max)
        for i in range(l - 2, -1, -1):
            p = b[i + 1] + emissions[:, i + 1]
            p_max = torch.max(p, dim=1, keepdims=True).values
            p = torch.exp(p - p_max)
            b[i] = torch.log(torch.matmul(p, t)) + t_max + p_max
        b = b.permute(1, 0, 2)
        return b

    def forward_backward(self, X=None, emissions=None, priors=None):
        if False:
            return 10
        'Run the forward-backward algorithm on some data.\n\n\t\tRuns the forward-backward algorithm on a batch of sequences. This\n\t\talgorithm combines the best of the forward and the backward algorithm.\n\t\tIt combines the probability of starting at the beginning of the sequence\n\t\tand working your way to each observation with the probability of\n\t\tstarting at the end of the sequence and working your way backward to it.\n\n\t\tA number of statistics can be calculated using this information. These\n\t\tstatistics are powerful inference tools but are also used during the\n\t\tBaum-Welch training process. \n\n\t\t\n\t\tParameters\n\t\t----------\n\t\tX: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, d)\n\t\t\tA set of examples to evaluate. Does not need to be passed in if\n\t\t\temissions are. \n\n\t\temissions: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, n_distributions)\n\t\t\tPrecalculated emission log probabilities. These are the\n\t\t\tprobabilities of each observation under each probability \n\t\t\tdistribution. When running some algorithms it is more efficient\n\t\t\tto precalculate these and pass them into each call.\n\n\t\tpriors: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, self.k)\n\t\t\tPrior probabilities of assigning each symbol to each node. If not\n\t\t\tprovided, do not include in the calculations (conceptually\n\t\t\tequivalent to a uniform probability, but without scaling the\n\t\t\tprobabilities). This can be used to assign labels to observatons\n\t\t\tby setting one of the probabilities for an observation to 1.0.\n\t\t\tNote that this can be used to assign hard labels, but does not\n\t\t\thave the same semantics for soft labels, in that it only\n\t\t\tinfluences the initial estimate of an observation being generated\n\t\t\tby a component, not gives a target. Default is None.\n\n\n\t\tReturns\n\t\t-------\n\t\ttransitions: torch.Tensor, shape=(-1, n, n)\n\t\t\tThe expected number of transitions across each edge that occur\n\t\t\tfor each example. The returned transitions follow the structure\n\t\t\tof the transition matrix and so will be dense or sparse as\n\t\t\tappropriate.\n\n\t\tresponsibility: torch.Tensor, shape=(-1, -1, n)\n\t\t\tThe posterior probabilities of each observation belonging to each\n\t\t\tstate given that one starts at the beginning of the sequence,\n\t\t\taligns observations across all paths to get to the current\n\t\t\tobservation, and then proceeds to align all remaining observations\n\t\t\tuntil the end of the sequence.\n\n\t\tstarts: torch.Tensor, shape=(-1, n)\n\t\t\tThe probabilities of starting at each node given the \n\t\t\tforward-backward algorithm.\n\n\t\tends: torch.Tensor, shape=(-1, n)\n\t\t\tThe probabilities of ending at each node given the forward-backward\n\t\t\talgorithm.\n\n\t\tlogp: torch.Tensor, shape=(-1,)\n\t\t\tThe log probabilities of each sequence given the model.\n\t\t'
        emissions = _check_inputs(self, X, emissions, priors)
        (n, l, _) = emissions.shape
        f = self.forward(emissions=emissions)
        b = self.backward(emissions=emissions)
        logp = torch.logsumexp(f[:, -1] + self.ends, dim=1)
        f_ = f[:, :-1].unsqueeze(-1)
        b_ = (b[:, 1:] + emissions[:, 1:]).unsqueeze(-2)
        t = f_ + b_ + self.edges.unsqueeze(0).unsqueeze(0)
        t = t.reshape(n, l - 1, -1)
        t = torch.exp(torch.logsumexp(t, dim=1).T - logp).T
        t = t.reshape(n, int(t.shape[1] ** 0.5), -1)
        starts = self.starts + emissions[:, 0] + b[:, 0]
        starts = torch.exp(starts.T - torch.logsumexp(starts, dim=-1)).T
        ends = self.ends + f[:, -1]
        ends = torch.exp(ends.T - torch.logsumexp(ends, dim=-1)).T
        r = f + b
        r = r - torch.logsumexp(r, dim=2).reshape(n, -1, 1)
        return (t, r, starts, ends, logp)

    def summarize(self, X, sample_weight=None, emissions=None, priors=None):
        if False:
            return 10
        'Extract the sufficient statistics from a batch of data.\n\n\t\tThis method calculates the sufficient statistics from optionally\n\t\tweighted data and adds them to the stored cache. The examples must be\n\t\tgiven in a 2D format. Sample weights can either be provided as one\n\t\tvalue per example or as a 2D matrix of weights for each feature in\n\t\teach example.\n\n\n\t\tParameters\n\t\t----------\n\t\tX: torch.Tensor, shape=(-1, -1, self.d)\n\t\t\tA set of examples to summarize.\n\n\t\ty: torch.Tensor, shape=(-1, -1), optional \n\t\t\tA set of labels with the same number of examples and length as the\n\t\t\tobservations that indicate which node in the model that each\n\t\t\tobservation should be assigned to. Passing this in means that the\n\t\t\tmodel uses labeled training instead of Baum-Welch. Default is None.\n\n\t\tsample_weight: torch.Tensor, optional\n\t\t\tA set of weights for the examples. This can be either of shape\n\t\t\t(-1, self.d) or a vector of shape (-1,). Default is ones.\n\n\t\temissions: torch.Tensor, shape=(-1, -1, self.n_distributions)\n\t\t\tPrecalculated emission log probabilities. These are the\n\t\t\tprobabilities of each observation under each probability \n\t\t\tdistribution. When running some algorithms it is more efficient\n\t\t\tto precalculate these and pass them into each call.\t\n\n\t\tpriors: list, numpy.ndarray, torch.Tensor, shape=(-1, -1, self.k)\n\t\t\tPrior probabilities of assigning each symbol to each node. If not\n\t\t\tprovided, do not include in the calculations (conceptually\n\t\t\tequivalent to a uniform probability, but without scaling the\n\t\t\tprobabilities). This can be used to assign labels to observatons\n\t\t\tby setting one of the probabilities for an observation to 1.0.\n\t\t\tNote that this can be used to assign hard labels, but does not\n\t\t\thave the same semantics for soft labels, in that it only\n\t\t\tinfluences the initial estimate of an observation being generated\n\t\t\tby a component, not gives a target. Default is None.\n\t\t'
        (X, emissions, sample_weight) = super().summarize(X, sample_weight=sample_weight, emissions=emissions, priors=priors)
        (t, r, starts, ends, logps) = self.forward_backward(emissions=emissions)
        self._xw_starts_sum += torch.sum(starts * sample_weight, dim=0)
        self._xw_ends_sum += torch.sum(ends * sample_weight, dim=0)
        self._xw_sum += torch.sum(t * sample_weight.unsqueeze(-1), dim=0)
        X = X.reshape(-1, X.shape[-1])
        r = torch.exp(r) * sample_weight.unsqueeze(-1)
        for (i, node) in enumerate(self.distributions):
            w = r[:, :, i].reshape(-1, 1)
            node.summarize(X, sample_weight=w)
        return logps

    def from_summaries(self):
        if False:
            return 10
        'Update the model parameters given the extracted statistics.\n\n\t\tThis method uses calculated statistics from calls to the `summarize`\n\t\tmethod to update the distribution parameters. Hyperparameters for the\n\t\tupdate are passed in at initialization time.\n\n\t\tNote: Internally, a call to `fit` is just a successive call to the\n\t\t`summarize` method followed by the `from_summaries` method.\n\t\t'
        for node in self.distributions:
            node.from_summaries()
        if self.frozen:
            return
        node_out_count = torch.sum(self._xw_sum, dim=1, keepdims=True)
        node_out_count += self._xw_ends_sum.unsqueeze(1)
        ends = torch.log(self._xw_ends_sum / node_out_count[:, 0])
        starts = torch.log(self._xw_starts_sum / self._xw_starts_sum.sum())
        edges = torch.log(self._xw_sum / node_out_count)
        _update_parameter(self.ends, ends, inertia=self.inertia)
        _update_parameter(self.starts, starts, inertia=self.inertia)
        _update_parameter(self.edges, edges, inertia=self.inertia)
        self._reset_cache()