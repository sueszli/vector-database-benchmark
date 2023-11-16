import time
import numpy
import torch
import itertools
import networkx as nx
from ._utils import _cast_as_tensor
from ._utils import _cast_as_parameter
from ._utils import _update_parameter
from ._utils import _check_parameter
from ._utils import _reshape_weights
from .distributions._distribution import Distribution
from .distributions import Categorical
from .distributions import JointCategorical
from .distributions import ConditionalCategorical
from .factor_graph import FactorGraph

class BayesianNetwork(Distribution):
    """A Bayesian network object.

	A Bayesian network is a probability distribution where dependencies between
	variables are explicitly encoded in a graph structure and the lack of an
	edge represents a conditional independence. These graphs are directed and
	typically must be acyclic, but this implementation allows for the networks
	to be cyclic as long as there is no assumption of convergence during
	inference.

	Inference is doing using loopy belief propogation along a factor graph
	representation. This is sometimes called the `sum-product` algorithm.
	It will yield exact results if the graph has a tree-like structure.
	Otherwise, if the graph is acyclic, it is guaranteed to converge but not
	necessarily to optimal results. If the graph is cyclic, there is no
	guarantee on convergence, but it is thought that the longer the loop is the 
	more likely one will get good results.

	Structure learning can be done using a variety of methods. 


	Parameters
	----------
	distributions: tuple or list or None
		A set of distribution objects. These do not need to be initialized,
		i.e. can be "Categorical()". Currently, they must be either Categorical
		or JointCategorical distributions. If provided, they must be consistent
		with the provided edges in that every conditional distribution must
		have at least one parent in the provided structure. Default is None.

	edges: tuple or list or None, optional
		A list or tuple of 2-tuples where the first element in the 2-tuple is 
		the parent distribution object and the second element is the child
		distribution object. If None, then no edges. Default is None.

	struture: tuple or list or None, optional
		A list or tuple of the parents for each distribution with a tuple
		containing no elements indicating a root node. For instance, 
		((), (0,), (), (0, 2)) would represent a graph with four nodes, 
		where the second distribution has the first distribution as a parent 
		and the fourth distribution has the first and third distributions as 
		parents. Use this only when you want new distribution objects to be
		created and fit when using the `fit` method. Default is None.

	max_iter: int, optional
		The number of iterations to do in the inference step. Default is 10.

	tol: float, optional
		The threshold at which to stop during fitting when the improvement
		goes under. Default is 1e-6.

	inertia: float, [0, 1], optional
		Indicates the proportion of the update to apply to the parameters
		during fitting. When the inertia is 0.0, the update is applied in
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

	verbose: bool, optional
		Whether to print the improvement and timings during training.
	"""

    def __init__(self, distributions=None, edges=None, structure=None, algorithm=None, include_parents=None, exclude_parents=None, max_parents=None, pseudocount=0.0, max_iter=20, tol=1e-06, inertia=0.0, frozen=False, check_data=True, verbose=False):
        if False:
            while True:
                i = 10
        super().__init__(inertia=inertia, frozen=frozen, check_data=check_data)
        self.name = 'BayesianNetwork'
        self.distributions = torch.nn.ModuleList([])
        self.edges = []
        self.structure = structure
        self._marginal_mapping = {}
        self._factor_mapping = {}
        self._distribution_mapping = {}
        self._parents = []
        self._factor_graph = FactorGraph()
        self.algorithm = algorithm
        self.include_parents = include_parents
        self.exclude_parents = exclude_parents
        self.max_parents = max_parents
        self.pseudocount = pseudocount
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.d = 0
        self._initialized = distributions is not None and distributions[0]._initialized
        self._reset_cache()
        if distributions is not None:
            _check_parameter(distributions, 'factors', dtypes=(list, tuple))
            for distribution in distributions:
                self.add_distribution(distribution)
        if edges is not None:
            _check_parameter(edges, 'edges', dtypes=(list, tuple))
            if isinstance(edges, (tuple, list)):
                for (parent, child) in edges:
                    self.add_edge(parent, child)
            else:
                raise ValueError('Edges must be tuple or list.')

    def _initialize(self, d):
        if False:
            print('Hello World!')
        self._initialized = True
        super()._initialize(d)

    def _reset_cache(self):
        if False:
            return 10
        return

    def add_distribution(self, distribution):
        if False:
            i = 10
            return i + 15
        'Adds a distribution to the set of distributions.\n\n\t\tAdds a distribution to the set of distributions being stored in the\n\t\tBayesianNetwork object but also updates the underlying factor graph,\n\t\tadding in a marginal node and a factor node.\n\n\t\t\n\t\tParameters\n\t\t----------\n\t\tdistribution: pomegranate.distributions.Distribution\n\t\t\tA distribution object to include as a node. Currently must be a\n\t\t\tCategorical or a ConditionalCategorical distribution.\n\t\t'
        if not isinstance(distribution, (Categorical, ConditionalCategorical)):
            raise ValueError('Must be Categorical or ConditionalCategorical')
        self.distributions.append(distribution)
        self.d += 1
        n_keys = distribution.probs[0].shape[-1]
        marginal = Categorical(torch.ones(1, n_keys) / n_keys)
        self._factor_graph.add_marginal(marginal)
        self._marginal_mapping[distribution] = marginal
        self._parents.append(tuple())
        self._distribution_mapping[distribution] = len(self.distributions) - 1
        if isinstance(distribution, ConditionalCategorical):
            p = torch.clone(distribution.probs[0])
            p /= torch.prod(torch.tensor(p.shape[:-1]))
            factor = JointCategorical(p)
            self._factor_graph.add_factor(factor)
            self._factor_mapping[distribution] = factor
        else:
            self._factor_graph.add_factor(distribution)
            self._factor_mapping[distribution] = distribution
            factor = distribution
        self._factor_graph.add_edge(marginal, factor)

    def add_distributions(self, distributions):
        if False:
            print('Hello World!')
        'Adds several distributions to the set of distributions.\n\n\t\t\n\t\tParameters\n\t\t----------\n\t\tdistribution: iterable\n\t\t\tAny object that can be iterated over that returns distributions.\n\t\t\tMust be Categorical or ConditionalCategorical distributions.\n\t\t'
        for distribution in distributions:
            self.add_distribution(distribution)

    def add_edge(self, parent, child):
        if False:
            return 10
        'Adds a directed edge from the parent to the child node.\n\n\t\tAdds an edge to the list of edges associated with the BayesianNetwork\n\t\tobject but also adds an appropriate edge in the underlying factor\n\t\tgraph between the marginal node associated with the parent and\n\t\tthe factor node associated with the child.\n\n\n\t\tParameters\n\t\t----------\n\t\tparent: pomegranate.distributions.Distribution\n\t\t\tThe distribution that the edge begins at.\n\n\t\tchild: pomegranate.distributions.Distribution\n\t\t\tThe distribution that the edge points to.\n\t\t'
        if not isinstance(child, ConditionalCategorical):
            raise ValueError('Child distribution must be conditional.')
        if parent not in self._marginal_mapping:
            raise ValueError('Parent distribution must be in network.')
        if child not in self._marginal_mapping:
            raise ValueError('Child distribution must be in network.')
        if parent is child:
            raise ValueError('Cannot have self-loops.')
        self.edges.append((parent, child))
        p_idx = self._distribution_mapping[parent]
        c_idx = self._distribution_mapping[child]
        self._parents[c_idx] += (p_idx,)
        marginal = self._marginal_mapping[parent]
        factor = self._factor_mapping[child]
        m_idx = self._factor_graph._marginal_idxs[marginal]
        f_idx = self._factor_graph._factor_idxs[factor]
        m = self._factor_graph._factor_edges[f_idx].pop()
        f = self._factor_graph._marginal_edges[m_idx].pop()
        self._factor_graph.add_edge(marginal, factor)
        self._factor_graph._factor_edges[f_idx].append(m)
        self._factor_graph._marginal_edges[m_idx].append(f)

    def add_edges(self, edges):
        if False:
            for i in range(10):
                print('nop')
        'Adds several edges to the network at once.\n\n\t\t\n\t\tParameters\n\t\t----------\n\t\tedges: iterable\n\t\t\tAny object that can be iterated over that returns tuples with\n\t\t\ta pair of distributions.\n\t\t'
        for edge in edges:
            self.add_edge(*edge)

    def sample(self, n):
        if False:
            while True:
                i = 10
        'Sample from the probability distribution.\n\n\t\tThis method will return `n` samples generated from the underlying\n\t\tprobability distribution. For a mixture model, this involves first\n\t\tsampling the component using the prior probabilities, and then sampling\n\t\tfrom the chosen distribution.\n\n\n\t\tParameters\n\t\t----------\n\t\tn: int\n\t\t\tThe number of samples to generate.\n\t\t\n\n\t\tReturns\n\t\t-------\n\t\tX: torch.tensor, shape=(n, self.d)\n\t\t\tRandomly generated samples.\n\t\t'
        X = torch.zeros(n, self.d, dtype=torch.int32) - 1
        for i in range(self.d):
            for (j, parents) in enumerate(self._parents):
                if (X[0, j] != -1).item():
                    continue
                if len(parents) == 0:
                    X[:, j] = self.distributions[j].sample(n)[:, 0]
                else:
                    X_ = X[:, parents].unsqueeze(-1)
                    if (X_ == -1).any().item():
                        continue
                    X[:, j] = self.distributions[j].sample(n, X_)[:, 0]
        return X

    def log_probability(self, X):
        if False:
            for i in range(10):
                print('nop')
        'Calculate the log probability of each example.\n\n\t\tThis method calculates the log probability of each example given the\n\t\tparameters of the distribution. The examples must be given in a 2D\n\t\tformat.\n\n\n\t\tParameters\n\t\t----------\n\t\tX: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)\n\t\t\tA set of examples to evaluate.\n\n\t\tReturns\n\t\t-------\n\t\tlogp: torch.Tensor, shape=(-1,)\n\t\t\tThe log probability of each example.\n\t\t'
        X = _check_parameter(_cast_as_tensor(X), 'X', ndim=2, check_parameter=self.check_data)
        logps = torch.zeros(X.shape[0], device=X.device, dtype=torch.float32)
        for (i, distribution) in enumerate(self.distributions):
            parents = self._parents[i] + (i,)
            X_ = X[:, parents]
            if len(parents) > 1:
                X_ = X_.unsqueeze(-1)
            logps += distribution.log_probability(X_)
        return logps

    def predict(self, X):
        if False:
            i = 10
            return i + 15
        "Infers the maximum likelihood value for each missing value.\n\n\t\tThis method infers a probability distribution for each of the missing\n\t\tvalues in the data. It uses the factor graph representation of the\n\t\tBayesian network to run the sum-product/loopy belief propogation\n\t\talgorithm. After the probability distribution is inferred, the maximum\n\t\tlikeihood value for each variable is returned.\n\n\t\tThe input to this method must be a torch.masked.MaskedTensor where the\n\t\tmask specifies which variables are observed (mask = True) and which ones\n\t\tare not observed (mask = False) for each of the values. When setting\n\t\tmask = False, it does not matter what the corresponding value in the\n\t\ttensor is. Different sets of variables can be observed or missing in\n\t\tdifferent examples. \n\n\t\tUnlike the `predict_proba` and `predict_log_proba` methods, this\n\t\tmethod preserves the dimensions of the original data because it does\n\t\tnot matter how many categories a variable can take when you're only\n\t\treturning the maximally likely one.\n\n\n\t\tParameters\n\t\t----------\n\t\tX: torch.masked.MaskedTensor, shape=(-1, d)\n\t\t\tThe data to predict values for. The mask should correspond to\n\t\t\twhether the variable is observed in the example. \n\t\t\n\n\t\tReturns\n\t\t-------\n\t\ty: torch.tensor, shape=(-1, d)\n\t\t\tA completed version of the incomplete input tensor. The missing\n\t\t\tvariables are replaced with the maximally likely values from\n\t\t\tthe sum-product algorithm, and observed variables are kept.\n\t\t"
        y = [t.argmax(dim=1) for t in self.predict_proba(X)]
        return torch.vstack(y).T.contiguous()

    def predict_proba(self, X):
        if False:
            return 10
        'Infers the probability of each category given the model and data.\n\n\t\tThis method infers a probability distribution for each of the missing \n\t\tvalues in the data. It uses the factor graph representation of the\n\t\tBayesian network to run the sum-product/loopy belief propogation\n\t\talgorithm.\n\n\t\tThe input to this method must be a torch.masked.MaskedTensor where the\n\t\tmask specifies which variables are observed (mask = True) and which ones\n\t\tare not observed (mask = False) for each of the values. When setting\n\t\tmask = False, it does not matter what the corresponding value in the\n\t\ttensor is. Different sets of variables can be observed or missing in\n\t\tdifferent examples. \n\n\t\tAn important note is that, because each variable can have a different\n\t\tnumber of categories in the categorical setting, the return is a list\n\t\tof tensors where each element in that list is the marginal probability\n\t\tdistribution for that variable. More concretely: the first element will\n\t\tbe the distribution of values for the first variable across all\n\t\texamples. When the first variable has been provided as evidence, the\n\t\tdistribution will be clamped to the value provided as evidence.\n\n\t\t..warning:: This inference is exact given a Bayesian network that has\n\t\ta tree-like structure, but is only approximate for other cases. When\n\t\tthe network is acyclic, this procedure will converge, but if the graph\n\t\tcontains cycles then there is no guarantee on convergence.\n\n\n\t\tParameters\n\t\t----------\n\t\tX: torch.masked.MaskedTensor, shape=(-1, d)\n\t\t\tThe data to predict values for. The mask should correspond to\n\t\t\twhether the variable is observed in the example. \n\t\t\n\n\t\tReturns\n\t\t-------\n\t\ty: list of tensors, shape=(d,)\n\t\t\tA list of tensors where each tensor contains the distribution of\n\t\t\tvalues for that dimension.\n\t\t'
        return self._factor_graph.predict_proba(X)

    def predict_log_proba(self, X):
        if False:
            return 10
        'Infers the probability of each category given the model and data.\n\n\t\tThis method is a wrapper around the `predict_proba` method and simply\n\t\ttakes the log of each returned tensor.\n\n\t\tThis method infers a log probability distribution for each of the \n\t\tmissing  values in the data. It uses the factor graph representation of \n\t\tthe Bayesian network to run the sum-product/loopy belief propogation\n\t\talgorithm.\n\n\t\tThe input to this method must be a torch.masked.MaskedTensor where the\n\t\tmask specifies which variables are observed (mask = True) and which ones\n\t\tare not observed (mask = False) for each of the values. When setting\n\t\tmask = False, it does not matter what the corresponding value in the\n\t\ttensor is. Different sets of variables can be observed or missing in\n\t\tdifferent examples. \n\n\t\tAn important note is that, because each variable can have a different\n\t\tnumber of categories in the categorical setting, the return is a list\n\t\tof tensors where each element in that list is the marginal probability\n\t\tdistribution for that variable. More concretely: the first element will\n\t\tbe the distribution of values for the first variable across all\n\t\texamples. When the first variable has been provided as evidence, the\n\t\tdistribution will be clamped to the value provided as evidence.\n\n\t\t..warning:: This inference is exact given a Bayesian network that has\n\t\ta tree-like structure, but is only approximate for other cases. When\n\t\tthe network is acyclic, this procedure will converge, but if the graph\n\t\tcontains cycles then there is no guarantee on convergence.\n\n\n\t\tParameters\n\t\t----------\n\t\tX: torch.masked.MaskedTensor, shape=(-1, d)\n\t\t\tThe data to predict values for. The mask should correspond to\n\t\t\twhether the variable is observed in the example. \n\t\t\n\n\t\tReturns\n\t\t-------\n\t\ty: list of tensors, shape=(d,)\n\t\t\tA list of tensors where each tensor contains the distribution of\n\t\t\tvalues for that dimension.\n\t\t'
        return [torch.log(t) for t in self.predict_proba(X)]

    def fit(self, X, sample_weight=None):
        if False:
            return 10
        'Fit the model to optionally weighted examples.\n\n\t\tThis method will fit the provided distributions given the data and\n\t\ttheir weights. If a structure is provided as a set of edges, then\n\t\tthis will use maximum likelihood estimates to fit each of those\n\t\tdistributions. If no structure is provided, this will use the\n\t\tstructure learning algorithm provided to jointly learn the structure\n\t\tand the parameters.\n\n\n\t\tParameters\n\t\t----------\n\t\tX: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)\n\t\t\tA set of examples to evaluate. \n\n\t\tsample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional\n\t\t\tA set of weights for the examples. This can be either of shape\n\t\t\t(-1, self.d) or a vector of shape (-1,). Default is ones.\n\n\n\t\tReturns\n\t\t-------\n\t\tself\n\t\t'
        if self.algorithm is not None:
            self.structure = _learn_structure(X, sample_weight=sample_weight, algorithm=self.algorithm, include_parents=self.include_parents, exclude_parents=self.exclude_parents, max_parents=self.max_parents, pseudocount=self.pseudocount)
        if self.structure is not None:
            distributions = _from_structure(X, sample_weight, self.structure)
            self.add_distributions(distributions)
            for (i, parents) in enumerate(self.structure):
                if len(parents) > 0:
                    for parent in parents:
                        self.add_edge(distributions[parent], distributions[i])
        self.summarize(X, sample_weight=sample_weight)
        self.from_summaries()
        return self

    def summarize(self, X, sample_weight=None):
        if False:
            for i in range(10):
                print('nop')
        'Extract the sufficient statistics from a batch of data.\n\n\t\tThis method calculates the sufficient statistics from optionally\n\t\tweighted data and adds them to the stored cache for each distribution\n\t\tin the network. Sample weights can either be provided as one\n\t\tvalue per example or as a 2D matrix of weights for each feature in\n\t\teach example.\n\n\n\t\tParameters\n\t\t----------\n\t\tX: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, len, self.d)\n\t\t\tA set of examples to summarize.\n\n\t\tsample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional\n\t\t\tA set of weights for the examples. This can be either of shape\n\t\t\t(-1, self.d) or a vector of shape (-1,). Default is ones.\n\n\n\t\tReturns\n\t\t-------\n\t\tlogp: torch.Tensor, shape=(-1,)\n\t\t\tThe log probability of each example.\n\t\t'
        if self.frozen:
            return
        if not self._initialized:
            self._initialize(len(X[0]))
        (X, sample_weight) = super().summarize(X, sample_weight=sample_weight)
        X = _check_parameter(X, 'X', min_value=0, ndim=2, check_parameter=self.check_data)
        for (i, distribution) in enumerate(self.distributions):
            parents = self._parents[i] + (i,)
            w = sample_weight[:, i]
            if len(parents) == 1:
                distribution.summarize(X[:, parents], sample_weight=w)
            else:
                distribution.summarize(X[:, parents].unsqueeze(-1), sample_weight=w)

    def from_summaries(self):
        if False:
            i = 10
            return i + 15
        'Update the model parameters given the extracted statistics.\n\n\t\tThis method uses calculated statistics from calls to the `summarize`\n\t\tmethod to update the distribution parameters. Hyperparameters for the\n\t\tupdate are passed in at initialization time.\n\n\t\tNote: Internally, a call to `fit` is just a successive call to the\n\t\t`summarize` method followed by the `from_summaries` method.\n\t\t'
        if self.frozen:
            return
        for distribution in self.distributions:
            distribution.from_summaries()
            if isinstance(distribution, ConditionalCategorical):
                p = torch.clone(distribution.probs[0])
                p /= torch.prod(torch.tensor(p.shape[:-1]))
                self._factor_mapping[distribution].probs = _cast_as_parameter(p)

def _from_structure(X, sample_weight=None, structure=None, pseudocount=0.0):
    if False:
        while True:
            i = 10
    'Fits a set of distributions to data given the structure.\n\n\tGiven the structure, create the distribution objects and fit their \n\tparameters to the given data. This does not perform structure learning.\n\n\n\tParameters\n\t----------\n\tX: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)\n\t\tA set of examples to evaluate. \n\n\tsample_weight: list, tuple, numpy.ndarray, torch.Tensor, or None\n\t\tA set of weights for the examples. This can be either of shape\n\t\t(-1, self.d) or a vector of shape (-1,). Default is ones.\n\n\tstructure: tuple\n\t\tA tuple of tuples where the internal tuples indicate the parents of\n\t\tthat variable. \n\n\tpseudocount: double\n\t\tA pseudocount to add to each count. Default is 0.\n\n\n\tReturns\n\t-------\n\tmodel: pomegranate.bayesian_network.BayesianNetwork\n\t\tThe fit Bayesian network.\n\t'
    X = _check_parameter(_cast_as_tensor(X), 'X', min_value=0, ndim=2, dtypes=(torch.int32, torch.int64))
    sample_weight = _check_parameter(_cast_as_tensor(sample_weight), 'X', min_value=0, ndim=1)
    (n, d) = X.shape
    if sample_weight is None:
        sample_weight = torch.ones(n, dtype=torch.float32, device=X.device)
    if structure is None:
        structure = tuple([tuple() for i in range(d)])
    model = BayesianNetwork()
    if X.device != model.device:
        model.to(X.device)
    d = len(structure)
    distributions = []
    for (i, parents) in enumerate(structure):
        if len(parents) == 0:
            d = Categorical()
            if d.device != X.device:
                d.to(X.device)
            d.fit(X[:, i:i + 1], sample_weight=sample_weight)
        else:
            parents = parents + (i,)
            d = ConditionalCategorical()
            if d.device != X.device:
                d.to(X.device)
            d.fit(X[:, parents].unsqueeze(-1), sample_weight=sample_weight)
        distributions.append(d)
    return distributions

def _learn_structure(X, sample_weight=None, algorithm='chow-liu', include_parents=None, exclude_parents=None, max_parents=None, pseudocount=0, penalty=None, root=0):
    if False:
        while True:
            i = 10
    "Learn the structure of a Bayesian network using data.\n\n\tThis function will take in data, an algorithm, and parameters for that\n\talgorithm, and will return the learned structure. Currently supported\n\talgorithms are:\n\n\t\t- 'chow-liu': Learn a maximal spanning tree that is the most likely\n\t\t\ttree given the data and weights.\n\t\t- 'exact': A dynamic programming solution to the exact BNSL task that\n\t\t\treduces the time from super-exponential to simply-exponential.\n\n\n\tParameters\n\t----------\n\tX: torch.tensor, shape=(n, d)\n\t\tThe data to fit the structure too, where each row is a sample and\n\t\teach column corresponds to the associated variable.\n\t\n\tsample_weight: torch.tensor, shape=(n,)\n\t\tThe weight of each sample as a positive double. If None, all items are \n\t\tweighted equally.\n\n\talgorithm: str\n\t\tThe name of the algorithm to use for structure learning. Must be one\n\t\tof 'chow-liu' or 'exact'.\n\t\n\tinclude_parents: list or None\n\t\tA list of tuples where each inner tuple is made up of integers\n\t\tindicating the parents that must exist for that variable. For example,\n\t\twhen passing in [(1,), (), ()], the first variable must have the second\n\t\tvariable as a parent, and potentially others if learned, and the others\n\t\tdo not have any parents that must be present. If None, no parents are\n\t\tforced. Default is None.\n\n\texclude_parents: list or None\n\t\tA list of tuples where each inner tuple is made up of integers\n\t\tindicating the parents that cannot exist for that variable. For example,\n\t\twhen passing in [(1,), (), ()], the first variable cannot have the \n\t\tsecond variable as a parent, and the other variables have no\n\t\trestrictions on it. If None, no parents are excluded. Default is None.\n\n\tmax_parents: int or None\n\t\tThe maximum number of parents a variable can have. If used, this means\n\t\tusing the k-learn procedure. Can drastically speed up algorithms.\n\t\tIf None, no max on parents. Default is None.\n\n\tpseudocount: double\n\t\tA pseudocount to add to each count. Default is 0.\n\t\n\tpenalty: float or None, optional\n\t\tThe weighting of the model complexity term in the objective function.\n\t\tIncreasing this value will encourage sparsity whereas setting the value\n\t\tto 0 will result in an unregularized structure. Default is\n\t\tlog2(|D|) / 2 where |D| is the sum of the weights of the data.\n\t\n\troot: int\n\t\tWhen using a tree-based algorithm, like Chow-Liu, sets the variable\n\t\tthat is the root of the tree.\n\n\n\tReturns\n\t-------\n\tstructure: tuple, shape=(d,)\n\t\tA tuple of tuples, where each inner tuple represents a dimension in\n\t\tthe data and contains the parents of that dimension.\n\t"
    X = _check_parameter(_cast_as_tensor(X), 'X', min_value=0, ndim=2, dtypes=(torch.int32, torch.int64))
    sample_weight = _check_parameter(_cast_as_tensor(sample_weight), 'X', min_value=0, ndim=1)
    if sample_weight is None:
        sample_weight = torch.ones(X.shape[0], dtype=torch.float32, device=X.device)
    if algorithm == 'chow-liu':
        structure = _categorical_chow_liu(X, sample_weight=sample_weight, pseudocount=pseudocount, root=root)
    elif algorithm == 'exact':
        structure = _categorical_exact(X, sample_weight=sample_weight, include_parents=include_parents, exclude_parents=exclude_parents, max_parents=max_parents, pseudocount=pseudocount)
    return structure

def _categorical_chow_liu(X, sample_weight=None, pseudocount=0.0, root=0):
    if False:
        return 10
    'An internal function for calculating a Chow-Liu tree.\n\n\tThis function calculates a Chow-Liu tree on categorical data which is\n\tpotentially weighted. A Chow-Liu tree is essentially a maximum spanning\n\ttree on the information content that adding each new variable might\n\tadd.\n\n\n\tParameters\n\t----------\n\tX: torch.tensor, shape=(n, d)\n\t\tThe data to fit the structure too, where each row is a sample and\n\t\teach column corresponds to the associated variable.\n\t\n\tsample_weight: torch.tensor or None, shape=(n,)\n\t\tThe weight of each sample as a positive double. If None, all items are \n\t\tweighted equally.\n\n\tpseudocount: double\n\t\tA pseudocount to add to each count. Default is 0.\n\t\n\troot: int\n\t\tWhen using a tree-based algorithm, like Chow-Liu, sets the variable\n\t\tthat is the root of the tree.\n\n\n\tReturns\n\t-------\n\tstructure: tuple\n\t\tA tuple of tuples where the internal tuples indicate the parents of\n\t\tthat variable. \n\t'
    X = _check_parameter(_cast_as_tensor(X), 'X', min_value=0, ndim=2, dtypes=(torch.int32, torch.int64))
    sample_weight = _check_parameter(_cast_as_tensor(sample_weight), 'X', min_value=0, ndim=1)
    pseudocount = _check_parameter(pseudocount, 'pseudocount', min_value=0, ndim=0)
    root = _check_parameter(root, 'root', min_value=0, max_value=X.shape[1], dtypes=(int, numpy.int32, numpy.int64, torch.int32, torch.int64))
    (n, d) = X.shape
    if sample_weight is None:
        sample_weight = torch.ones(n)
    start = time.time()
    n_categories = tuple((x.item() for x in X.max(dim=0).values + 1))
    max_categories = max(n_categories)
    c = max_categories ** 2
    dtype = sample_weight.dtype
    count = torch.empty(c, d, dtype=dtype, device=X.device)
    mutual_info = torch.zeros(d, d, dtype=dtype, device=X.device)
    marginals_i = torch.empty(d, c, dtype=dtype, device=X.device)
    marginals_r = torch.empty(d, c, dtype=dtype, device=X.device)
    for j in range(d):
        marg = torch.zeros(max_categories, dtype=dtype, device=X.device)
        marg.scatter_add_(0, X[:, j], sample_weight)
        marginals_i[j] = marg.repeat_interleave(max_categories)
        marginals_r[j] = marg.repeat(max_categories)
    w = sample_weight.unsqueeze(-1).expand(-1, d)
    for j in range(d):
        X_j = X[:, j:j + 1] * max_categories + X
        m = marginals_i[j:j + 1] * marginals_r
        count[:] = pseudocount
        count.scatter_add_(0, X_j, w)
        mutual_info[j] -= torch.sum(count * torch.log(count / m.T), dim=0)
        mutual_info[:, j] = mutual_info[j]
    structure = [[] for i in range(d)]
    visited = [root]
    unvisited = list(range(d))
    unvisited.remove(root)
    for i in range(d - 1):
        min_score = float('inf')
        min_x = -1
        min_y = -1
        idx = mutual_info[visited][:, unvisited].argmin()
        (row, col) = ((idx // len(unvisited)).item(), (idx % len(unvisited)).item())
        (min_x, min_y) = (visited[row], unvisited[col])
        structure[min_y].append(min_x)
        visited.append(min_y)
        unvisited.remove(min_y)
    return tuple((tuple(x) for x in structure))

def _categorical_exact(X, sample_weight=None, include_parents=None, exclude_parents=None, pseudocount=0, penalty=None, max_parents=None):
    if False:
        i = 10
        return i + 15
    "Find the optimal graph over a set of variables with no other knowledge.\n\t\n\tThis is the naive dynamic programming structure learning task where the\n\toptimal graph is identified from a set of variables using an order graph\n\tand parent graphs. This can be used either when no constraint graph is\n\tprovided or for a SCC which is made up of a node containing a self-loop.\n\tThis is a reference implementation that uses the naive shortest path\n\talgorithm over the entire order graph. The 'exact' option uses the A* path\n\tin order to avoid considering the full order graph.\n\t\n\n\tParameters\n\t----------\n\tX: numpy.ndarray, shape=(n, d)\n\t\tThe data to fit the structure too, where each row is a sample and\n\t\teach column corresponds to the associated variable.\n\t\n\tsample_weight: numpy.ndarray, shape=(n,)\n\t\tThe weight of each sample as a positive double. Default is None.\n\t\n\tinclude_parents: list or None\n\t\tA set of (parent, child) tuples where each tuple is an edge that\n\t\tmust exist in the found structure.\n\t\n\texclude_parents: list or None\n\t\tA set of (parent, child) tuples where each tuple is an edge that\n\t\tcannot exist in the found structure.\n\t\n\tpenalty: float or None, optional\n\t\tThe weighting of the model complexity term in the objective function.\n\t\tIncreasing this value will encourage sparsity whereas setting the value\n\t\tto 0 will result in an unregularized structure. Default is\n\t\tlog2(|D|) / 2 where |D| is the sum of the weights of the data.\n\t\n\tmax_parents: int\n\t\tThe maximum number of parents a node can have. If used, this means\n\t\tusing the k-learn procedure. Can drastically speed up algorithms.\n\t\tIf -1, no max on parents. Default is -1.\n\t\t\n\n\tReturns\n\t-------\n\tstructure: tuple, shape=(d,)\n\t\tThe parents for each variable in this SCC\n\t"
    X = _check_parameter(_cast_as_tensor(X), 'X', min_value=0, ndim=2, dtypes=(torch.int32, torch.int64))
    sample_weight = _check_parameter(_cast_as_tensor(sample_weight), 'X', min_value=0, ndim=1)
    (n, d) = X.shape
    n_categories = X.max(axis=0).values + 1
    if sample_weight is None:
        sample_weight = torch.ones(n)
    w = sample_weight.sum()
    if max_parents is None:
        max_parents = int(torch.log2(2 * w / torch.log2(w)).item())
    parent_graphs = []
    for i in range(d):
        exclude = None if exclude_parents is None else exclude_parents[i]
        parent_set = tuple(set(range(d)) - set([i]))
        parent_graph = _generate_parent_graph(X, sample_weight=sample_weight, n_categories=n_categories, column_idx=i, include_parents=include_parents, exclude_parents=exclude, pseudocount=pseudocount, penalty=penalty, max_parents=max_parents, parent_set=parent_set)
        parent_graphs.append(parent_graph)
    order_graph = nx.DiGraph()
    for i in range(d + 1):
        for subset in itertools.combinations(range(d), i):
            order_graph.add_node(subset)
            for variable in subset:
                parent = tuple((v for v in subset if v != variable))
                (structure, weight) = parent_graphs[variable][parent]
                weight = -weight if weight < 0 else 0
                order_graph.add_edge(parent, subset, weight=weight, structure=structure)
    path = sorted(nx.all_shortest_paths(order_graph, source=(), target=tuple(range(d)), weight='weight'))[0]
    (score, structure) = (0, list((None for i in range(d))))
    for (u, v) in zip(path[:-1], path[1:]):
        idx = list(set(v) - set(u))[0]
        parents = order_graph.get_edge_data(u, v)['structure']
        structure[idx] = parents
        score -= order_graph.get_edge_data(u, v)['weight']
    return tuple(structure)

def _generate_parent_graph(X, sample_weight, n_categories, column_idx, include_parents=None, exclude_parents=None, pseudocount=0, penalty=None, max_parents=None, parent_set=None):
    if False:
        i = 10
        return i + 15
    'Generate a parent graph for a single variable over its parents.\n\n\tThis will generate the parent graph for a single parents given the data.\n\tA parent graph is the dynamically generated best parent set and respective\n\tscore for each combination of parent variables. For example, if we are\n\tgenerating a parent graph for x1 over x2, x3, and x4, we may calculate that\n\thaving x2 as a parent is better than x2,x3 and so store the value\n\tof x2 in the node for x2,x3.\n\t\n\tParameters\n\t----------\n\tX: list, numpy.ndarray, torch.tensor, shape=(n, d)\n\t\tThe data to fit the structure too, where each row is a sample and\n\t\teach column corresponds to the associated variable.\n\t\n\tweights: list, numpy.ndarray, torch.tensor, shape=(n,)\n\t\tThe weight of each sample as a positive double. Default is None.\n\t\n\tn_categories: list, numpy.ndarray, torch.tensor, shape=(d,)\n\t\tThe number of unique keys in each column.\n\t\n\tcolumn_idx: int\n\t\tThe column index to build the parent graph for.\n\t\n\tinclude_parents: list or tuple or None\n\t\tA set of integers indicating the parents that must be included in the\n\t\treturned structure. Default is None.\n\t\n\texclude_parents: list or tuple or None\n\t\tA set of integers indicating the parents that must be excluded from\n\t\tthe returned structure. Default is None.\n\n\tpseudocount: double\n\t\tA pseudocount to add to each count. Default is 0.\n\t\n\tpenalty: float or None, optional\n\t\tThe weighting of the model complexity term in the objective function.\n\t\tIncreasing this value will encourage sparsity whereas setting the value\n\t\tto 0 will result in an unregularized structure. Default is\n\t\tlog2(|D|) / 2 where |D| is the sum of the weights of the data.\n\t\n\tmax_parents: int\n\t\tThe maximum number of parents a node can have. If used, this means\n\t\tusing the k-learn procedure. Can drastically speed up algorithms.\n\t\tIf -1, no max on parents. Default is -1.\n\t\n\tparent_set: tuple, default ()\n\t\tThe variables which are possible parents for this variable. By default,\n\t\tthis should be all variables except for idx. If excluded parents are\n\t\tpassed in, this should exclude those variables as well.\n\t\n\n\tReturns\n\t-------\n\tstructure: tuple, shape=(d,)\n\t\tThe parents for each variable in this SCC\n\t'
    (n, d) = X.shape
    n_categories = n_categories.numpy()
    max_n_categories = int(n_categories.max())
    parent_graph = {}
    X_dicts = {}
    X_cols = X.T.contiguous().type(torch.int64)
    w = sample_weight.sum()
    log_w = torch.log2(sample_weight.sum()).item() / 2
    if max_parents is None:
        max_parents = int(torch.log2(2 * w / torch.log2(w)).item())
    if parent_set is None:
        parent_set = tuple(set(range(d)) - set([column_idx]))
    if include_parents is None:
        include_parents = []
    if exclude_parents is None:
        exclude_parents = set()
    else:
        exclude_parents = set(exclude_parents)
    for j in range(len(parent_set) + 1):
        c_dims = tuple([max_n_categories for _ in range(j + 1)])
        counts = torch.zeros(*c_dims, dtype=sample_weight.dtype)
        offset = max_n_categories ** j
        X_cols_ = X_cols * offset
        for subset in itertools.combinations(parent_set, j):
            best_structure = ()
            best_score = float('-inf')
            if j <= max_parents:
                for parent in include_parents:
                    if parent not in subset:
                        break
                else:
                    for var in subset:
                        if var in exclude_parents:
                            break
                    else:
                        if j == 0:
                            X_idxs = torch.zeros(n, dtype=torch.int64)
                            idx = column_idx
                        else:
                            X_idxs = X_dicts[subset[:-1]]
                            idx = subset[-1]
                        n_params = numpy.prod(n_categories[list(subset)]) * (n_categories[column_idx] - 1)
                        X_idxs = torch.clone(X_idxs) + X_cols_[idx]
                        counts[:] = pseudocount
                        counts.view(-1).scatter_add_(0, X_idxs, sample_weight)
                        marginal_counts = counts.sum(dim=-1, keepdims=True)
                        logp = torch.sum(counts * torch.log2(counts / marginal_counts))
                        best_structure = subset
                        best_score = logp - log_w * n_params
                        X_dicts[subset] = X_idxs
            for (k, variable) in enumerate(subset):
                parent_subset = tuple((l for l in subset if l != variable))
                (structure, score) = parent_graph[parent_subset]
                if score > best_score:
                    best_score = score
                    best_structure = structure
            parent_graph[subset] = (best_structure, best_score)
    return parent_graph