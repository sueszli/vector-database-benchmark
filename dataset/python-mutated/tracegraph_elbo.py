import weakref
from collections import defaultdict
from operator import itemgetter
import torch
import pyro
import pyro.ops.jit
from pyro.distributions.util import detach, is_identically_zero
from pyro.infer import ELBO
from pyro.infer.enum import get_importance_trace
from pyro.infer.util import MultiFrameTensor, get_plate_stacks, is_validation_enabled, torch_backward, torch_item
from pyro.ops.provenance import detach_provenance, get_provenance, track_provenance
from pyro.poutine.messenger import Messenger
from pyro.poutine.subsample_messenger import _Subsample
from pyro.util import check_if_enumerated, warn_if_nan

def _get_baseline_options(site):
    if False:
        while True:
            i = 10
    '\n    Extracts baseline options from ``site["infer"]["baseline"]``.\n    '
    options_dict = site['infer'].get('baseline', {}).copy()
    options_tuple = (options_dict.pop('nn_baseline', None), options_dict.pop('nn_baseline_input', None), options_dict.pop('use_decaying_avg_baseline', False), options_dict.pop('baseline_beta', 0.9), options_dict.pop('baseline_value', None))
    if options_dict:
        raise ValueError('Unrecognized baseline options: {}'.format(options_dict.keys()))
    return options_tuple

def _construct_baseline(node, guide_site, downstream_cost):
    if False:
        print('Hello World!')
    baseline = 0.0
    baseline_loss = 0.0
    (nn_baseline, nn_baseline_input, use_decaying_avg_baseline, baseline_beta, baseline_value) = _get_baseline_options(guide_site)
    use_nn_baseline = nn_baseline is not None
    use_baseline_value = baseline_value is not None
    use_baseline = use_nn_baseline or use_decaying_avg_baseline or use_baseline_value
    assert not (use_nn_baseline and use_baseline_value), 'cannot use baseline_value and nn_baseline simultaneously'
    if use_decaying_avg_baseline:
        dc_shape = downstream_cost.shape
        param_name = '__baseline_avg_downstream_cost_' + node
        with torch.no_grad():
            avg_downstream_cost_old = pyro.param(param_name, torch.zeros(dc_shape, device=guide_site['value'].device))
            avg_downstream_cost_new = (1 - baseline_beta) * downstream_cost + baseline_beta * avg_downstream_cost_old
        pyro.get_param_store()[param_name] = avg_downstream_cost_new
        baseline += avg_downstream_cost_old
    if use_nn_baseline:
        baseline += nn_baseline(detach(nn_baseline_input))
    elif use_baseline_value:
        baseline += baseline_value
    if use_nn_baseline or use_baseline_value:
        baseline_loss += torch.pow(downstream_cost.detach() - baseline, 2.0).sum()
    if use_baseline:
        if downstream_cost.shape != baseline.shape:
            raise ValueError('Expected baseline at site {} to be {} instead got {}'.format(node, downstream_cost.shape, baseline.shape))
    return (use_baseline, baseline_loss, baseline)

def _compute_downstream_costs(model_trace, guide_trace, non_reparam_nodes):
    if False:
        for i in range(10):
            print('nop')
    topo_sort_guide_nodes = guide_trace.topological_sort(reverse=True)
    topo_sort_guide_nodes = [x for x in topo_sort_guide_nodes if guide_trace.nodes[x]['type'] == 'sample']
    ordered_guide_nodes_dict = {n: i for (i, n) in enumerate(topo_sort_guide_nodes)}
    downstream_guide_cost_nodes = {}
    downstream_costs = {}
    stacks = get_plate_stacks(model_trace)
    for node in topo_sort_guide_nodes:
        downstream_costs[node] = MultiFrameTensor((stacks[node], model_trace.nodes[node]['log_prob'] - guide_trace.nodes[node]['log_prob']))
        nodes_included_in_sum = set([node])
        downstream_guide_cost_nodes[node] = set([node])
        children = [(k, -ordered_guide_nodes_dict[k]) for k in guide_trace.successors(node)]
        sorted_children = sorted(children, key=itemgetter(1))
        for (child, _) in sorted_children:
            child_cost_nodes = downstream_guide_cost_nodes[child]
            downstream_guide_cost_nodes[node].update(child_cost_nodes)
            if nodes_included_in_sum.isdisjoint(child_cost_nodes):
                downstream_costs[node].add(*downstream_costs[child].items())
                nodes_included_in_sum.update(child_cost_nodes)
        missing_downstream_costs = downstream_guide_cost_nodes[node] - nodes_included_in_sum
        for missing_node in missing_downstream_costs:
            downstream_costs[node].add((stacks[missing_node], model_trace.nodes[missing_node]['log_prob'] - guide_trace.nodes[missing_node]['log_prob']))
    for site in non_reparam_nodes:
        children_in_model = set()
        for node in downstream_guide_cost_nodes[site]:
            children_in_model.update(model_trace.successors(node))
        children_in_model.difference_update(downstream_guide_cost_nodes[site])
        for child in children_in_model:
            assert model_trace.nodes[child]['type'] == 'sample'
            downstream_costs[site].add((stacks[child], model_trace.nodes[child]['log_prob']))
            downstream_guide_cost_nodes[site].update([child])
    for k in non_reparam_nodes:
        downstream_costs[k] = downstream_costs[k].sum_to(guide_trace.nodes[k]['cond_indep_stack'])
    return (downstream_costs, downstream_guide_cost_nodes)

def _compute_elbo(model_trace, guide_trace):
    if False:
        return 10
    elbo = 0.0
    surrogate_elbo = 0.0
    baseline_loss = 0.0
    downstream_costs = defaultdict(lambda : MultiFrameTensor())
    for (name, site) in model_trace.nodes.items():
        if site['type'] == 'sample':
            elbo += site['log_prob_sum']
            surrogate_elbo += site['log_prob_sum']
            for key in get_provenance(site['log_prob_sum']):
                downstream_costs[key].add((site['cond_indep_stack'], site['log_prob']))
    for (name, site) in guide_trace.nodes.items():
        if site['type'] == 'sample':
            elbo -= site['log_prob_sum']
            entropy_term = site['score_parts'].entropy_term
            if not is_identically_zero(entropy_term):
                surrogate_elbo -= entropy_term.sum()
            for key in get_provenance(site['log_prob_sum']):
                downstream_costs[key].add((site['cond_indep_stack'], -site['log_prob']))
    for (node, downstream_cost) in downstream_costs.items():
        guide_site = guide_trace.nodes[node]
        downstream_cost = downstream_cost.sum_to(guide_site['cond_indep_stack'])
        score_function = guide_site['score_parts'].score_function
        (use_baseline, baseline_loss_term, baseline) = _construct_baseline(node, guide_site, downstream_cost)
        if use_baseline:
            downstream_cost = downstream_cost - baseline
            baseline_loss = baseline_loss + baseline_loss_term
        surrogate_elbo += (score_function * downstream_cost.detach()).sum()
    surrogate_loss = -surrogate_elbo + baseline_loss
    return (detach_provenance(elbo), detach_provenance(surrogate_loss))

class TrackNonReparam(Messenger):
    """
    Track non-reparameterizable sample sites.

    **References:**

    1. *Nonstandard Interpretations of Probabilistic Programs for Efficient Inference*,
        David Wingate, Noah Goodman, Andreas Stuhlmüller, Jeffrey Siskind

    **Example:**

    .. doctest::

       >>> import torch
       >>> import pyro
       >>> import pyro.distributions as dist
       >>> from pyro.infer.tracegraph_elbo import TrackNonReparam
       >>> from pyro.ops.provenance import get_provenance
       >>> from pyro.poutine import trace

       >>> def model():
       ...     probs_a = torch.tensor([0.3, 0.7])
       ...     probs_b = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
       ...     probs_c = torch.tensor([[0.5, 0.5], [0.6, 0.4]])
       ...     a = pyro.sample("a", dist.Categorical(probs_a))
       ...     b = pyro.sample("b", dist.Categorical(probs_b[a]))
       ...     pyro.sample("c", dist.Categorical(probs_c[b]), obs=torch.tensor(0))

       >>> with TrackNonReparam():
       ...     model_tr = trace(model).get_trace()
       >>> model_tr.compute_log_prob()

       >>> print(get_provenance(model_tr.nodes["a"]["log_prob"]))  # doctest: +SKIP
       frozenset({'a'})
       >>> print(get_provenance(model_tr.nodes["b"]["log_prob"]))  # doctest: +SKIP
       frozenset({'b', 'a'})
       >>> print(get_provenance(model_tr.nodes["c"]["log_prob"]))  # doctest: +SKIP
       frozenset({'b', 'a'})
    """

    def _pyro_post_sample(self, msg):
        if False:
            i = 10
            return i + 15
        if msg['type'] == 'sample' and (not isinstance(msg['fn'], _Subsample)) and (not msg['is_observed']) and (not getattr(msg['fn'], 'has_rsample', False)):
            provenance = frozenset({msg['name']})
            msg['value'] = track_provenance(msg['value'], provenance)

class TraceGraph_ELBO(ELBO):
    """
    A TraceGraph implementation of ELBO-based SVI. The gradient estimator
    is constructed along the lines of reference [1] specialized to the case
    of the ELBO. It supports arbitrary dependency structure for the model
    and guide as well as baselines for non-reparameterizable random variables.
    Fine-grained conditional dependency information as recorded in the
    :class:`~pyro.poutine.trace.Trace` is used to reduce the variance of the gradient estimator.
    In particular provenance tracking [3] is used to find the ``cost`` terms
    that depend on each non-reparameterizable sample site.

    References

    [1] `Gradient Estimation Using Stochastic Computation Graphs`,
        John Schulman, Nicolas Heess, Theophane Weber, Pieter Abbeel

    [2] `Neural Variational Inference and Learning in Belief Networks`
        Andriy Mnih, Karol Gregor

    [3] `Nonstandard Interpretations of Probabilistic Programs for Efficient Inference`,
        David Wingate, Noah Goodman, Andreas Stuhlmüller, Jeffrey Siskind
    """

    def _get_trace(self, model, guide, args, kwargs):
        if False:
            return 10
        '\n        Returns a single trace from the guide, and the model that is run\n        against it.\n        '
        with TrackNonReparam():
            (model_trace, guide_trace) = get_importance_trace('dense', self.max_plate_nesting, model, guide, args, kwargs)
        if is_validation_enabled():
            check_if_enumerated(guide_trace)
        return (model_trace, guide_trace)

    def loss(self, model, guide, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        :returns: returns an estimate of the ELBO\n        :rtype: float\n\n        Evaluates the ELBO with an estimator that uses num_particles many samples/particles.\n        '
        elbo = 0.0
        for (model_trace, guide_trace) in self._get_traces(model, guide, args, kwargs):
            elbo_particle = torch_item(model_trace.log_prob_sum()) - torch_item(guide_trace.log_prob_sum())
            elbo += elbo_particle / float(self.num_particles)
        loss = -elbo
        warn_if_nan(loss, 'loss')
        return loss

    def loss_and_grads(self, model, guide, *args, **kwargs):
        if False:
            return 10
        '\n        :returns: returns an estimate of the ELBO\n        :rtype: float\n\n        Computes the ELBO as well as the surrogate ELBO that is used to form the gradient estimator.\n        Performs backward on the latter. Num_particle many samples are used to form the estimators.\n        If baselines are present, a baseline loss is also constructed and differentiated.\n        '
        (elbo, surrogate_loss) = self._loss_and_surrogate_loss(model, guide, args, kwargs)
        torch_backward(surrogate_loss, retain_graph=self.retain_graph)
        elbo = torch_item(elbo)
        loss = -elbo
        warn_if_nan(loss, 'loss')
        return loss

    def _loss_and_surrogate_loss(self, model, guide, args, kwargs):
        if False:
            for i in range(10):
                print('nop')
        loss = 0.0
        surrogate_loss = 0.0
        for (model_trace, guide_trace) in self._get_traces(model, guide, args, kwargs):
            (lp, slp) = self._loss_and_surrogate_loss_particle(model_trace, guide_trace)
            loss += lp
            surrogate_loss += slp
        loss /= self.num_particles
        surrogate_loss /= self.num_particles
        return (loss, surrogate_loss)

    def _loss_and_surrogate_loss_particle(self, model_trace, guide_trace):
        if False:
            i = 10
            return i + 15
        (elbo, surrogate_loss) = _compute_elbo(model_trace, guide_trace)
        return (elbo, surrogate_loss)

class JitTraceGraph_ELBO(TraceGraph_ELBO):
    """
    Like :class:`TraceGraph_ELBO` but uses :func:`torch.jit.trace` to
    compile :meth:`loss_and_grads`.

    This works only for a limited set of models:

    -   Models must have static structure.
    -   Models must not depend on any global data (except the param store).
    -   All model inputs that are tensors must be passed in via ``*args``.
    -   All model inputs that are *not* tensors must be passed in via
        ``**kwargs``, and compilation will be triggered once per unique
        ``**kwargs``.
    """

    def loss_and_grads(self, model, guide, *args, **kwargs):
        if False:
            while True:
                i = 10
        kwargs['_pyro_model_id'] = id(model)
        kwargs['_pyro_guide_id'] = id(guide)
        if getattr(self, '_jit_loss_and_surrogate_loss', None) is None:
            weakself = weakref.ref(self)

            @pyro.ops.jit.trace(ignore_warnings=self.ignore_jit_warnings, jit_options=self.jit_options)
            def jit_loss_and_surrogate_loss(*args, **kwargs):
                if False:
                    i = 10
                    return i + 15
                kwargs.pop('_pyro_model_id')
                kwargs.pop('_pyro_guide_id')
                self = weakself()
                return self._loss_and_surrogate_loss(model, guide, args, kwargs)
            self._jit_loss_and_surrogate_loss = jit_loss_and_surrogate_loss
        (elbo, surrogate_loss) = self._jit_loss_and_surrogate_loss(*args, **kwargs)
        surrogate_loss.backward(retain_graph=self.retain_graph)
        loss = -elbo.item()
        warn_if_nan(loss, 'loss')
        return loss