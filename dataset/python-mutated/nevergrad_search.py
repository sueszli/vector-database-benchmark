import inspect
import logging
import pickle
from typing import Dict, Optional, Type, Union, List, Sequence
from ray.tune.result import DEFAULT_METRIC
from ray.tune.search.sample import Categorical, Domain, Float, Integer, LogUniform, Quantized
from ray.tune.search import UNRESOLVED_SEARCH_SPACE, UNDEFINED_METRIC_MODE, UNDEFINED_SEARCH_SPACE, Searcher
from ray.tune.search.variant_generator import parse_spec_vars
from ray.tune.utils.util import flatten_dict, unflatten_dict
try:
    import nevergrad as ng
    from nevergrad.optimization import Optimizer
    from nevergrad.optimization.base import ConfiguredOptimizer
    Parameter = ng.p.Parameter
except ImportError:
    ng = None
    Optimizer = None
    ConfiguredOptimizer = None
    Parameter = None
logger = logging.getLogger(__name__)

class NevergradSearch(Searcher):
    """Uses Nevergrad to optimize hyperparameters.

    Nevergrad is an open source tool from Facebook for derivative free
    optimization.  More info can be found at:
    https://github.com/facebookresearch/nevergrad.

    You will need to install Nevergrad via the following command:

    .. code-block:: bash

        $ pip install nevergrad

    Parameters:
        optimizer: Optimizer class provided from Nevergrad.
            See here for available optimizers:
            https://facebookresearch.github.io/nevergrad/optimizers_ref.html#optimizers
            This can also be an instance of a `ConfiguredOptimizer`. See the
            section on configured optimizers in the above link.
        optimizer_kwargs: Kwargs passed in when instantiating the `optimizer`
        space: Nevergrad parametrization
            to be passed to optimizer on instantiation, or list of parameter
            names if you passed an optimizer object.
        metric: The training result objective value attribute. If None
            but a mode was passed, the anonymous metric `_metric` will be used
            per default.
        mode: One of {min, max}. Determines whether objective is
            minimizing or maximizing the metric attribute.
        points_to_evaluate: Initial parameter suggestions to be run
            first. This is for when you already have some good parameters
            you want to run first to help the algorithm make better suggestions
            for future parameters. Needs to be a list of dicts containing the
            configurations.

    Tune automatically converts search spaces to Nevergrad's format:

    .. code-block:: python

        import nevergrad as ng

        config = {
            "width": tune.uniform(0, 20),
            "height": tune.uniform(-100, 100),
            "activation": tune.choice(["relu", "tanh"])
        }

        current_best_params = [{
            "width": 10,
            "height": 0,
            "activation": relu",
        }]

        ng_search = NevergradSearch(
            optimizer=ng.optimizers.OnePlusOne,
            metric="mean_loss",
            mode="min",
            points_to_evaluate=current_best_params)

        run(my_trainable, config=config, search_alg=ng_search)

    If you would like to pass the search space manually, the code would
    look like this:

    .. code-block:: python

        import nevergrad as ng

        space = ng.p.Dict(
            width=ng.p.Scalar(lower=0, upper=20),
            height=ng.p.Scalar(lower=-100, upper=100),
            activation=ng.p.Choice(choices=["relu", "tanh"])
        )

        ng_search = NevergradSearch(
            optimizer=ng.optimizers.OnePlusOne,
            space=space,
            metric="mean_loss",
            mode="min")

        run(my_trainable, search_alg=ng_search)

    """

    def __init__(self, optimizer: Optional[Union[Optimizer, Type[Optimizer], ConfiguredOptimizer]]=None, optimizer_kwargs: Optional[Dict]=None, space: Optional[Union[Dict, Parameter]]=None, metric: Optional[str]=None, mode: Optional[str]=None, points_to_evaluate: Optional[List[Dict]]=None):
        if False:
            print('Hello World!')
        assert ng is not None, 'Nevergrad must be installed!\n            You can install Nevergrad with the command:\n            `pip install nevergrad`.'
        if mode:
            assert mode in ['min', 'max'], "`mode` must be 'min' or 'max'."
        super(NevergradSearch, self).__init__(metric=metric, mode=mode)
        self._space = None
        self._opt_factory = None
        self._nevergrad_opt = None
        self._optimizer_kwargs = optimizer_kwargs or {}
        if points_to_evaluate is None:
            self._points_to_evaluate = None
        elif not isinstance(points_to_evaluate, Sequence):
            raise ValueError(f'Invalid object type passed for `points_to_evaluate`: {type(points_to_evaluate)}. Please pass a list of points (dictionaries) instead.')
        else:
            self._points_to_evaluate = list(points_to_evaluate)
        if isinstance(space, dict) and space:
            (resolved_vars, domain_vars, grid_vars) = parse_spec_vars(space)
            if domain_vars or grid_vars:
                logger.warning(UNRESOLVED_SEARCH_SPACE.format(par='space', cls=type(self)))
                space = self.convert_search_space(space)
        if isinstance(optimizer, Optimizer):
            if space is not None and (not isinstance(space, list)):
                raise ValueError('If you pass a configured optimizer to Nevergrad, either pass a list of parameter names or None as the `space` parameter.')
            if self._optimizer_kwargs:
                raise ValueError('If you pass in optimizer kwargs, either pass an `Optimizer` subclass or an instance of `ConfiguredOptimizer`.')
            self._parameters = space
            self._nevergrad_opt = optimizer
        elif inspect.isclass(optimizer) and issubclass(optimizer, Optimizer) or isinstance(optimizer, ConfiguredOptimizer):
            self._opt_factory = optimizer
            self._parameters = None
            self._space = space
        else:
            raise ValueError('The `optimizer` argument passed to NevergradSearch must be either an `Optimizer` or a `ConfiguredOptimizer`.')
        self._live_trial_mapping = {}
        if self._nevergrad_opt or self._space:
            self._setup_nevergrad()

    def _setup_nevergrad(self):
        if False:
            for i in range(10):
                print('nop')
        if self._opt_factory:
            self._nevergrad_opt = self._opt_factory(self._space, **self._optimizer_kwargs)
        if self._mode == 'max':
            self._metric_op = -1.0
        elif self._mode == 'min':
            self._metric_op = 1.0
        if self._metric is None and self._mode:
            self._metric = DEFAULT_METRIC
        if hasattr(self._nevergrad_opt, 'instrumentation'):
            if self._nevergrad_opt.instrumentation.kwargs:
                if self._nevergrad_opt.instrumentation.args:
                    raise ValueError('Instrumented optimizers should use kwargs only')
                if self._parameters is not None:
                    raise ValueError('Instrumented optimizers should provide None as parameter_names')
            else:
                if self._parameters is None:
                    raise ValueError('Non-instrumented optimizers should have a list of parameter_names')
                if len(self._nevergrad_opt.instrumentation.args) != 1:
                    raise ValueError('Instrumented optimizers should use kwargs only')
        if self._parameters is not None and self._nevergrad_opt.dimension != len(self._parameters):
            raise ValueError('len(parameters_names) must match optimizer dimension for non-instrumented optimizers')
        if self._points_to_evaluate:
            for i in range(len(self._points_to_evaluate) - 1, -1, -1):
                self._nevergrad_opt.suggest(self._points_to_evaluate[i])

    def set_search_properties(self, metric: Optional[str], mode: Optional[str], config: Dict, **spec) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if self._nevergrad_opt or self._space:
            return False
        space = self.convert_search_space(config)
        self._space = space
        if metric:
            self._metric = metric
        if mode:
            self._mode = mode
        self._setup_nevergrad()
        return True

    def suggest(self, trial_id: str) -> Optional[Dict]:
        if False:
            while True:
                i = 10
        if not self._nevergrad_opt:
            raise RuntimeError(UNDEFINED_SEARCH_SPACE.format(cls=self.__class__.__name__, space='space'))
        if not self._metric or not self._mode:
            raise RuntimeError(UNDEFINED_METRIC_MODE.format(cls=self.__class__.__name__, metric=self._metric, mode=self._mode))
        suggested_config = self._nevergrad_opt.ask()
        self._live_trial_mapping[trial_id] = suggested_config
        if not suggested_config.kwargs:
            if self._parameters:
                return unflatten_dict(dict(zip(self._parameters, suggested_config.args[0])))
            return unflatten_dict(suggested_config.value)
        else:
            return unflatten_dict(suggested_config.kwargs)

    def on_trial_complete(self, trial_id: str, result: Optional[Dict]=None, error: bool=False):
        if False:
            i = 10
            return i + 15
        'Notification for the completion of trial.\n\n        The result is internally negated when interacting with Nevergrad\n        so that Nevergrad Optimizers can "maximize" this value,\n        as it minimizes on default.\n        '
        if result:
            self._process_result(trial_id, result)
        self._live_trial_mapping.pop(trial_id)

    def _process_result(self, trial_id: str, result: Dict):
        if False:
            print('Hello World!')
        ng_trial_info = self._live_trial_mapping[trial_id]
        self._nevergrad_opt.tell(ng_trial_info, self._metric_op * result[self._metric])

    def save(self, checkpoint_path: str):
        if False:
            return 10
        save_object = self.__dict__
        with open(checkpoint_path, 'wb') as outputFile:
            pickle.dump(save_object, outputFile)

    def restore(self, checkpoint_path: str):
        if False:
            while True:
                i = 10
        with open(checkpoint_path, 'rb') as inputFile:
            save_object = pickle.load(inputFile)
        self.__dict__.update(save_object)

    @staticmethod
    def convert_search_space(spec: Dict) -> Parameter:
        if False:
            print('Hello World!')
        (resolved_vars, domain_vars, grid_vars) = parse_spec_vars(spec)
        if grid_vars:
            raise ValueError('Grid search parameters cannot be automatically converted to a Nevergrad search space.')
        spec = flatten_dict(spec, prevent_delimiter=True)
        (resolved_vars, domain_vars, grid_vars) = parse_spec_vars(spec)

        def resolve_value(domain: Domain) -> Parameter:
            if False:
                print('Hello World!')
            sampler = domain.get_sampler()
            if isinstance(sampler, Quantized):
                logger.warning('Nevergrad does not support quantization. Dropped quantization.')
                sampler = sampler.get_sampler()
            if isinstance(domain, Float):
                if isinstance(sampler, LogUniform):
                    return ng.p.Log(lower=domain.lower, upper=domain.upper, exponent=sampler.base)
                return ng.p.Scalar(lower=domain.lower, upper=domain.upper)
            elif isinstance(domain, Integer):
                if isinstance(sampler, LogUniform):
                    return ng.p.Log(lower=domain.lower, upper=domain.upper - 1, exponent=sampler.base).set_integer_casting()
                return ng.p.Scalar(lower=domain.lower, upper=domain.upper - 1).set_integer_casting()
            elif isinstance(domain, Categorical):
                return ng.p.Choice(choices=domain.categories)
            raise ValueError('Nevergrad does not support parameters of type `{}` with samplers of type `{}`'.format(type(domain).__name__, type(domain.sampler).__name__))
        space = {'/'.join(path): resolve_value(domain) for (path, domain) in domain_vars}
        return ng.p.Dict(**space)