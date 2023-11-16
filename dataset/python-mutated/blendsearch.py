from typing import Dict, Optional, List, Tuple, Callable, Union
import numpy as np
import time
import pickle
try:
    from ray import __version__ as ray_version
    assert ray_version >= '1.10.0'
    if ray_version.startswith('1.'):
        from ray.tune.suggest import Searcher
        from ray.tune.suggest.optuna import OptunaSearch as GlobalSearch
    else:
        from ray.tune.search import Searcher
        from ray.tune.search.optuna import OptunaSearch as GlobalSearch
except (ImportError, AssertionError):
    from .suggestion import Searcher
    from .suggestion import OptunaSearch as GlobalSearch
from ..trial import unflatten_dict, flatten_dict
from .. import INCUMBENT_RESULT
from .search_thread import SearchThread
from .flow2 import FLOW2
from ..space import add_cost_to_space, indexof, normalize, define_by_run_func
from ..result import TIME_TOTAL_S
import logging
SEARCH_THREAD_EPS = 1.0
PENALTY = 10000000000.0
logger = logging.getLogger(__name__)

class BlendSearch(Searcher):
    """class for BlendSearch algorithm."""
    lagrange = '_lagrange'
    LocalSearch = FLOW2

    def __init__(self, metric: Optional[str]=None, mode: Optional[str]=None, space: Optional[dict]=None, low_cost_partial_config: Optional[dict]=None, cat_hp_cost: Optional[dict]=None, points_to_evaluate: Optional[List[dict]]=None, evaluated_rewards: Optional[List]=None, time_budget_s: Union[int, float]=None, num_samples: Optional[int]=None, resource_attr: Optional[str]=None, min_resource: Optional[float]=None, max_resource: Optional[float]=None, reduction_factor: Optional[float]=None, global_search_alg: Optional[Searcher]=None, config_constraints: Optional[List[Tuple[Callable[[dict], float], str, float]]]=None, metric_constraints: Optional[List[Tuple[str, str, float]]]=None, seed: Optional[int]=20, cost_attr: Optional[str]='auto', cost_budget: Optional[float]=None, experimental: Optional[bool]=False, lexico_objectives: Optional[dict]=None, use_incumbent_result_in_evaluation=False, allow_empty_config=False):
        if False:
            for i in range(10):
                print('nop')
        'Constructor.\n\n        Args:\n            metric: A string of the metric name to optimize for.\n            mode: A string in [\'min\', \'max\'] to specify the objective as\n                minimization or maximization.\n            space: A dictionary to specify the search space.\n            low_cost_partial_config: A dictionary from a subset of\n                controlled dimensions to the initial low-cost values.\n                E.g., ```{\'n_estimators\': 4, \'max_leaves\': 4}```.\n            cat_hp_cost: A dictionary from a subset of categorical dimensions\n                to the relative cost of each choice.\n                E.g., ```{\'tree_method\': [1, 1, 2]}```.\n                I.e., the relative cost of the three choices of \'tree_method\'\n                is 1, 1 and 2 respectively.\n            points_to_evaluate: Initial parameter suggestions to be run first.\n            evaluated_rewards (list): If you have previously evaluated the\n                parameters passed in as points_to_evaluate you can avoid\n                re-running those trials by passing in the reward attributes\n                as a list so the optimiser can be told the results without\n                needing to re-compute the trial. Must be the same or shorter length than\n                points_to_evaluate. When provided, `mode` must be specified.\n            time_budget_s: int or float | Time budget in seconds.\n            num_samples: int | The number of configs to try. -1 means no limit on the\n                number of configs to try.\n            resource_attr: A string to specify the resource dimension and the best\n                performance is assumed to be at the max_resource.\n            min_resource: A float of the minimal resource to use for the resource_attr.\n            max_resource: A float of the maximal resource to use for the resource_attr.\n            reduction_factor: A float of the reduction factor used for\n                incremental pruning.\n            global_search_alg: A Searcher instance as the global search\n                instance. If omitted, Optuna is used. The following algos have\n                known issues when used as global_search_alg:\n                - HyperOptSearch raises exception sometimes\n                - TuneBOHB has its own scheduler\n            config_constraints: A list of config constraints to be satisfied.\n                E.g., ```config_constraints = [(mem_size, \'<=\', 1024**3)]```.\n                `mem_size` is a function which produces a float number for the bytes\n                needed for a config.\n                It is used to skip configs which do not fit in memory.\n            metric_constraints: A list of metric constraints to be satisfied.\n                E.g., `[\'precision\', \'>=\', 0.9]`. The sign can be ">=" or "<=".\n            seed: An integer of the random seed.\n            cost_attr: None or str to specify the attribute to evaluate the cost of different trials.\n                Default is "auto", which means that we will automatically choose the cost attribute to use (depending\n                on the nature of the resource budget). When cost_attr is set to None, cost differences between different trials will be omitted\n                in our search algorithm. When cost_attr is set to a str different from "auto" and "time_total_s",\n                this cost_attr must be available in the result dict of the trial.\n            cost_budget: A float of the cost budget. Only valid when cost_attr is a str different from "auto" and "time_total_s".\n            lexico_objectives: dict, default=None | It specifics information needed to perform multi-objective\n                optimization with lexicographic preferences. This is only supported in CFO currently.\n                When lexico_objectives is not None, the arguments metric, mode will be invalid.\n                This dictionary shall contain the  following fields of key-value pairs:\n                - "metrics":  a list of optimization objectives with the orders reflecting the priorities/preferences of the\n                objectives.\n                - "modes" (optional): a list of optimization modes (each mode either "min" or "max") corresponding to the\n                objectives in the metric list. If not provided, we use "min" as the default mode for all the objectives.\n                - "targets" (optional): a dictionary to specify the optimization targets on the objectives. The keys are the\n                metric names (provided in "metric"), and the values are the numerical target values.\n                - "tolerances" (optional): a dictionary to specify the optimality tolerances on objectives. The keys are the metric names (provided in "metrics"), and the values are the absolute/percentage tolerance in the form of numeric/string.\n                E.g.,\n                ```python\n                lexico_objectives = {\n                    "metrics": ["error_rate", "pred_time"],\n                    "modes": ["min", "min"],\n                    "tolerances": {"error_rate": 0.01, "pred_time": 0.0},\n                    "targets": {"error_rate": 0.0},\n                }\n                ```\n                We also support percentage tolerance.\n                E.g.,\n                ```python\n                lexico_objectives = {\n                    "metrics": ["error_rate", "pred_time"],\n                    "modes": ["min", "min"],\n                    "tolerances": {"error_rate": "5%", "pred_time": "0%"},\n                    "targets": {"error_rate": 0.0},\n                   }\n                ```\n            experimental: A bool of whether to use experimental features.\n        '
        self._eps = SEARCH_THREAD_EPS
        self._input_cost_attr = cost_attr
        if cost_attr == 'auto':
            if time_budget_s is not None:
                self.cost_attr = TIME_TOTAL_S
            else:
                self.cost_attr = None
            self._cost_budget = None
        else:
            self.cost_attr = cost_attr
            self._cost_budget = cost_budget
        self.penalty = PENALTY
        (self._metric, self._mode) = (metric, mode)
        self._use_incumbent_result_in_evaluation = use_incumbent_result_in_evaluation
        self.lexico_objectives = lexico_objectives
        init_config = low_cost_partial_config or {}
        if not init_config:
            logger.info("No low-cost partial config given to the search algorithm. For cost-frugal search, consider providing low-cost values for cost-related hps via 'low_cost_partial_config'. More info can be found at https://microsoft.github.io/FLAML/docs/FAQ#about-low_cost_partial_config-in-tune")
        if evaluated_rewards:
            assert mode, 'mode must be specified when evaluted_rewards is provided.'
            self._points_to_evaluate = []
            self._evaluated_rewards = []
            n = len(evaluated_rewards)
            self._evaluated_points = points_to_evaluate[:n]
            new_points_to_evaluate = points_to_evaluate[n:]
            self._all_rewards = evaluated_rewards
            best = max(evaluated_rewards) if mode == 'max' else min(evaluated_rewards)
            for (i, r) in enumerate(evaluated_rewards):
                if r == best:
                    p = points_to_evaluate[i]
                    self._points_to_evaluate.append(p)
                    self._evaluated_rewards.append(r)
            self._points_to_evaluate.extend(new_points_to_evaluate)
        else:
            self._points_to_evaluate = points_to_evaluate or []
            self._evaluated_rewards = evaluated_rewards or []
        self._config_constraints = config_constraints
        self._metric_constraints = metric_constraints
        if metric_constraints:
            assert all((x[1] in ['<=', '>='] for x in metric_constraints)), 'sign of metric constraints must be <= or >=.'
            metric += self.lagrange
        self._cat_hp_cost = cat_hp_cost or {}
        if space:
            add_cost_to_space(space, init_config, self._cat_hp_cost)
        self._ls = self.LocalSearch(init_config, metric, mode, space, resource_attr, min_resource, max_resource, reduction_factor, self.cost_attr, seed, self.lexico_objectives)
        if global_search_alg is not None:
            self._gs = global_search_alg
        elif getattr(self, '__name__', None) != 'CFO':
            if space and self._ls.hierarchical:
                from functools import partial
                gs_space = partial(define_by_run_func, space=space)
                evaluated_rewards = None
            else:
                gs_space = space
            gs_seed = seed - 10 if seed - 10 >= 0 else seed - 11 + (1 << 32)
            self._gs_seed = gs_seed
            if experimental:
                import optuna as ot
                sampler = ot.samplers.TPESampler(seed=gs_seed, multivariate=True, group=True)
            else:
                sampler = None
            try:
                assert evaluated_rewards
                self._gs = GlobalSearch(space=gs_space, metric=metric, mode=mode, seed=gs_seed, sampler=sampler, points_to_evaluate=self._evaluated_points, evaluated_rewards=evaluated_rewards)
            except (AssertionError, ValueError):
                self._gs = GlobalSearch(space=gs_space, metric=metric, mode=mode, seed=gs_seed, sampler=sampler)
            self._gs.space = space
        else:
            self._gs = None
        self._experimental = experimental
        if getattr(self, '__name__', None) == 'CFO' and points_to_evaluate and (len(self._points_to_evaluate) > 1):
            self._candidate_start_points = {}
            self._started_from_low_cost = not low_cost_partial_config
        else:
            self._candidate_start_points = None
        (self._time_budget_s, self._num_samples) = (time_budget_s, num_samples)
        self._allow_empty_config = allow_empty_config
        if space is not None:
            self._init_search()

    def set_search_properties(self, metric: Optional[str]=None, mode: Optional[str]=None, config: Optional[Dict]=None, **spec) -> bool:
        if False:
            i = 10
            return i + 15
        metric_changed = mode_changed = False
        if metric and self._metric != metric:
            metric_changed = True
            self._metric = metric
            if self._metric_constraints:
                metric += self.lagrange
        if mode and self._mode != mode:
            mode_changed = True
            self._mode = mode
        if not self._ls.space:
            if self._gs is not None:
                self._gs.set_search_properties(metric, mode, config)
                self._gs.space = config
            if config:
                add_cost_to_space(config, self._ls.init_config, self._cat_hp_cost)
            self._ls.set_search_properties(metric, mode, config)
            self._init_search()
        elif metric_changed or mode_changed:
            self._ls.set_search_properties(metric, mode)
            if self._gs is not None:
                self._gs = GlobalSearch(space=self._gs._space, metric=metric, mode=mode, seed=self._gs_seed)
                self._gs.space = self._ls.space
            self._init_search()
        if spec:
            if 'time_budget_s' in spec:
                self._time_budget_s = spec['time_budget_s']
                now = time.time()
                self._time_used += now - self._start_time
                self._start_time = now
                self._set_deadline()
                if self._input_cost_attr == 'auto' and self._time_budget_s:
                    self.cost_attr = self._ls.cost_attr = TIME_TOTAL_S
            if 'metric_target' in spec:
                self._metric_target = spec.get('metric_target')
            num_samples = spec.get('num_samples')
            if num_samples is not None:
                self._num_samples = num_samples + len(self._result) + len(self._trial_proposed_by) if num_samples > 0 else num_samples
        return True

    def _set_deadline(self):
        if False:
            while True:
                i = 10
        if self._time_budget_s is not None:
            self._deadline = self._time_budget_s + self._start_time
            self._set_eps()
        else:
            self._deadline = np.inf

    def _set_eps(self):
        if False:
            i = 10
            return i + 15
        'set eps for search threads according to time budget'
        self._eps = max(min(self._time_budget_s / 1000.0, 1.0), 1e-09)

    def _init_search(self):
        if False:
            print('Hello World!')
        'initialize the search'
        self._start_time = time.time()
        self._time_used = 0
        self._set_deadline()
        self._is_ls_ever_converged = False
        self._subspace = {}
        self._metric_target = np.inf * self._ls.metric_op
        self._search_thread_pool = {0: SearchThread(self._ls.mode, self._gs, self.cost_attr, self._eps)}
        self._thread_count = 1
        self._init_used = self._ls.init_config is None
        self._trial_proposed_by = {}
        self._ls_bound_min = normalize(self._ls.init_config.copy(), self._ls.space, self._ls.init_config, {}, recursive=True)
        self._ls_bound_max = normalize(self._ls.init_config.copy(), self._ls.space, self._ls.init_config, {}, recursive=True)
        self._gs_admissible_min = self._ls_bound_min.copy()
        self._gs_admissible_max = self._ls_bound_max.copy()
        if self._metric_constraints:
            self._metric_constraint_satisfied = False
            self._metric_constraint_penalty = [self.penalty for _ in self._metric_constraints]
        else:
            self._metric_constraint_satisfied = True
            self._metric_constraint_penalty = None
        self.best_resource = self._ls.min_resource
        i = 0
        self._result = {}
        self._cost_used = 0
        while self._evaluated_rewards:
            trial_id = f'trial_for_evaluated_{i}'
            self.suggest(trial_id)
            i += 1

    def save(self, checkpoint_path: str):
        if False:
            i = 10
            return i + 15
        'save states to a checkpoint path.'
        self._time_used += time.time() - self._start_time
        self._start_time = time.time()
        save_object = self
        with open(checkpoint_path, 'wb') as outputFile:
            pickle.dump(save_object, outputFile)

    def restore(self, checkpoint_path: str):
        if False:
            print('Hello World!')
        'restore states from checkpoint.'
        with open(checkpoint_path, 'rb') as inputFile:
            state = pickle.load(inputFile)
        self.__dict__ = state.__dict__
        self._start_time = time.time()
        self._set_deadline()

    @property
    def metric_target(self):
        if False:
            print('Hello World!')
        return self._metric_target

    @property
    def is_ls_ever_converged(self):
        if False:
            while True:
                i = 10
        return self._is_ls_ever_converged

    def on_trial_complete(self, trial_id: str, result: Optional[Dict]=None, error: bool=False):
        if False:
            print('Hello World!')
        'search thread updater and cleaner.'
        metric_constraint_satisfied = True
        if result and (not error) and self._metric_constraints:
            objective = result[self._metric]
            for (i, constraint) in enumerate(self._metric_constraints):
                (metric_constraint, sign, threshold) = constraint
                value = result.get(metric_constraint)
                if value:
                    sign_op = 1 if sign == '<=' else -1
                    violation = (value - threshold) * sign_op
                    if violation > 0:
                        objective += self._metric_constraint_penalty[i] * violation * self._ls.metric_op
                        metric_constraint_satisfied = False
                        if self._metric_constraint_penalty[i] < self.penalty:
                            self._metric_constraint_penalty[i] += violation
            result[self._metric + self.lagrange] = objective
            if metric_constraint_satisfied and (not self._metric_constraint_satisfied):
                self._metric_constraint_penalty = [1 for _ in self._metric_constraints]
            self._metric_constraint_satisfied |= metric_constraint_satisfied
        thread_id = self._trial_proposed_by.get(trial_id)
        if thread_id in self._search_thread_pool:
            self._search_thread_pool[thread_id].on_trial_complete(trial_id, result, error)
            del self._trial_proposed_by[trial_id]
        if result:
            config = result.get('config', {})
            if not config:
                for (key, value) in result.items():
                    if key.startswith('config/'):
                        config[key[7:]] = value
            if self._allow_empty_config and (not config):
                return
            signature = self._ls.config_signature(config, self._subspace.get(trial_id, {}))
            if error:
                del self._result[signature]
            else:
                self._cost_used += result.get(self.cost_attr, 0)
                self._result[signature] = result
                objective = result[self._ls.metric]
                if (objective - self._metric_target) * self._ls.metric_op < 0:
                    self._metric_target = objective
                    if self._ls.resource:
                        self._best_resource = config[self._ls.resource_attr]
                if thread_id:
                    if not self._metric_constraint_satisfied:
                        self._expand_admissible_region(self._ls_bound_min, self._ls_bound_max, self._subspace.get(trial_id, self._ls.space))
                    if self._gs is not None and self._experimental and (not self._ls.hierarchical):
                        self._gs.add_evaluated_point(flatten_dict(config), objective)
                elif metric_constraint_satisfied and self._create_condition(result):
                    thread_id = self._thread_count
                    self._started_from_given = self._candidate_start_points and trial_id in self._candidate_start_points
                    if self._started_from_given:
                        del self._candidate_start_points[trial_id]
                    else:
                        self._started_from_low_cost = True
                    self._create_thread(config, result, self._subspace.get(trial_id, self._ls.space))
                self._gs_admissible_min.update(self._ls_bound_min)
                self._gs_admissible_max.update(self._ls_bound_max)
        if thread_id and thread_id in self._search_thread_pool:
            self._clean(thread_id)
        if trial_id in self._subspace and (not (self._candidate_start_points and trial_id in self._candidate_start_points)):
            del self._subspace[trial_id]

    def _create_thread(self, config, result, space):
        if False:
            while True:
                i = 10
        if self.lexico_objectives is None:
            obj = result[self._ls.metric]
        else:
            obj = {k: result[k] for k in self.lexico_objectives['metrics']}
        self._search_thread_pool[self._thread_count] = SearchThread(self._ls.mode, self._ls.create(config, obj, cost=result.get(self.cost_attr, 1), space=space), self.cost_attr, self._eps)
        self._thread_count += 1
        self._update_admissible_region(unflatten_dict(config), self._ls_bound_min, self._ls_bound_max, space, self._ls.space)

    def _update_admissible_region(self, config, admissible_min, admissible_max, subspace: Dict={}, space: Dict={}):
        if False:
            i = 10
            return i + 15
        normalized_config = normalize(config, subspace, config, {})
        for key in admissible_min:
            value = normalized_config[key]
            if isinstance(admissible_max[key], list):
                domain = space[key]
                choice = indexof(domain, value)
                self._update_admissible_region(value, admissible_min[key][choice], admissible_max[key][choice], subspace[key], domain[choice])
                if len(admissible_max[key]) > len(domain.categories):
                    normal = (choice + 0.5) / len(domain.categories)
                    admissible_max[key][-1] = max(normal, admissible_max[key][-1])
                    admissible_min[key][-1] = min(normal, admissible_min[key][-1])
            elif isinstance(value, dict):
                self._update_admissible_region(value, admissible_min[key], admissible_max[key], subspace[key], space[key])
            elif value > admissible_max[key]:
                admissible_max[key] = value
            elif value < admissible_min[key]:
                admissible_min[key] = value

    def _create_condition(self, result: Dict) -> bool:
        if False:
            print('Hello World!')
        'create thread condition'
        if len(self._search_thread_pool) < 2:
            return True
        obj_median = np.median([thread.obj_best1 for (id, thread) in self._search_thread_pool.items() if id])
        return result[self._ls.metric] * self._ls.metric_op < obj_median

    def _clean(self, thread_id: int):
        if False:
            print('Hello World!')
        'delete thread and increase admissible region if converged,\n        merge local threads if they are close\n        '
        assert thread_id
        todelete = set()
        for id in self._search_thread_pool:
            if id and id != thread_id:
                if self._inferior(id, thread_id):
                    todelete.add(id)
        for id in self._search_thread_pool:
            if id and id != thread_id:
                if self._inferior(thread_id, id):
                    todelete.add(thread_id)
                    break
        create_new = False
        if self._search_thread_pool[thread_id].converged:
            self._is_ls_ever_converged = True
            todelete.add(thread_id)
            self._expand_admissible_region(self._ls_bound_min, self._ls_bound_max, self._search_thread_pool[thread_id].space)
            if self._candidate_start_points:
                if not self._started_from_given:
                    obj = self._search_thread_pool[thread_id].obj_best1
                    worse = [trial_id for (trial_id, r) in self._candidate_start_points.items() if r and r[self._ls.metric] * self._ls.metric_op >= obj]
                    for trial_id in worse:
                        del self._candidate_start_points[trial_id]
                if self._candidate_start_points and self._started_from_low_cost:
                    create_new = True
        for id in todelete:
            del self._search_thread_pool[id]
        if create_new:
            self._create_thread_from_best_candidate()

    def _create_thread_from_best_candidate(self):
        if False:
            i = 10
            return i + 15
        best_trial_id = None
        obj_best = None
        for (trial_id, r) in self._candidate_start_points.items():
            if r and (best_trial_id is None or r[self._ls.metric] * self._ls.metric_op < obj_best):
                best_trial_id = trial_id
                obj_best = r[self._ls.metric] * self._ls.metric_op
        if best_trial_id:
            config = {}
            result = self._candidate_start_points[best_trial_id]
            for (key, value) in result.items():
                if key.startswith('config/'):
                    config[key[7:]] = value
            self._started_from_given = True
            del self._candidate_start_points[best_trial_id]
            self._create_thread(config, result, self._subspace.get(best_trial_id, self._ls.space))

    def _expand_admissible_region(self, lower, upper, space):
        if False:
            for i in range(10):
                print('nop')
        'expand the admissible region for the subspace `space`'
        for key in upper:
            ub = upper[key]
            if isinstance(ub, list):
                choice = space[key].get('_choice_')
                if choice:
                    self._expand_admissible_region(lower[key][choice], upper[key][choice], space[key])
            elif isinstance(ub, dict):
                self._expand_admissible_region(lower[key], ub, space[key])
            else:
                upper[key] += self._ls.STEPSIZE
                lower[key] -= self._ls.STEPSIZE

    def _inferior(self, id1: int, id2: int) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'whether thread id1 is inferior to id2'
        t1 = self._search_thread_pool[id1]
        t2 = self._search_thread_pool[id2]
        if t1.obj_best1 < t2.obj_best2:
            return False
        elif t1.resource and t1.resource < t2.resource:
            return False
        elif t2.reach(t1):
            return True
        return False

    def on_trial_result(self, trial_id: str, result: Dict):
        if False:
            i = 10
            return i + 15
        'receive intermediate result.'
        if trial_id not in self._trial_proposed_by:
            return
        thread_id = self._trial_proposed_by[trial_id]
        if thread_id not in self._search_thread_pool:
            return
        if result and self._metric_constraints:
            result[self._metric + self.lagrange] = result[self._metric]
        self._search_thread_pool[thread_id].on_trial_result(trial_id, result)

    def suggest(self, trial_id: str) -> Optional[Dict]:
        if False:
            for i in range(10):
                print('nop')
        'choose thread, suggest a valid config.'
        if self._init_used and (not self._points_to_evaluate):
            if self._cost_budget and self._cost_used >= self._cost_budget:
                return None
            (choice, backup) = self._select_thread()
            config = self._search_thread_pool[choice].suggest(trial_id)
            if not choice and config is not None and self._ls.resource:
                config[self._ls.resource_attr] = self.best_resource
            elif choice and config is None:
                if self._search_thread_pool[choice].converged:
                    self._expand_admissible_region(self._ls_bound_min, self._ls_bound_max, self._search_thread_pool[choice].space)
                    del self._search_thread_pool[choice]
                return
            space = self._search_thread_pool[choice].space
            skip = self._should_skip(choice, trial_id, config, space)
            use_rs = 0
            if skip:
                if choice:
                    return
                (config, space) = self._ls.complete_config({})
                skip = self._should_skip(-1, trial_id, config, space)
                if skip:
                    return
                use_rs = 1
            if choice or self._valid(config, self._ls.space, space, self._gs_admissible_min, self._gs_admissible_max):
                self._trial_proposed_by[trial_id] = choice
                self._search_thread_pool[choice].running += use_rs
            elif choice == backup:
                init_config = self._ls.init_config
                (config, space) = self._ls.complete_config(init_config, self._ls_bound_min, self._ls_bound_max)
                self._trial_proposed_by[trial_id] = choice
                self._search_thread_pool[choice].running += 1
            else:
                thread = self._search_thread_pool[backup]
                config = thread.suggest(trial_id)
                space = thread.space
                skip = self._should_skip(backup, trial_id, config, space)
                if skip:
                    return
                self._trial_proposed_by[trial_id] = backup
                choice = backup
            if not choice:
                self._update_admissible_region(config, self._gs_admissible_min, self._gs_admissible_max, space, self._ls.space)
            else:
                self._update_admissible_region(config, self._ls_bound_min, self._ls_bound_max, space, self._ls.space)
                self._gs_admissible_min.update(self._ls_bound_min)
                self._gs_admissible_max.update(self._ls_bound_max)
            signature = self._ls.config_signature(config, space)
            self._result[signature] = {}
            self._subspace[trial_id] = space
        else:
            if self._candidate_start_points is not None and self._points_to_evaluate:
                self._candidate_start_points[trial_id] = None
            reward = None
            if self._points_to_evaluate:
                init_config = self._points_to_evaluate.pop(0)
                if self._evaluated_rewards:
                    reward = self._evaluated_rewards.pop(0)
            else:
                init_config = self._ls.init_config
            if self._allow_empty_config and (not init_config):
                assert reward is None, "Empty config can't have reward."
                return init_config
            (config, space) = self._ls.complete_config(init_config, self._ls_bound_min, self._ls_bound_max)
            config_signature = self._ls.config_signature(config, space)
            if reward is None:
                result = self._result.get(config_signature)
                if result:
                    return
                elif result is None:
                    if self._violate_config_constriants(config, config_signature):
                        return
                    self._result[config_signature] = {}
                else:
                    return
            self._init_used = True
            self._trial_proposed_by[trial_id] = 0
            self._search_thread_pool[0].running += 1
            self._subspace[trial_id] = space
            if reward is not None:
                result = {self._metric: reward, self.cost_attr: 1, 'config': config}
                self.on_trial_complete(trial_id, result)
                return
        if self._use_incumbent_result_in_evaluation:
            if self._trial_proposed_by[trial_id] > 0:
                choice_thread = self._search_thread_pool[self._trial_proposed_by[trial_id]]
                config[INCUMBENT_RESULT] = choice_thread.best_result
        return config

    def _violate_config_constriants(self, config, config_signature):
        if False:
            for i in range(10):
                print('nop')
        'check if config violates config constraints.\n        If so, set the result to worst and return True.\n        '
        if not self._config_constraints:
            return False
        for constraint in self._config_constraints:
            (func, sign, threshold) = constraint
            value = func(config)
            if sign == '<=' and value > threshold or (sign == '>=' and value < threshold) or (sign == '>' and value <= threshold) or (sign == '<' and value > threshold):
                self._result[config_signature] = {self._metric: np.inf * self._ls.metric_op, 'time_total_s': 1}
                return True
        return False

    def _should_skip(self, choice, trial_id, config, space) -> bool:
        if False:
            return 10
        "if config is None or config's result is known or constraints are violated\n        return True; o.w. return False\n        "
        if config is None:
            return True
        config_signature = self._ls.config_signature(config, space)
        exists = config_signature in self._result
        if not exists:
            exists = self._violate_config_constriants(config, config_signature)
        if exists:
            if choice >= 0:
                result = self._result.get(config_signature)
                if result:
                    self._search_thread_pool[choice].on_trial_complete(trial_id, result, error=False)
                    if choice:
                        self._clean(choice)
            return True
        return False

    def _select_thread(self) -> Tuple:
        if False:
            while True:
                i = 10
        'thread selector; use can_suggest to check LS availability'
        min_eci = np.inf
        if self.cost_attr == TIME_TOTAL_S:
            now = time.time()
            min_eci = self._deadline - now
            if min_eci <= 0:
                min_eci = 0
            elif self._num_samples and self._num_samples > 0:
                num_finished = len(self._result)
                num_proposed = num_finished + len(self._trial_proposed_by)
                num_left = max(self._num_samples - num_proposed, 0)
                if num_proposed > 0:
                    time_used = now - self._start_time + self._time_used
                    min_eci = min(min_eci, time_used / num_finished * num_left)
        elif self.cost_attr is not None and self._cost_budget:
            min_eci = max(self._cost_budget - self._cost_used, 0)
        elif self._num_samples and self._num_samples > 0:
            num_finished = len(self._result)
            num_proposed = num_finished + len(self._trial_proposed_by)
            min_eci = max(self._num_samples - num_proposed, 0)
        max_speed = 0
        for thread in self._search_thread_pool.values():
            if thread.speed > max_speed:
                max_speed = thread.speed
        for thread in self._search_thread_pool.values():
            thread.update_eci(self._metric_target, max_speed)
            if thread.eci < min_eci:
                min_eci = thread.eci
        for thread in self._search_thread_pool.values():
            thread.update_priority(min_eci)
        top_thread_id = backup_thread_id = 0
        priority1 = priority2 = self._search_thread_pool[0].priority
        for (thread_id, thread) in self._search_thread_pool.items():
            if thread_id and thread.can_suggest:
                priority = thread.priority
                if priority > priority1:
                    priority1 = priority
                    top_thread_id = thread_id
                if priority > priority2 or backup_thread_id == 0:
                    priority2 = priority
                    backup_thread_id = thread_id
        return (top_thread_id, backup_thread_id)

    def _valid(self, config: Dict, space: Dict, subspace: Dict, lower: Dict, upper: Dict) -> bool:
        if False:
            print('Hello World!')
        'config validator'
        normalized_config = normalize(config, subspace, config, {})
        for (key, lb) in lower.items():
            if key in config:
                value = normalized_config[key]
                if isinstance(lb, list):
                    domain = space[key]
                    index = indexof(domain, value)
                    nestedspace = subspace[key]
                    lb = lb[index]
                    ub = upper[key][index]
                elif isinstance(lb, dict):
                    nestedspace = subspace[key]
                    domain = space[key]
                    ub = upper[key]
                else:
                    nestedspace = None
                if nestedspace:
                    valid = self._valid(value, domain, nestedspace, lb, ub)
                    if not valid:
                        return False
                elif value + self._ls.STEPSIZE < lower[key] or value > upper[key] + self._ls.STEPSIZE:
                    return False
        return True

    @property
    def results(self) -> List[Dict]:
        if False:
            print('Hello World!')
        'A list of dicts of results for each evaluated configuration.\n\n        Each dict has "config" and metric names as keys.\n        The returned dict includes the initial results provided via `evaluated_reward`.\n        '
        return [x for x in getattr(self, '_result', {}).values() if x]
try:
    from ray import __version__ as ray_version
    assert ray_version >= '1.10.0'
    from ray.tune import uniform, quniform, choice, randint, qrandint, randn, qrandn, loguniform, qloguniform
except (ImportError, AssertionError):
    from ..sample import uniform, quniform, choice, randint, qrandint, randn, qrandn, loguniform, qloguniform
try:
    from nni.tuner import Tuner as NNITuner
    from nni.utils import extract_scalar_reward
except ImportError:
    NNITuner = object

    def extract_scalar_reward(x: Dict):
        if False:
            while True:
                i = 10
        return x.get('default')

class BlendSearchTuner(BlendSearch, NNITuner):
    """Tuner class for NNI."""

    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        if False:
            i = 10
            return i + 15
        "Receive trial's final result.\n\n        Args:\n            parameter_id: int.\n            parameters: object created by `generate_parameters()`.\n            value: final metrics of the trial, including default metric.\n        "
        result = {'config': parameters, self._metric: extract_scalar_reward(value), self.cost_attr: 1 if isinstance(value, float) else value.get(self.cost_attr, value.get('sequence', 1))}
        self.on_trial_complete(str(parameter_id), result)
    ...

    def generate_parameters(self, parameter_id, **kwargs) -> Dict:
        if False:
            while True:
                i = 10
        'Returns a set of trial (hyper-)parameters, as a serializable object.\n\n        Args:\n            parameter_id: int.\n        '
        return self.suggest(str(parameter_id))
    ...

    def update_search_space(self, search_space):
        if False:
            return 10
        'Required by NNI.\n\n        Tuners are advised to support updating search space at run-time.\n        If a tuner can only set search space once before generating first hyper-parameters,\n        it should explicitly document this behaviour.\n\n        Args:\n            search_space: JSON object created by experiment owner.\n        '
        config = {}
        for (key, value) in search_space.items():
            v = value.get('_value')
            _type = value['_type']
            if _type == 'choice':
                config[key] = choice(v)
            elif _type == 'randint':
                config[key] = randint(*v)
            elif _type == 'uniform':
                config[key] = uniform(*v)
            elif _type == 'quniform':
                config[key] = quniform(*v)
            elif _type == 'loguniform':
                config[key] = loguniform(*v)
            elif _type == 'qloguniform':
                config[key] = qloguniform(*v)
            elif _type == 'normal':
                config[key] = randn(*v)
            elif _type == 'qnormal':
                config[key] = qrandn(*v)
            else:
                raise ValueError(f'unsupported type in search_space {_type}')
        init_config = self._ls.init_config
        add_cost_to_space(config, init_config, self._cat_hp_cost)
        self._ls = self.LocalSearch(init_config, self._ls.metric, self._mode, config, self._ls.resource_attr, self._ls.min_resource, self._ls.max_resource, self._ls.resource_multiple_factor, cost_attr=self.cost_attr, seed=self._ls.seed, lexico_objectives=self.lexico_objectives)
        if self._gs is not None:
            self._gs = GlobalSearch(space=config, metric=self._metric, mode=self._mode, sampler=self._gs._sampler)
            self._gs.space = config
        self._init_search()

class CFO(BlendSearchTuner):
    """class for CFO algorithm."""
    __name__ = 'CFO'

    def suggest(self, trial_id: str) -> Optional[Dict]:
        if False:
            while True:
                i = 10
        assert len(self._search_thread_pool) < 3, len(self._search_thread_pool)
        if len(self._search_thread_pool) < 2:
            self._init_used = False
        return super().suggest(trial_id)

    def _select_thread(self) -> Tuple:
        if False:
            while True:
                i = 10
        for key in self._search_thread_pool:
            if key:
                return (key, key)

    def _create_condition(self, result: Dict) -> bool:
        if False:
            while True:
                i = 10
        'create thread condition'
        if self._points_to_evaluate:
            return False
        if len(self._search_thread_pool) == 2:
            return False
        if self._candidate_start_points and self._thread_count == 1:
            obj_best = min((self._ls.metric_op * r[self._ls.metric] for r in self._candidate_start_points.values() if r), default=-np.inf)
            return result[self._ls.metric] * self._ls.metric_op <= obj_best
        else:
            return True

    def on_trial_complete(self, trial_id: str, result: Optional[Dict]=None, error: bool=False):
        if False:
            for i in range(10):
                print('nop')
        super().on_trial_complete(trial_id, result, error)
        if self._candidate_start_points and trial_id in self._candidate_start_points:
            self._candidate_start_points[trial_id] = result
            if len(self._search_thread_pool) < 2 and (not self._points_to_evaluate):
                self._create_thread_from_best_candidate()

class RandomSearch(CFO):
    """Class for random search."""

    def suggest(self, trial_id: str) -> Optional[Dict]:
        if False:
            for i in range(10):
                print('nop')
        if self._points_to_evaluate:
            return super().suggest(trial_id)
        (config, _) = self._ls.complete_config({})
        return config

    def on_trial_complete(self, trial_id: str, result: Optional[Dict]=None, error: bool=False):
        if False:
            while True:
                i = 10
        return

    def on_trial_result(self, trial_id: str, result: Dict):
        if False:
            print('Hello World!')
        return