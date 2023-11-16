import copy
import itertools
import logging
from typing import Callable, Optional
from torch.utils._triton import has_triton
from .utils import red_text, triton_config_to_hashable
if has_triton():
    import triton
else:
    triton = None
from . import config as inductor_config
log = logging.getLogger(__name__)

def get_field(config, name):
    if False:
        i = 10
        return i + 15
    if name == 'num_warps':
        return config.num_warps
    elif name == 'num_stages':
        return config.num_stages
    else:
        return config.kwargs.get(name, None)

def set_field(config, name, value):
    if False:
        while True:
            i = 10
    if name == 'num_warps':
        config.num_warps = value
    elif name == 'num_stages':
        config.num_stages = value
    else:
        config.kwargs[name] = value

class CoordescTuner:
    """
    The coordinate descent tuner. Tune one field/coordinate at a time.

    TODO will it be necessary to tune multiple fields simultaneously.


    TODO: what if both increasing and decreasing a field can improve perf.
          i.e., there are multiple local optima..
    """

    def __init__(self, is_mm=False, name='unknown', size_hints=None):
        if False:
            while True:
                i = 10
        self.is_mm = is_mm
        self.cached_benchmark_results = {}
        self.name = name
        self.size_hints = size_hints

    def get_xmax(self):
        if False:
            while True:
                i = 10
        xmax = inductor_config.triton.max_block['X']
        if self.size_hints and len(self.size_hints) > 0:
            xmax = min(xmax, self.size_hints[0])
        return xmax

    def get_ymax(self):
        if False:
            for i in range(10):
                print('nop')
        ymax = inductor_config.triton.max_block['Y']
        if self.size_hints and len(self.size_hints) > 1:
            ymax = min(ymax, self.size_hints[1])
        return ymax

    def get_zmax(self):
        if False:
            print('Hello World!')
        zmax = inductor_config.triton.max_block['Z']
        if self.size_hints and len(self.size_hints) > 2:
            zmax = min(zmax, self.size_hints[2])
        return zmax

    def get_rmax(self):
        if False:
            return 10
        if self.size_hints and len(self.size_hints) > 0:
            return self.size_hints[-1]
        else:
            return 2 ** 30

    def get_warpsmax(self):
        if False:
            while True:
                i = 10
        return 1024 // 32

    def cache_benchmark_result(self, config, timing):
        if False:
            return 10
        self.cached_benchmark_results[triton_config_to_hashable(config)] = timing

    def lookup_in_cache(self, config):
        if False:
            return 10
        return self.cached_benchmark_results.get(triton_config_to_hashable(config))

    def call_func(self, func, config):
        if False:
            i = 10
            return i + 15
        found = self.lookup_in_cache(config)
        if found is not None:
            log.debug('  CACHED')
            return found
        timing = func(config)
        self.cache_benchmark_result(config, timing)
        return timing

    @property
    def tunable_fields(self):
        if False:
            for i in range(10):
                print('nop')
        out = ['XBLOCK', 'YBLOCK', 'ZBLOCK', 'RBLOCK', 'BLOCK_M', 'BLOCK_N', 'BLOCK_K', 'num_warps']
        if self.is_mm:
            out.append('num_stages')
        return out

    def value_too_large(self, name, val):
        if False:
            i = 10
            return i + 15
        if name == 'XBLOCK':
            return val > self.get_xmax()
        if name == 'YBLOCK':
            return val > self.get_ymax()
        if name == 'ZBLOCK':
            return val > self.get_zmax()
        if name == 'RBLOCK':
            return val > self.get_rmax()
        if name == 'num_warps':
            return val > self.get_warpsmax()
        return False

    def get_neighbour_values(self, name, orig_val, radius=1, include_self=False):
        if False:
            print('Hello World!')
        "\n        Get neighbour values in 'radius' steps. The original value is not\n        returned as it's own neighbour.\n        "
        assert radius >= 1

        def update(cur_val, inc=True):
            if False:
                while True:
                    i = 10
            if name == 'num_stages':
                if inc:
                    return cur_val + 1
                else:
                    return cur_val - 1
            elif inc:
                return cur_val * 2
            else:
                return cur_val // 2
        out = []
        cur_val = orig_val
        for _ in range(radius):
            cur_val = update(cur_val, True)
            if self.value_too_large(name, cur_val):
                break
            out.append(cur_val)
        cur_val = orig_val
        for _ in range(radius):
            cur_val = update(cur_val, False)
            if cur_val <= 0:
                break
            out.append(cur_val)
        if include_self:
            out.append(orig_val)
        return out

    @staticmethod
    def has_improvement(baseline, test):
        if False:
            return 10
        threshold = 0.001
        return test is not None and test < baseline * (1 - threshold)

    def check_all_tuning_directions(self, func: Callable[['triton.Config'], float], best_config, best_timing):
        if False:
            while True:
                i = 10
        '\n        Check all directions. We only do this once the regular coordinate\n        descent tuning find no better choices any more.\n        We only have a few tunable fields, so this should be fine.\n        '
        candidate_values_list = []
        effective_fields = []
        for field in self.tunable_fields:
            old_value = get_field(best_config, field)
            if old_value is None:
                continue
            candidate_values = self.get_neighbour_values(field, old_value, radius=inductor_config.coordinate_descent_search_radius, include_self=True)
            candidate_values_list.append(candidate_values)
            effective_fields.append(field)
        choices = itertools.product(*candidate_values_list)
        improved = False
        for choice in choices:
            assert len(choice) == len(effective_fields)
            candidate_config = copy.deepcopy(best_config)
            for (new_val, field) in zip(choice, effective_fields):
                set_field(candidate_config, field, new_val)
            (cmp_res, candidate_timing) = self.compare_config(func, candidate_config, best_config, best_timing)
            if cmp_res:
                improved = True
                best_config = candidate_config
                best_timing = candidate_timing
        return (improved, best_config, best_timing)

    def compare_config(self, func, candidate_config, best_config, best_timing):
        if False:
            i = 10
            return i + 15
        '\n        Check if candidate_config is better than best_config.\n\n        Return a touple of (compare_result, candidate_timing).\n        compare_result is true iff candidate_config is better.\n        '
        log.debug('Try config %s', candidate_config)
        try:
            candidate_timing = self.call_func(func, candidate_config)
        except Exception as e:
            log.debug('Got exception %s', e)
            return (False, float('inf'))
        if self.has_improvement(best_timing, candidate_timing):
            log.debug('Tune from %s %f -> %s %f', best_config, best_timing, candidate_config, candidate_timing)
            return (True, candidate_timing)
        return (False, candidate_timing)

    def autotune(self, func: Callable[['triton.Config'], float], baseline_config: 'triton.Config', baseline_timing: Optional[float]=None) -> 'triton.Config':
        if False:
            while True:
                i = 10
        if baseline_timing is None:
            baseline_timing = self.call_func(func, baseline_config)
        log.debug('= Do coordinate descent tuning for %s =', self.name)
        log.debug('Baseline Config %s, baseline timing %f', baseline_config, baseline_timing)
        improved = True
        best_config = baseline_config
        best_timing = baseline_timing
        tunable_fields = self.tunable_fields
        while improved:
            improved = False
            for name in tunable_fields:
                cur_val = get_field(best_config, name)
                if cur_val is None:
                    continue
                candidate_values = self.get_neighbour_values(name, cur_val)
                for next_val in candidate_values:
                    candidate_config = copy.deepcopy(best_config)
                    set_field(candidate_config, name, next_val)
                    (cmp_res, candidate_timing) = self.compare_config(func, candidate_config, best_config, best_timing)
                    if cmp_res:
                        improved = True
                        (best_config, best_timing) = (candidate_config, candidate_timing)
            if not improved and inductor_config.coordinate_descent_check_all_directions:
                old_best_timing = best_timing
                (improved, best_config, best_timing) = self.check_all_tuning_directions(func, best_config, best_timing)
                if improved:
                    msg = red_text('Coordinate descend tuning found improvement of %.3fx by looking in all directions.')
                    log.debug(msg, old_best_timing / best_timing)
        log.debug('Improve from %s %f -> %s %f, %.3fx', baseline_config, baseline_timing, best_config, best_timing, baseline_timing / best_timing)
        return best_config