import contextlib
import threading
from typing import TYPE_CHECKING
from typing import Dict
from typing import NamedTuple
from typing import Optional
from typing import Union
from apache_beam.runners import common
from apache_beam.utils.counters import Counter
from apache_beam.utils.counters import CounterFactory
from apache_beam.utils.counters import CounterName
try:
    from apache_beam.runners.worker import statesampler_fast as statesampler_impl
    FAST_SAMPLER = True
except ImportError:
    from apache_beam.runners.worker import statesampler_slow as statesampler_impl
    FAST_SAMPLER = False
if TYPE_CHECKING:
    from apache_beam.metrics.execution import MetricsContainer
_STATE_SAMPLERS = threading.local()

def set_current_tracker(tracker):
    if False:
        i = 10
        return i + 15
    _STATE_SAMPLERS.tracker = tracker

def get_current_tracker():
    if False:
        for i in range(10):
            print('nop')
    try:
        return _STATE_SAMPLERS.tracker
    except AttributeError:
        return None
_INSTRUCTION_IDS = threading.local()

def get_current_instruction_id():
    if False:
        while True:
            i = 10
    try:
        return _INSTRUCTION_IDS.instruction_id
    except AttributeError:
        return None

@contextlib.contextmanager
def instruction_id(id):
    if False:
        for i in range(10):
            print('nop')
    try:
        _INSTRUCTION_IDS.instruction_id = id
        yield
    finally:
        _INSTRUCTION_IDS.instruction_id = None

def for_test():
    if False:
        i = 10
        return i + 15
    set_current_tracker(StateSampler('test', CounterFactory()))
    return get_current_tracker()
StateSamplerInfo = NamedTuple('StateSamplerInfo', [('state_name', CounterName), ('transition_count', int), ('time_since_transition', int), ('tracked_thread', Optional[threading.Thread])])
DEFAULT_SAMPLING_PERIOD_MS = 200

class StateSampler(statesampler_impl.StateSampler):

    def __init__(self, prefix, counter_factory, sampling_period_ms=DEFAULT_SAMPLING_PERIOD_MS):
        if False:
            while True:
                i = 10
        self._prefix = prefix
        self._counter_factory = counter_factory
        self._states_by_name = {}
        self.sampling_period_ms = sampling_period_ms
        self.tracked_thread = None
        self.finished = False
        self.started = False
        super().__init__(sampling_period_ms)

    @property
    def stage_name(self):
        if False:
            while True:
                i = 10
        return self._prefix

    def stop(self):
        if False:
            print('Hello World!')
        set_current_tracker(None)
        super().stop()

    def stop_if_still_running(self):
        if False:
            print('Hello World!')
        if self.started and (not self.finished):
            self.stop()

    def start(self):
        if False:
            for i in range(10):
                print('nop')
        self.tracked_thread = threading.current_thread()
        set_current_tracker(self)
        super().start()
        self.started = True

    def get_info(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns StateSamplerInfo with transition statistics.'
        return StateSamplerInfo(self.current_state().name, self.state_transition_count, self.time_since_transition, self.tracked_thread)

    def scoped_state(self, name_context, state_name, io_target=None, metrics_container=None):
        if False:
            i = 10
            return i + 15
        "Returns a ScopedState object associated to a Step and a State.\n\n    Args:\n      name_context: common.NameContext. It is the step name information.\n      state_name: str. It is the state name (e.g. process / start / finish).\n      io_target:\n      metrics_container: MetricsContainer. The step's metrics container.\n\n    Returns:\n      A ScopedState that keeps the execution context and is able to switch it\n      for the execution thread.\n    "
        if not isinstance(name_context, common.NameContext):
            name_context = common.NameContext(name_context)
        counter_name = CounterName(state_name + '-msecs', stage_name=self._prefix, step_name=name_context.metrics_name(), io_target=io_target)
        if counter_name in self._states_by_name:
            return self._states_by_name[counter_name]
        else:
            output_counter = self._counter_factory.get_counter(counter_name, Counter.SUM)
            self._states_by_name[counter_name] = super()._scoped_state(counter_name, name_context, output_counter, metrics_container)
            return self._states_by_name[counter_name]

    def commit_counters(self):
        if False:
            while True:
                i = 10
        'Updates output counters with latest state statistics.'
        for state in self._states_by_name.values():
            state_msecs = int(1e-06 * state.nsecs)
            state.counter.update(state_msecs - state.counter.value())