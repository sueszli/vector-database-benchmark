from typing import Optional
from apache_beam.runners import common
from apache_beam.utils import counters

class StateSampler(object):

    def __init__(self, sampling_period_ms):
        if False:
            i = 10
            return i + 15
        self._state_stack = [ScopedState(self, counters.CounterName('unknown'), None)]
        self.state_transition_count = 0
        self.time_since_transition = 0

    def current_state(self):
        if False:
            print('Hello World!')
        'Returns the current execution state.\n\n    This operation is not thread safe, and should only be called from the\n    execution thread.'
        return self._state_stack[-1]

    def _scoped_state(self, counter_name, name_context, output_counter, metrics_container=None):
        if False:
            while True:
                i = 10
        assert isinstance(name_context, common.NameContext)
        return ScopedState(self, counter_name, name_context, output_counter, metrics_container)

    def update_metric(self, typed_metric_name, value):
        if False:
            while True:
                i = 10
        metrics_container = self.current_state().metrics_container
        if metrics_container is not None:
            metrics_container.get_metric_cell(typed_metric_name).update(value)

    def _enter_state(self, state):
        if False:
            i = 10
            return i + 15
        self.state_transition_count += 1
        self._state_stack.append(state)

    def _exit_state(self):
        if False:
            while True:
                i = 10
        self.state_transition_count += 1
        self._state_stack.pop()

    def start(self):
        if False:
            print('Hello World!')
        pass

    def stop(self):
        if False:
            print('Hello World!')
        pass

    def reset(self):
        if False:
            return 10
        pass

class ScopedState(object):

    def __init__(self, sampler, name, step_name_context, counter=None, metrics_container=None):
        if False:
            print('Hello World!')
        self.state_sampler = sampler
        self.name = name
        self.name_context = step_name_context
        self.counter = counter
        self.nsecs = 0
        self.metrics_container = metrics_container

    def sampled_seconds(self):
        if False:
            return 10
        return 1e-09 * self.nsecs

    def sampled_msecs_int(self):
        if False:
            i = 10
            return i + 15
        return int(1e-06 * self.nsecs)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return 'ScopedState[%s, %s]' % (self.name, self.nsecs)

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        self.state_sampler._enter_state(self)

    def __exit__(self, exc_type, exc_value, traceback):
        if False:
            while True:
                i = 10
        self.state_sampler._exit_state()