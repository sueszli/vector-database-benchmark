from __future__ import print_function, absolute_import, division
import sys
import traceback
from greenlet import settrace
from greenlet import getcurrent
from gevent.util import format_run_info
from gevent._compat import perf_counter
from gevent._util import gmctime
__all__ = ['GreenletTracer', 'HubSwitchTracer', 'MaxSwitchTracer']

class GreenletTracer(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.greenlet_switch_counter = 0
        self.active_greenlet = None
        prev_trace = settrace(self)
        self.previous_trace_function = prev_trace
        self._killed = False

    def kill(self):
        if False:
            while True:
                i = 10
        if not self._killed:
            self._killed = True
            settrace(self.previous_trace_function)
            self.previous_trace_function = None

    def _trace(self, event, args):
        if False:
            while True:
                i = 10
        self.greenlet_switch_counter += 1
        if event in ('switch', 'throw'):
            self.active_greenlet = args[1]
        else:
            self.active_greenlet = None
        if self.previous_trace_function is not None:
            self.previous_trace_function(event, args)

    def __call__(self, event, args):
        if False:
            print('Hello World!')
        return self._trace(event, args)

    def did_block_hub(self, hub):
        if False:
            print('Hello World!')
        active_greenlet = self.active_greenlet
        did_switch = self.greenlet_switch_counter != 0
        self.greenlet_switch_counter = 0
        if did_switch or active_greenlet is None or active_greenlet is hub:
            return False
        return (True, active_greenlet)

    def ignore_current_greenlet_blocking(self):
        if False:
            while True:
                i = 10
        self.active_greenlet = None

    def monitor_current_greenlet_blocking(self):
        if False:
            while True:
                i = 10
        self.active_greenlet = getcurrent()

    def did_block_hub_report(self, hub, active_greenlet, format_kwargs):
        if False:
            while True:
                i = 10
        report = ['=' * 80, '\n%s : Greenlet %s appears to be blocked' % (gmctime(), str(active_greenlet))]
        report.append('    Reported by %s' % (self,))
        try:
            frame = sys._current_frames()[hub.thread_ident]
        except KeyError:
            stack = ['Unknown: No thread found for hub %r\n' % (hub,)]
        else:
            stack = traceback.format_stack(frame)
        report.append('Blocked Stack (for thread id %s):' % (hex(hub.thread_ident),))
        report.append(''.join(stack))
        report.append('Info:')
        report.extend(format_run_info(**format_kwargs))
        return report

class _HubTracer(GreenletTracer):

    def __init__(self, hub, max_blocking_time):
        if False:
            print('Hello World!')
        GreenletTracer.__init__(self)
        self.max_blocking_time = max_blocking_time
        self.hub = hub

    def kill(self):
        if False:
            print('Hello World!')
        self.hub = None
        GreenletTracer.kill(self)

class HubSwitchTracer(_HubTracer):

    def __init__(self, hub, max_blocking_time):
        if False:
            for i in range(10):
                print('nop')
        _HubTracer.__init__(self, hub, max_blocking_time)
        self.last_entered_hub = 0

    def _trace(self, event, args):
        if False:
            return 10
        GreenletTracer._trace(self, event, args)
        if self.active_greenlet is self.hub:
            self.last_entered_hub = perf_counter()

    def did_block_hub(self, hub):
        if False:
            i = 10
            return i + 15
        if perf_counter() - self.last_entered_hub > self.max_blocking_time:
            return (True, self.active_greenlet)

class MaxSwitchTracer(_HubTracer):

    def __init__(self, hub, max_blocking_time):
        if False:
            while True:
                i = 10
        _HubTracer.__init__(self, hub, max_blocking_time)
        self.last_switch = perf_counter()
        self.max_blocking = 0

    def _trace(self, event, args):
        if False:
            for i in range(10):
                print('nop')
        old_active = self.active_greenlet
        GreenletTracer._trace(self, event, args)
        if old_active is not self.hub and old_active is not None:
            switched_at = perf_counter()
            self.max_blocking = max(self.max_blocking, switched_at - self.last_switch)

    def did_block_hub(self, hub):
        if False:
            while True:
                i = 10
        if self.max_blocking == 0:
            self.max_blocking = perf_counter() - self.last_switch
        if self.max_blocking > self.max_blocking_time:
            return (True, self.active_greenlet)
from gevent._util import import_c_accel
import_c_accel(globals(), 'gevent.__tracer')