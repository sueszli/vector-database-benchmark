import json

class ChromeTraceFormatter:

    def __init__(self):
        if False:
            print('Hello World!')
        self._events = []
        self._metadata = []

    def _create_event(self, ph, category, name, pid, tid, timestamp):
        if False:
            while True:
                i = 10
        'Creates a new Chrome Trace event.\n\n        For details of the file format, see:\n        https://github.com/catapult-project/catapult/blob/master/tracing/README.md\n\n        Args:\n          ph:  The type of event - usually a single character.\n          category: The event category as a string.\n          name:  The event name as a string.\n          pid:  Identifier of the process generating this event as an integer.\n          tid:  Identifier of the thread generating this event as an integer.\n          timestamp:  The timestamp of this event as a long integer.\n\n        Returns:\n          A JSON compatible event object.\n        '
        event = {}
        event['ph'] = ph
        event['cat'] = category
        event['name'] = name
        event['pid'] = pid
        event['tid'] = tid
        event['ts'] = timestamp
        return event

    def emit_pid(self, name, pid):
        if False:
            for i in range(10):
                print('nop')
        'Adds a process metadata event to the trace.\n\n        Args:\n          name:  The process name as a string.\n          pid:  Identifier of the process as an integer.\n        '
        event = {}
        event['name'] = 'process_name'
        event['ph'] = 'M'
        event['pid'] = pid
        event['args'] = {'name': name}
        self._metadata.append(event)

    def emit_region(self, timestamp, duration, pid, tid, category, name, args):
        if False:
            i = 10
            return i + 15
        'Adds a region event to the trace.\n\n        Args:\n          timestamp:  The start timestamp of this region as a long integer.\n          duration:  The duration of this region as a long integer.\n          pid:  Identifier of the process generating this event as an integer.\n          tid:  Identifier of the thread generating this event as an integer.\n          category: The event category as a string.\n          name:  The event name as a string.\n          args:  A JSON-compatible dictionary of event arguments.\n        '
        event = self._create_event('X', category, name, pid, tid, timestamp)
        event['dur'] = duration
        event['args'] = args
        self._events.append(event)

    def emit_counter(self, category, name, pid, timestamp, counter, value):
        if False:
            print('Hello World!')
        'Emits a record for a single counter.\n\n        Args:\n            category: The event category as string\n            name: The event name as string\n            pid: Identifier of the process generating this event as integer\n            timestamp: The timestamps of this event as long integer\n            counter: Name of the counter as string\n            value: Value of the counter as integer\n            tid: Thread id of the allocation as integer\n        '
        event = self._create_event('C', category, name, pid, 0, timestamp)
        event['args'] = {counter: value}
        self._events.append(event)

    def format_to_string(self, pretty=False):
        if False:
            for i in range(10):
                print('nop')
        'Formats the chrome trace to a string.\n\n        Args:\n          pretty: (Optional.)  If True, produce human-readable JSON output.\n\n        Returns:\n          A JSON-formatted string in Chrome Trace format.\n        '
        trace = {}
        trace['traceEvents'] = self._metadata + self._events
        if pretty:
            return json.dumps(trace, indent=4, separators=(',', ': '))
        else:
            return json.dumps(trace, separators=(',', ':'))

    def clear(self):
        if False:
            for i in range(10):
                print('nop')
        self._events = []
        self._metadata = []
if __name__ == '__main__':
    pass