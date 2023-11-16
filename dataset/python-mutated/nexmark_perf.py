"""
performance summary for a run of nexmark query
"""

class NexmarkPerf(object):

    def __init__(self, runtime_sec=None, event_count=None, event_per_sec=None, result_count=None):
        if False:
            while True:
                i = 10
        self.runtime_sec = runtime_sec if runtime_sec else -1.0
        self.event_count = event_count if event_count else -1
        self.event_per_sec = event_per_sec if event_per_sec else -1.0
        self.result_count = result_count if result_count else -1

    def has_progress(self, previous_perf):
        if False:
            return 10
        '\n    Args:\n      previous_perf: a NexmarkPerf object to be compared to self\n\n    Returns:\n      True if there are observed pipeline activity between self and other\n        NexmarkPerf values\n    '
        if self.runtime_sec != previous_perf.runtime_sec or self.event_count != previous_perf.event_count or self.result_count != previous_perf.result_count:
            return True
        return False