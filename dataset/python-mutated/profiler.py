"""
Profiling utilities
"""
import cProfile
import io
import pstats
import tracemalloc

class Profiler:
    """
    A class for quick and easy profiling.
    Usage:
        p = Profiler()
        with p:
            # call methods that need to be profiled here
        print(p.report())

    The 'with' statement can be replaced with calls to
    p.enable() and p.disable().
    """
    profile: cProfile.Profile = None
    profile_stats: pstats.Stats = None
    profile_stream = None

    def __init__(self, o_stream=None):
        if False:
            return 10
        self.profile = cProfile.Profile()
        self.profile_stream = o_stream

    def __enter__(self):
        if False:
            while True:
                i = 10
        '\n        Activate data collection.\n        '
        self.enable()

    def __exit__(self, exc_type, exc_value, traceback):
        if False:
            return 10
        '\n        Stop profiling.\n        '
        self.disable()

    def write_report(self, sortby: str='calls') -> None:
        if False:
            return 10
        "\n        Write the profile stats to profile_stream's file.\n        "
        self.profile_stats = pstats.Stats(self.profile, stream=self.profile_stream)
        self.profile_stats.sort_stats(sortby)
        self.profile_stats.print_stats()

    def report(self, sortby: str='calls'):
        if False:
            while True:
                i = 10
        '\n        Return the profile_stats to the console.\n        '
        self.profile_stats = pstats.Stats(self.profile, stream=io.StringIO())
        self.profile_stats.sort_stats(sortby)
        self.profile_stats.print_stats()
        return self.profile_stats.stream.getvalue()

    def enable(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Begins profiling calls.\n        '
        self.profile.enable()

    def disable(self):
        if False:
            print('Hello World!')
        '\n        Stop profiling calls.\n        '
        self.profile.disable()

class Tracemalloc:
    """
    A class for memory profiling.
    Usage:
        p = Tracemalloc()
        with p:
            # call methods that need to be profiled here
        print(p.report())

    The 'with' statement can be replaced with calls to
    p.enable() and p.disable().
    """
    snapshot0 = None
    snapshot1 = None
    peak = None

    def __init__(self, o_stream=None):
        if False:
            while True:
                i = 10
        self.tracemalloc_stream = o_stream

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        '\n        Activate data collection.\n        '
        self.enable()

    def __exit__(self, exc_type, exc_value, traceback):
        if False:
            i = 10
            return i + 15
        '\n        Stop profiling.\n        '
        self.disable()

    def snapshot(self):
        if False:
            i = 10
            return i + 15
        '\n        Take a manual snapshot. Up to two snapshots can be saved.\n        report() compares the last two snapshots.\n        '
        if self.snapshot0 is None:
            self.snapshot0 = tracemalloc.take_snapshot()
        elif self.snapshot1 is None:
            self.snapshot1 = tracemalloc.take_snapshot()
        else:
            self.snapshot0 = self.snapshot1
            self.snapshot1 = tracemalloc.take_snapshot()

    def report(self, sortby: str='lineno', cumulative: bool=False, limit: int=100) -> None:
        if False:
            print('Hello World!')
        '\n        Return the snapshot statistics to the console.\n        '
        if self.snapshot1:
            stats = self.snapshot1.compare_to(self.snapshot0, sortby, cumulative)[:limit]
        else:
            stats = self.snapshot0.statistics(sortby, cumulative)[:limit]
        for stat in stats:
            print(stat)

    def get_peak(self) -> int:
        if False:
            return 10
        '\n        Return the peak memory consumption.\n        '
        if not self.peak:
            return tracemalloc.get_traced_memory()[1]
        return self.peak

    @staticmethod
    def enable() -> None:
        if False:
            print('Hello World!')
        '\n        Begins profiling calls.\n        '
        tracemalloc.start()

    def disable(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Stop profiling calls.\n        '
        if self.snapshot0 is None:
            self.snapshot()
        self.peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()