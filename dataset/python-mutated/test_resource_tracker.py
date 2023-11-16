import time
from tests.base import TestBase
from aim.sdk.types import QueryReportMode
from aim.sdk import Run

class TestRunResourceTracker(TestBase):

    def test_default_tracking_interval(self):
        if False:
            i = 10
            return i + 15
        run = Run()
        run_hash = run.hash
        run.track(1, name='metric')
        time.sleep(0.1)
        del run
        metrics = list(self.repo.query_metrics(f'run.hash == "{run_hash}" and metric.name.startswith("__")', report_mode=QueryReportMode.DISABLED))
        expected_metrics = {'__system__cpu', '__system__disk_percent', '__system__memory_percent', '__system__p_memory_percent'}
        metric_names = set((m.name for m in metrics))
        for name in expected_metrics:
            self.assertIn(name, metric_names)

    def test_custom_tracking_interval(self):
        if False:
            print('Hello World!')
        run = Run(system_tracking_interval=1)
        run_hash = run.hash
        run.track(1, name='metric')
        time.sleep(3)
        del run
        metrics = list(self.repo.query_metrics(f'run.hash == "{run_hash}" and metric.name.startswith("__")', report_mode=QueryReportMode.DISABLED))
        expected_metrics = {'__system__cpu', '__system__disk_percent', '__system__memory_percent', '__system__p_memory_percent'}
        metric_names = set((m.name for m in metrics))
        for name in expected_metrics:
            self.assertIn(name, metric_names)
        for metric in metrics:
            self.assertGreaterEqual(len(metric.data.indices_list()), 3)
            self.assertLessEqual(len(metric.data.indices_list()), 4)

    def test_disable_resource_tracking(self):
        if False:
            for i in range(10):
                print('nop')
        run = Run(system_tracking_interval=None)
        run_hash = run.hash
        run.track(1, name='metric')
        time.sleep(0.1)
        del run
        metrics = list(self.repo.query_metrics(f'run.hash == "{run_hash}" and metric.name.startswith("__")', report_mode=QueryReportMode.DISABLED))
        self.assertListEqual([], metrics)

    def test_resource_tracking_interval_limits(self):
        if False:
            while True:
                i = 10
        run = Run(system_tracking_interval=0, capture_terminal_logs=False)
        self.assertIsNone(run._system_resource_tracker)
        run = Run(system_tracking_interval=2 * 24 * 3600, capture_terminal_logs=False)
        self.assertIsNone(run._system_resource_tracker)