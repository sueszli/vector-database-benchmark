import time
import os
from tests.base import TestBase
from aim.sdk.run import Run
from aim.sdk.configs import AIM_ENABLE_TRACKING_THREAD

class TestRunFinalizedAt(TestBase):

    def _query_run_finalized_at(self, run_hash):
        if False:
            i = 10
            return i + 15
        run = self.repo.get_run(run_hash)
        return run.end_time

    def test_implicit_run_delete(self):
        if False:
            return 10
        run_hash = []

        def _func():
            if False:
                print('Hello World!')
            run = Run(system_tracking_interval=None)
            run_hash.append(run.hash)
            self.assertIsNone(run.end_time)
            for i in range(10):
                run.track(i, name='seq')
            self.assertIsNone(run.end_time)
        _func()
        time.sleep(0.1)
        self.assertIsNotNone(self._query_run_finalized_at(run_hash[0]))

    def test_explicit_run_delete(self):
        if False:
            return 10
        run = Run(system_tracking_interval=None)
        run_hash = run.hash
        for i in range(10):
            run.track(i, name='seq')
        del run
        time.sleep(0.1)
        self.assertIsNotNone(self._query_run_finalized_at(run_hash))

    def test_explicit_run_finalize(self):
        if False:
            for i in range(10):
                print('nop')
        run = Run(system_tracking_interval=None)
        for i in range(10):
            run.track(i, name='seq')
        self.assertIsNone(run.end_time)
        run.finalize()
        self.assertIsNotNone(run.end_time)

class TestRunFinalizedAtWithTrackingQueue(TestRunFinalizedAt):

    @classmethod
    def setUpClass(cls) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().setUpClass()
        Run.track_in_thread = True

    @classmethod
    def tearDownClass(cls) -> None:
        if False:
            i = 10
            return i + 15
        Run.track_in_thread = False
        super().tearDownClass()