from tests.base import TestBase
from aim.sdk.run import Run

class TestRunCreationChecks(TestBase):

    @classmethod
    def setUpClass(cls) -> None:
        if False:
            print('Hello World!')
        super().setUpClass()
        run = Run(system_tracking_interval=None, log_system_params=False, capture_terminal_logs=False)
        cls.existing_run_hash = run.hash

    def test_reopen_existing_run_in_write_mode(self):
        if False:
            print('Hello World!')
        run = Run(self.existing_run_hash, system_tracking_interval=None, log_system_params=False, capture_terminal_logs=False)
        self.assertEqual(run.hash, self.existing_run_hash)

    def test_reopen_existing_run_in_read_mode(self):
        if False:
            for i in range(10):
                print('nop')
        run = Run(self.existing_run_hash, read_only=True)
        self.assertEqual(run.hash, self.existing_run_hash)