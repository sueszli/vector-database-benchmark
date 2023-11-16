"""Tests for watchdog.py."""
import os
import time
from absl.testing import parameterized
from tensorflow.python.distribute.coordinator import watchdog
from tensorflow.python.eager import test

class WatchDogTest(test.TestCase, parameterized.TestCase):

    @parameterized.parameters(True, False)
    def testWatchDogTimeout(self, use_env_var):
        if False:
            return 10
        tmp_file = self.create_tempfile()
        f = open(tmp_file, 'w+')
        triggerred_count = [0]

        def on_triggered_fn():
            if False:
                for i in range(10):
                    print('nop')
            triggerred_count[0] += 1
        timeout = 3
        if use_env_var:
            os.environ['TF_CLUSTER_COORDINATOR_WATCH_DOG_TIMEOUT'] = str(timeout)
            wd = watchdog.WatchDog(traceback_file=f, on_triggered=on_triggered_fn)
        else:
            wd = watchdog.WatchDog(timeout=timeout, traceback_file=f, on_triggered=on_triggered_fn)
        time.sleep(6)
        self.assertGreaterEqual(triggerred_count[0], 1)
        wd.report_closure_done()
        time.sleep(1)
        self.assertGreaterEqual(triggerred_count[0], 1)
        time.sleep(5)
        self.assertGreaterEqual(triggerred_count[0], 2)
        wd.stop()
        time.sleep(5)
        last_triggered_count = triggerred_count[0]
        time.sleep(10)
        self.assertEqual(last_triggered_count, triggerred_count[0])
        f.close()
        with open(tmp_file) as f:
            self.assertIn('Current thread', f.read())
if __name__ == '__main__':
    test.main()