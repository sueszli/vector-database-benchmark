"""Tests for `multi_process_runner` for non-initialization."""
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.eager import test

class MultiProcessRunnerNoInitTest(test.TestCase):

    def test_not_calling_correct_main(self):
        if False:
            print('Hello World!')

        def simple_func():
            if False:
                return 10
            return 'foobar'
        with self.assertRaisesRegex(multi_process_runner.NotInitializedError, '`multi_process_runner` is not initialized.'):
            multi_process_runner.run(simple_func, multi_worker_test_base.create_cluster_spec(num_workers=1))
if __name__ == '__main__':
    test.main()