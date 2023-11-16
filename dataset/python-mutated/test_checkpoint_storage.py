from pyflink.datastream.checkpoint_storage import JobManagerCheckpointStorage, FileSystemCheckpointStorage
from pyflink.java_gateway import get_gateway
from pyflink.testing.test_case_utils import PyFlinkTestCase

class JobManagerCheckpointStorageTests(PyFlinkTestCase):

    def test_constant(self):
        if False:
            print('Hello World!')
        gateway = get_gateway()
        JJobManagerCheckpointStorage = gateway.jvm.org.apache.flink.runtime.state.storage.JobManagerCheckpointStorage
        self.assertEqual(JobManagerCheckpointStorage.DEFAULT_MAX_STATE_SIZE, JJobManagerCheckpointStorage.DEFAULT_MAX_STATE_SIZE)

    def test_create_jobmanager_checkpoint_storage(self):
        if False:
            return 10
        self.assertIsNotNone(JobManagerCheckpointStorage())
        self.assertIsNotNone(JobManagerCheckpointStorage('file://var/checkpoints/'))
        self.assertIsNotNone(JobManagerCheckpointStorage('file://var/checkpoints/', 10000000))

    def test_get_max_state_size(self):
        if False:
            print('Hello World!')
        checkpoint_storage = JobManagerCheckpointStorage()
        self.assertEqual(checkpoint_storage.get_max_state_size(), JobManagerCheckpointStorage.DEFAULT_MAX_STATE_SIZE)
        checkpoint_storage = JobManagerCheckpointStorage(max_state_size=50000)
        self.assertEqual(checkpoint_storage.get_max_state_size(), 50000)

class FileSystemCheckpointStorageTests(PyFlinkTestCase):

    def test_create_fs_checkpoint_storage(self):
        if False:
            print('Hello World!')
        self.assertIsNotNone(FileSystemCheckpointStorage('file://var/checkpoints/'))
        self.assertIsNotNone(FileSystemCheckpointStorage('file://var/checkpoints/', 2048))
        self.assertIsNotNone(FileSystemCheckpointStorage('file://var/checkpoints/', 2048, 4096))

    def test_get_min_file_size_threshold(self):
        if False:
            print('Hello World!')
        checkpoint_storage = FileSystemCheckpointStorage('file://var/checkpoints/')
        self.assertEqual(checkpoint_storage.get_min_file_size_threshold(), 20480)
        checkpoint_storage = FileSystemCheckpointStorage('file://var/checkpoints/', file_state_size_threshold=2048)
        self.assertEqual(checkpoint_storage.get_min_file_size_threshold(), 2048)

    def test_get_checkpoint_path(self):
        if False:
            for i in range(10):
                print('nop')
        checkpoint_storage = FileSystemCheckpointStorage('file://var/checkpoints/')
        self.assertEqual(checkpoint_storage.get_checkpoint_path(), 'file://var/checkpoints')