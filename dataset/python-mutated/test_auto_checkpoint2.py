import os
import unittest
from auto_checkpoint_utils import get_logger
from test_auto_checkpoint import AutoCheckPointACLBase
import paddle
paddle.enable_static()
logger = get_logger()

class AutoCheckpointTest2(AutoCheckPointACLBase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        get_logger()
        logger.info('enter tests')
        self._old_environ = dict(os.environ)
        proc_env = {'PADDLE_RUNNING_ENV': 'PADDLE_EDL_AUTO_CHECKPOINT', 'PADDLE_TRAINER_ID': '0', 'PADDLE_RUNNING_PLATFORM': 'PADDLE_CLOUD', 'PADDLE_JOB_ID': 'test_job_auto_2', 'PADDLE_EDL_HDFS_HOME': '/usr/local/hadoop-2.7.7', 'PADDLE_EDL_HDFS_NAME': '', 'PADDLE_EDL_HDFS_UGI': '', 'PADDLE_EDL_HDFS_CHECKPOINT_PATH': 'auto_checkpoint_2', 'PADDLE_EDL_ONLY_FOR_CE_TEST': '1', 'PADDLE_EDL_FS_CACHE': '.auto_checkpoint_test_2', 'PADDLE_EDL_SAVE_CHECKPOINT_INTER': '0'}
        os.environ.update(proc_env)

    def test_corner_epoch_no(self):
        if False:
            while True:
                i = 10
        self._test_corner_epoch_no(1)
if __name__ == '__main__':
    unittest.main()