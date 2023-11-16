import os
import unittest
from hdfs_test_utils import FSTestBase
from paddle.distributed.fleet.utils.fs import HDFSClient, LocalFS
java_home = os.environ['JAVA_HOME']

class FSTest2(FSTestBase):

    def test_hdfs(self):
        if False:
            i = 10
            return i + 15
        fs = HDFSClient('/usr/local/hadoop-2.7.7/', None, time_out=5 * 1000, sleep_inter=100)
        self._test_rm(fs)
        self._test_touch(fs)
        self._test_dirs(fs)
        self._test_list_files_info(fs)

    def test_local(self):
        if False:
            i = 10
            return i + 15
        fs = LocalFS()
        self._test_rm(fs)
        self._test_touch(fs)
        self._test_dirs(fs)
        self._test_touch_file(fs)
if __name__ == '__main__':
    unittest.main()