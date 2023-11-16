import os
import unittest
from hdfs_test_utils import FSTestBase
from paddle.distributed.fleet.utils.fs import HDFSClient, LocalFS
java_home = os.environ['JAVA_HOME']

class FSTest3(FSTestBase):

    def test_hdfs(self):
        if False:
            print('Hello World!')
        fs = HDFSClient('/usr/local/hadoop-2.7.7/', None, time_out=5 * 1000, sleep_inter=100)
        self._test_mkdirs(fs)
        self._test_list_dir(fs)
        self._test_try_upload(fs)
        self._test_try_download(fs)
        self._test_upload(fs)
        self._test_upload_dir(fs)
        self._test_download(fs)
        self._test_download_dir(fs)

    def test_local(self):
        if False:
            for i in range(10):
                print('nop')
        fs = LocalFS()
        self._test_mkdirs(fs)
        self._test_list_dir(fs)
        self._test_try_upload(fs)
        self._test_try_download(fs)
if __name__ == '__main__':
    unittest.main()