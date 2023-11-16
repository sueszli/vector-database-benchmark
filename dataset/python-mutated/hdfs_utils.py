from __future__ import annotations

class FakeWebHDFSHook:

    def __init__(self, conn_id):
        if False:
            while True:
                i = 10
        self.conn_id = conn_id

    def get_conn(self):
        if False:
            for i in range(10):
                print('nop')
        return self.conn_id

    def check_for_path(self, hdfs_path):
        if False:
            i = 10
            return i + 15
        return hdfs_path