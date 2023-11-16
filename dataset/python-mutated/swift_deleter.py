from swiftclient.exceptions import ClientException
from wal_e import retries
from wal_e.worker.base import _Deleter

class Deleter(_Deleter):

    def __init__(self, swift_conn, container):
        if False:
            return 10
        super(Deleter, self).__init__()
        self.swift_conn = swift_conn
        self.container = container

    @retries.retry()
    def _delete_batch(self, page):
        if False:
            print('Hello World!')
        for blob in page:
            try:
                self.swift_conn.delete_object(self.container, blob.name)
            except ClientException as e:
                if e.http_status != 404:
                    raise