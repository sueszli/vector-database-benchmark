import sys
import uuid

class UploadController(object):

    def __init__(self, fs):
        if False:
            for i in range(10):
                print('nop')
        self.fs = fs
        self.meta = {'chunk_size': 131072 * 4, 'platform': sys.platform, 'syspath': fs.getsyspath('')}
        self.fd_id_map = {}

    def open(self, path, mode):
        if False:
            print('Hello World!')
        id_ = str(uuid.uuid4())
        self.fd_id_map[id_] = self.fs.open(path, mode)
        return id_

    def upload(self, id_, data):
        if False:
            for i in range(10):
                print('nop')
        count = self.fd_id_map[id_].write(data)
        if len(data) < self.meta['chunk_size']:
            self.fd_id_map[id_].close()
            del self.fd_id_map[id_]
        return count

    def download(self, id_):
        if False:
            while True:
                i = 10
        data = self.fd_id_map[id_].read(self.meta['chunk_size'])
        if len(data) < self.meta['chunk_size']:
            self.fd_id_map[id_].close()
            del self.fd_id_map[id_]
        return data