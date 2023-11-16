import os
import time
import ctypes
from mxnet.base import _LIB
from mxnet.base import check_call
import mxnet as mx
import argparse

class IndexCreator(mx.recordio.MXRecordIO):
    """Reads `RecordIO` data format, and creates index file
    that enables random access.

    Example usage:
    ----------
    >> creator = IndexCreator('data/test.rec','data/test.idx')
    >> record.create_index()
    >> record.close()
    >> !ls data/
    test.rec  test.idx

    Parameters
    ----------
    uri : str
        Path to the record file.
    idx_path : str
        Path to the index file, that will be created/overwritten.
    key_type : type
        Data type for keys (optional, default = int).
    """

    def __init__(self, uri, idx_path, key_type=int):
        if False:
            i = 10
            return i + 15
        self.key_type = key_type
        self.fidx = None
        self.idx_path = idx_path
        super(IndexCreator, self).__init__(uri, 'r')

    def open(self):
        if False:
            i = 10
            return i + 15
        super(IndexCreator, self).open()
        self.fidx = open(self.idx_path, 'w')

    def close(self):
        if False:
            print('Hello World!')
        'Closes the record and index files.'
        if not self.is_open:
            return
        super(IndexCreator, self).close()
        self.fidx.close()

    def tell(self):
        if False:
            return 10
        'Returns the current position of read head.\n        '
        pos = ctypes.c_size_t()
        check_call(_LIB.MXRecordIOReaderTell(self.handle, ctypes.byref(pos)))
        return pos.value

    def create_index(self):
        if False:
            while True:
                i = 10
        'Creates the index file from open record file\n        '
        self.reset()
        counter = 0
        pre_time = time.time()
        while True:
            if counter % 1000 == 0:
                cur_time = time.time()
                print('time:', cur_time - pre_time, ' count:', counter)
            pos = self.tell()
            cont = self.read()
            if cont is None:
                break
            key = self.key_type(counter)
            self.fidx.write('%s\t%d\n' % (str(key), pos))
            counter = counter + 1

def parse_args():
    if False:
        print('Hello World!')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Create an index file from .rec file')
    parser.add_argument('record', help='path to .rec file.')
    parser.add_argument('index', help='path to index file.')
    args = parser.parse_args()
    args.record = os.path.abspath(args.record)
    args.index = os.path.abspath(args.index)
    return args

def main():
    if False:
        while True:
            i = 10
    args = parse_args()
    creator = IndexCreator(args.record, args.index)
    creator.create_index()
    creator.close()
if __name__ == '__main__':
    main()