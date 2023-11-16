from deeplake.core.storage.deeplake_memory_object import DeepLakeMemoryObject
from deeplake.core.storage import LRUCache
from deeplake.util.keys import get_dataset_diff_key
import typing
from collections import OrderedDict
import deeplake.core.dataset

class DatasetDiff(DeepLakeMemoryObject):

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        self.is_dirty = False
        self.info_updated = False
        self.renamed: typing.OrderedDict = OrderedDict()
        self.deleted: typing.List[str] = []

    def tobytes(self) -> bytes:
        if False:
            while True:
                i = 10
        "Returns bytes representation of the dataset diff\n\n        The format stores the following information in order:\n        1. The first byte is a boolean value indicating whether the Dataset info was modified or not.\n        2. The next 8 bytes give the number of renamed tensors, let's call this m.\n        3. Next, there will be m blocks of bytes with the following format:\n            1. 8 + 8 bytes giving the length of old and new names, let's call them x and y.\n            2. x bytes of old name.\n            3. y bytes of new name.\n        4. The next 8 bytes give the number of deleted tensors, let's call this n.\n        5. Next, there will be n blocks of bytes with the following format:\n            1. 8 bytes giving the length of the name of the deleted tensor, let's call this z.\n            2. n bytes of name of the deleted tensor.\n        "
        return b''.join([self.info_updated.to_bytes(1, 'big'), len(self.renamed).to_bytes(8, 'big'), *(b''.join([len(old_name).to_bytes(8, 'big'), len(new_name).to_bytes(8, 'big'), old_name + new_name]) for (old_name, new_name) in map(lambda n: (n[0].encode('utf-8'), n[1].encode('utf-8')), self.renamed.items())), len(self.deleted).to_bytes(8, 'big'), *(b''.join([len(name).to_bytes(8, 'big'), name.encode('utf-8')]) for name in self.deleted)])

    @classmethod
    def frombuffer(cls, data: bytes) -> 'DatasetDiff':
        if False:
            while True:
                i = 10
        'Creates a DatasetDiff object from bytes'
        dataset_diff = cls()
        dataset_diff.info_updated = bool(int.from_bytes(data[:1], 'big'))
        len_renamed = int.from_bytes(data[1:9], 'big')
        pos = 9
        for _ in range(len_renamed):
            (len_old, len_new) = (int.from_bytes(data[pos:pos + 8], 'big'), int.from_bytes(data[pos + 8:pos + 16], 'big'))
            pos += 16
            (old_name, new_name) = (data[pos:pos + len_old].decode('utf-8'), data[pos + len_old:pos + len_old + len_new].decode('utf-8'))
            pos += len_old + len_new
            dataset_diff.renamed[old_name] = new_name
        len_deleted = int.from_bytes(data[pos:pos + 8], 'big')
        pos += 8
        for _ in range(len_deleted):
            len_name = int.from_bytes(data[pos:pos + 8], 'big')
            pos += 8
            name = data[pos:pos + len_name].decode('utf-8')
            pos += len_name
            dataset_diff.deleted.append(name)
        return dataset_diff

    @property
    def nbytes(self):
        if False:
            print('Hello World!')
        'Returns number of bytes required to store the dataset diff'
        return 1

    def modify_info(self) -> None:
        if False:
            print('Hello World!')
        'Stores information that the info has changed'
        self.info_updated = True
        self.is_dirty = True

    def tensor_renamed(self, old_name, new_name):
        if False:
            for i in range(10):
                print('nop')
        'Adds old and new name of a tensor that was renamed to renamed'
        for (old, new) in self.renamed.items():
            if old_name == new:
                if old == new_name:
                    self.renamed.pop(old)
                else:
                    self.renamed[old] = new_name
                break
        else:
            self.renamed[old_name] = new_name
        self.is_dirty = True

    def tensor_deleted(self, name):
        if False:
            while True:
                i = 10
        'Adds name of deleted tensor to deleted'
        if name not in self.deleted:
            for (old, new) in self.renamed.items():
                if name == new:
                    self.renamed.pop(old)
                    self.deleted.append(old)
                    break
            else:
                self.deleted.append(name)
            self.is_dirty = True

def load_dataset_diff(dataset: 'deeplake.core.dataset.Dataset'):
    if False:
        while True:
            i = 10
    storage: LRUCache = dataset.storage
    path = get_dataset_diff_key(dataset.version_state['commit_id'])
    try:
        diff = storage.get_deeplake_object(path, DatasetDiff)
    except KeyError:
        diff = DatasetDiff()
    storage.register_deeplake_object(path, diff)
    return diff