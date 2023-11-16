from typing import Dict, Optional, Tuple
from deeplake.core.storage.deeplake_memory_object import DeepLakeMemoryObject
from collections import defaultdict

class CommitChunkMap(DeepLakeMemoryObject):
    """Stores set of chunks stored for a particular tensor in a commit."""

    def __init__(self) -> None:
        if False:
            return 10
        self.is_dirty = False
        self.chunks: Dict[str, Dict] = {}

    @staticmethod
    def _serialize_entry(kv):
        if False:
            for i in range(10):
                print('nop')
        (k, v) = kv
        if not v:
            return k
        key = v.get('key')
        if key is None:
            return f"{k}:{v['commit_id']}"
        return f"{k}:{v['commit_id']}:{v['key']}"

    @staticmethod
    def _deserialize_entry(e: str) -> Tuple[str, Dict]:
        if False:
            while True:
                i = 10
        sp = e.split(':')
        k = sp[0]
        v = {}
        try:
            v['commit_id'] = sp[1]
            v['key'] = sp[2]
        except IndexError:
            pass
        return (k, v)

    def tobytes(self) -> bytes:
        if False:
            print('Hello World!')
        'Dumps self.chunks in csv format.'
        return bytes(','.join(map(self._serialize_entry, self.chunks.items())), 'utf-8')

    @classmethod
    def frombuffer(cls, buffer: bytes):
        if False:
            print('Hello World!')
        'Loads a CommitChunkMap from a buffer.'
        instance = cls()
        if buffer:
            entries = buffer.decode('utf-8').split(',')
            instance.chunks = dict(map(cls._deserialize_entry, entries))
        instance.is_dirty = False
        return instance

    @property
    def nbytes(self) -> int:
        if False:
            return 10
        if not self.chunks:
            return 0
        return 16 + (len(self.chunks) - 1) * 17

    def add(self, chunk_name: str, commit_id: Optional[str]=None, key: Optional[str]=None) -> None:
        if False:
            while True:
                i = 10
        'Adds a new chunk name to the CommitChunkMap.'
        v = {}
        if commit_id:
            v['commit_id'] = commit_id
            if key:
                v['key'] = key
        self.chunks[chunk_name] = v
        self.is_dirty = True