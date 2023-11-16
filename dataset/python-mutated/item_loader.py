import os
from abc import ABC, abstractmethod
from time import sleep
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from lightning.data.streaming.constants import _TORCH_DTYPES_MAPPING, _TORCH_GREATER_EQUAL_2_1_0
from lightning.data.streaming.serializers import Serializer
if _TORCH_GREATER_EQUAL_2_1_0:
    from torch.utils._pytree import PyTree, tree_unflatten

class BaseItemLoader(ABC):
    """The base item loader is responsible to decide how the items within a chunk are loaded."""

    def setup(self, config: Dict, chunks: List, serializers: Dict[str, Serializer]) -> None:
        if False:
            return 10
        self._config = config
        self._chunks = chunks
        self._serializers = serializers

    @abstractmethod
    def generate_intervals(self) -> List[Tuple[int, int]]:
        if False:
            print('Hello World!')
        'Returns a list of tuple describing the indexes intervals of the chunks.'
        pass

    @abstractmethod
    def load_item_from_chunk(self, index: int, chunk_index: int, chunk_filepath: str, begin: int) -> Any:
        if False:
            return 10
        'Returns an item loaded from a chunk.'
        pass

class PyTreeLoader(BaseItemLoader):
    """The Pytree Loader is the default loader of the Cache object."""

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        self._chunk_filepaths: Dict[str, bool] = {}

    def generate_intervals(self) -> List[Tuple[int, int]]:
        if False:
            i = 10
            return i + 15
        intervals = []
        begin = 0
        end = 0
        for chunk in self._chunks:
            end += chunk['chunk_size']
            intervals.append((begin, end))
            begin += chunk['chunk_size']
        return intervals

    def load_item_from_chunk(self, index: int, chunk_index: int, chunk_filepath: str, begin: int) -> bytes:
        if False:
            i = 10
            return i + 15
        offset = (1 + (index - begin) if index >= begin else index + 1) * 4
        if chunk_filepath in self._chunk_filepaths and (not os.path.isfile(chunk_filepath)):
            del self._chunk_filepaths[chunk_filepath]
        if chunk_filepath not in self._chunk_filepaths:
            while not os.path.exists(chunk_filepath):
                sleep(0.01)
            sleep(0.01)
            self._chunk_filepaths[chunk_filepath] = True
        with open(chunk_filepath, 'rb', 0) as fp:
            fp.seek(offset)
            pair = fp.read(8)
            (begin, end) = np.frombuffer(pair, np.uint32)
            fp.seek(begin)
            data = fp.read(end - begin)
        return self.deserialize(data)

    def deserialize(self, raw_item_data: bytes) -> 'PyTree':
        if False:
            print('Hello World!')
        'Deserialize the raw bytes into their python equivalent.'
        idx = len(self._config['data_format']) * 4
        sizes = np.frombuffer(raw_item_data[:idx], np.uint32)
        data = []
        for (size, data_format) in zip(sizes, self._config['data_format']):
            serializer = self._serializers[data_format]
            data_bytes = raw_item_data[idx:idx + size]
            data.append(serializer.deserialize(data_bytes))
            idx += size
        return tree_unflatten(data, self._config['data_spec'])

class TokensLoader(BaseItemLoader):

    def __init__(self, block_size: int):
        if False:
            i = 10
            return i + 15
        'The Tokens Loader is an optimizer item loader for NLP.\n\n        Arguments:\n            block_size: The context length to use during training.\n\n        '
        super().__init__()
        self._block_size = block_size
        self._intervals: List[Tuple[int, int]] = []
        self._mmaps: Dict[int, np.memmap] = {}
        self._buffers: Dict[int, bytes] = {}
        self._dtype: Optional[torch.dtype] = None
        self._chunk_filepaths: Dict[str, bool] = {}

    def setup(self, config: Dict, chunks: List, serializers: Dict[str, Serializer]) -> None:
        if False:
            i = 10
            return i + 15
        super().setup(config, chunks, serializers)
        self._dtype = _TORCH_DTYPES_MAPPING[int(config['data_format'][0].split(':')[1])]
        if all((chunk['dim'] is None for chunk in self._chunks)):
            raise ValueError("The provided chunks isn't properly setup.")

    def generate_intervals(self) -> List[Tuple[int, int]]:
        if False:
            return 10
        begin = 0
        end = 0
        for chunk in self._chunks:
            dim = chunk['dim']
            num_blocks = dim // self._block_size
            end += num_blocks
            self._intervals.append((begin, end))
            begin += num_blocks
        return self._intervals

    def load_item_from_chunk(self, index: int, chunk_index: int, chunk_filepath: str, begin: int) -> torch.Tensor:
        if False:
            print('Hello World!')
        if chunk_filepath in self._chunk_filepaths and (not os.path.isfile(chunk_filepath)):
            del self._chunk_filepaths[chunk_filepath]
        if chunk_filepath not in self._chunk_filepaths:
            while not os.path.exists(chunk_filepath):
                sleep(0.01)
            sleep(0.01)
            self._chunk_filepaths[chunk_filepath] = True
        if chunk_index not in self._mmaps:
            chunk = self._chunks[chunk_index]
            offset = (1 + chunk['chunk_size'] + 1) * 4
            mmap = np.memmap(chunk_filepath, mode='r', order='C', offset=offset)
            self._mmaps[chunk_index] = mmap
            self._buffers[chunk_index] = memoryview(mmap)
        assert self._dtype
        buffer: bytes = self._buffers[chunk_index]
        offset = self._dtype.itemsize * (index - begin if index >= begin else index + 1)
        return torch.frombuffer(buffer, dtype=self._dtype, count=self._block_size, offset=offset)