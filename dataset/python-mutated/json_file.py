from __future__ import annotations
import logging
from pathlib import Path
from typing import Iterator
import orjson
from autogpt.config import Config
from ..memory_item import MemoryItem
from .base import VectorMemoryProvider
logger = logging.getLogger(__name__)

class JSONFileMemory(VectorMemoryProvider):
    """Memory backend that stores memories in a JSON file"""
    SAVE_OPTIONS = orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_SERIALIZE_DATACLASS
    file_path: Path
    memories: list[MemoryItem]

    def __init__(self, config: Config) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Initialize a class instance\n\n        Args:\n            config: Config object\n\n        Returns:\n            None\n        '
        self.file_path = config.workspace_path / f'{config.memory_index}.json'
        self.file_path.touch()
        logger.debug(f'Initialized {__class__.__name__} with index path {self.file_path}')
        self.memories = []
        try:
            self.load_index()
            logger.debug(f'Loaded {len(self.memories)} MemoryItems from file')
        except Exception as e:
            logger.warn(f'Could not load MemoryItems from file: {e}')
            self.save_index()

    def __iter__(self) -> Iterator[MemoryItem]:
        if False:
            return 10
        return iter(self.memories)

    def __contains__(self, x: MemoryItem) -> bool:
        if False:
            print('Hello World!')
        return x in self.memories

    def __len__(self) -> int:
        if False:
            i = 10
            return i + 15
        return len(self.memories)

    def add(self, item: MemoryItem):
        if False:
            return 10
        self.memories.append(item)
        logger.debug(f'Adding item to memory: {item.dump()}')
        self.save_index()
        return len(self.memories)

    def discard(self, item: MemoryItem):
        if False:
            while True:
                i = 10
        try:
            self.remove(item)
        except:
            pass

    def clear(self):
        if False:
            for i in range(10):
                print('nop')
        'Clears the data in memory.'
        self.memories.clear()
        self.save_index()

    def load_index(self):
        if False:
            print('Hello World!')
        'Loads all memories from the index file'
        if not self.file_path.is_file():
            logger.debug(f"Index file '{self.file_path}' does not exist")
            return
        with self.file_path.open('r') as f:
            logger.debug(f"Loading memories from index file '{self.file_path}'")
            json_index = orjson.loads(f.read())
            for memory_item_dict in json_index:
                self.memories.append(MemoryItem.parse_obj(memory_item_dict))

    def save_index(self):
        if False:
            return 10
        logger.debug(f'Saving memory index to file {self.file_path}')
        with self.file_path.open('wb') as f:
            return f.write(orjson.dumps([m.dict() for m in self.memories], option=self.SAVE_OPTIONS))