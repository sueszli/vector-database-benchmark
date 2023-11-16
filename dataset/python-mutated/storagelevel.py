__all__ = ['StorageLevel']
from typing import Any, ClassVar

class StorageLevel:
    """
    Flags for controlling the storage of an RDD. Each StorageLevel records whether to use memory,
    whether to drop the RDD to disk if it falls out of memory, whether to keep the data in memory
    in a JAVA-specific serialized format, and whether to replicate the RDD partitions on multiple
    nodes. Also contains static constants for some commonly used storage levels, MEMORY_ONLY.
    Since the data is always serialized on the Python side, all the constants use the serialized
    formats.
    """
    NONE: ClassVar['StorageLevel']
    DISK_ONLY: ClassVar['StorageLevel']
    DISK_ONLY_2: ClassVar['StorageLevel']
    DISK_ONLY_3: ClassVar['StorageLevel']
    MEMORY_ONLY: ClassVar['StorageLevel']
    MEMORY_ONLY_2: ClassVar['StorageLevel']
    MEMORY_AND_DISK: ClassVar['StorageLevel']
    MEMORY_AND_DISK_2: ClassVar['StorageLevel']
    OFF_HEAP: ClassVar['StorageLevel']
    MEMORY_AND_DISK_DESER: ClassVar['StorageLevel']

    def __init__(self, useDisk: bool, useMemory: bool, useOffHeap: bool, deserialized: bool, replication: int=1):
        if False:
            while True:
                i = 10
        self.useDisk = useDisk
        self.useMemory = useMemory
        self.useOffHeap = useOffHeap
        self.deserialized = deserialized
        self.replication = replication

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return 'StorageLevel(%s, %s, %s, %s, %s)' % (self.useDisk, self.useMemory, self.useOffHeap, self.deserialized, self.replication)

    def __str__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        result = ''
        result += 'Disk ' if self.useDisk else ''
        result += 'Memory ' if self.useMemory else ''
        result += 'OffHeap ' if self.useOffHeap else ''
        result += 'Deserialized ' if self.deserialized else 'Serialized '
        result += '%sx Replicated' % self.replication
        return result

    def __eq__(self, other: Any) -> bool:
        if False:
            return 10
        return isinstance(other, StorageLevel) and self.useMemory == other.useMemory and (self.useDisk == other.useDisk) and (self.useOffHeap == other.useOffHeap) and (self.deserialized == other.deserialized) and (self.replication == other.replication)
StorageLevel.NONE = StorageLevel(False, False, False, False)
StorageLevel.DISK_ONLY = StorageLevel(True, False, False, False)
StorageLevel.DISK_ONLY_2 = StorageLevel(True, False, False, False, 2)
StorageLevel.DISK_ONLY_3 = StorageLevel(True, False, False, False, 3)
StorageLevel.MEMORY_ONLY = StorageLevel(False, True, False, False)
StorageLevel.MEMORY_ONLY_2 = StorageLevel(False, True, False, False, 2)
StorageLevel.MEMORY_AND_DISK = StorageLevel(True, True, False, False)
StorageLevel.MEMORY_AND_DISK_2 = StorageLevel(True, True, False, False, 2)
StorageLevel.OFF_HEAP = StorageLevel(True, True, True, False, 1)
StorageLevel.MEMORY_AND_DISK_DESER = StorageLevel(True, True, False, True)