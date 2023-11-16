__all__ = ['MemorySize', 'SlotSharingGroup']
from typing import Optional
from pyflink.java_gateway import get_gateway

class MemorySize(object):
    """
    MemorySize is a representation of a number of bytes, viewable in different units.
    """

    def __init__(self, j_memory_size=None, bytes_size: int=None):
        if False:
            print('Hello World!')
        self._j_memory_size = get_gateway().jvm.org.apache.flink.configuration.MemorySize(bytes_size) if j_memory_size is None else j_memory_size

    @staticmethod
    def of_mebi_bytes(mebi_bytes: int) -> 'MemorySize':
        if False:
            for i in range(10):
                print('nop')
        return MemorySize(get_gateway().jvm.org.apache.flink.configuration.MemorySize.ofMebiBytes(mebi_bytes))

    def get_bytes(self) -> int:
        if False:
            return 10
        '\n        Gets the memory size in bytes.\n\n        :return: The memory size in bytes.\n        '
        return self._j_memory_size.getBytes()

    def get_kibi_bytes(self) -> int:
        if False:
            i = 10
            return i + 15
        '\n        Gets the memory size in Kibibytes (= 1024 bytes).\n\n        :return: The memory size in Kibibytes.\n        '
        return self._j_memory_size.getKibiBytes()

    def get_mebi_bytes(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        '\n        Gets the memory size in Mebibytes (= 1024 Kibibytes).\n\n        :return: The memory size in Mebibytes.\n        '
        return self._j_memory_size.getMebiBytes()

    def get_gibi_bytes(self) -> int:
        if False:
            print('Hello World!')
        '\n        Gets the memory size in Gibibytes (= 1024 Mebibytes).\n\n        :return: The memory size in Gibibytes.\n        '
        return self._j_memory_size.getGibiBytes()

    def get_tebi_bytes(self) -> int:
        if False:
            while True:
                i = 10
        '\n        Gets the memory size in Tebibytes (= 1024 Gibibytes).\n\n        :return: The memory size in Tebibytes.\n        '
        return self._j_memory_size.getTebiBytes()

    def get_java_memory_size(self):
        if False:
            i = 10
            return i + 15
        '\n        Gets the Java MemorySize object.\n\n        :return: The Java MemorySize object.\n        '
        return self._j_memory_size

    def __eq__(self, other):
        if False:
            return 10
        return isinstance(other, self.__class__) and self._j_memory_size == other._j_memory_size

    def __hash__(self):
        if False:
            while True:
                i = 10
        return self._j_memory_size.hashCode()

    def __lt__(self, other: 'MemorySize'):
        if False:
            i = 10
            return i + 15
        if not isinstance(other, MemorySize):
            raise Exception('Does not support comparison with non-MemorySize %s' % other)
        return self._j_memory_size.compareTo(other._j_memory_size) == -1

    def __le__(self, other: 'MemorySize'):
        if False:
            return 10
        return self.__eq__(other) and self.__lt__(other)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return self._j_memory_size.toString()

class SlotSharingGroup(object):
    """
    Describe the name and the different resource components of a slot sharing group.
    """

    def __init__(self, j_slot_sharing_group):
        if False:
            while True:
                i = 10
        self._j_slot_sharing_group = j_slot_sharing_group

    def get_name(self) -> str:
        if False:
            return 10
        '\n        Gets the name of this SlotSharingGroup.\n\n        :return: The name of the SlotSharingGroup.\n        '
        return self._j_slot_sharing_group.getName()

    def get_managed_memory(self) -> Optional[MemorySize]:
        if False:
            while True:
                i = 10
        '\n        Gets the task managed memory for this SlotSharingGroup.\n\n        :return: The task managed memory of the SlotSharingGroup.\n        '
        managed_memory = self._j_slot_sharing_group.getManagedMemory()
        return MemorySize(managed_memory.get()) if managed_memory.isPresent() else None

    def get_task_heap_memory(self) -> Optional[MemorySize]:
        if False:
            i = 10
            return i + 15
        '\n        Gets the task heap memory for this SlotSharingGroup.\n\n        :return: The task heap memory of the SlotSharingGroup.\n        '
        task_heap_memory = self._j_slot_sharing_group.getTaskHeapMemory()
        return MemorySize(task_heap_memory.get()) if task_heap_memory.isPresent() else None

    def get_task_off_heap_memory(self) -> Optional[MemorySize]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Gets the task off-heap memory for this SlotSharingGroup.\n\n        :return: The task off-heap memory of the SlotSharingGroup.\n        '
        task_off_heap_memory = self._j_slot_sharing_group.getTaskOffHeapMemory()
        return MemorySize(task_off_heap_memory.get()) if task_off_heap_memory.isPresent() else None

    def get_cpu_cores(self) -> Optional[float]:
        if False:
            return 10
        '\n       Gets the CPU cores for this SlotSharingGroup.\n\n        :return: The CPU cores of the SlotSharingGroup.\n        '
        cpu_cores = self._j_slot_sharing_group.getCpuCores()
        return cpu_cores.get() if cpu_cores.isPresent() else None

    def get_external_resources(self) -> dict:
        if False:
            for i in range(10):
                print('nop')
        '\n        Gets the external resource from this SlotSharingGroup.\n\n        :return: User specified resources of the SlotSharingGroup.\n        '
        return dict(self._j_slot_sharing_group.getExternalResources())

    def get_java_slot_sharing_group(self):
        if False:
            print('Hello World!')
        '\n        Gets the Java SlotSharingGroup object.\n\n        :return: The Java SlotSharingGroup object.\n        '
        return self._j_slot_sharing_group

    @staticmethod
    def builder(name: str) -> 'Builder':
        if False:
            i = 10
            return i + 15
        '\n        Gets the Builder with the given name for this SlotSharingGroup.\n\n        :param name: The name of the SlotSharingGroup.\n        :return: The builder for the SlotSharingGroup.\n        '
        return SlotSharingGroup.Builder(get_gateway().jvm.org.apache.flink.api.common.operators.SlotSharingGroup.newBuilder(name))

    def __eq__(self, other):
        if False:
            return 10
        return isinstance(other, self.__class__) and self._j_slot_sharing_group == other._j_slot_sharing_group

    def __hash__(self):
        if False:
            while True:
                i = 10
        return self._j_slot_sharing_group.hashCode()

    class Builder(object):
        """
        Builder for the SlotSharingGroup.
        """

        def __init__(self, j_builder):
            if False:
                while True:
                    i = 10
            self._j_builder = j_builder

        def set_cpu_cores(self, cpu_cores: float) -> 'SlotSharingGroup.Builder':
            if False:
                while True:
                    i = 10
            '\n            Sets the CPU cores for this SlotSharingGroup.\n\n            :param cpu_cores: The CPU cores of the SlotSharingGroup.\n            :return: This object.\n            '
            self._j_builder.setCpuCores(cpu_cores)
            return self

        def set_task_heap_memory(self, task_heap_memory: MemorySize) -> 'SlotSharingGroup.Builder':
            if False:
                i = 10
                return i + 15
            '\n            Sets the task heap memory for this SlotSharingGroup.\n\n            :param task_heap_memory: The task heap memory of the SlotSharingGroup.\n            :return: This object.\n            '
            self._j_builder.setTaskHeapMemory(task_heap_memory.get_java_memory_size())
            return self

        def set_task_heap_memory_mb(self, task_heap_memory_mb: int) -> 'SlotSharingGroup.Builder':
            if False:
                print('Hello World!')
            '\n            Sets the task heap memory for this SlotSharingGroup in MB.\n\n            :param task_heap_memory_mb: The task heap memory of the SlotSharingGroup in MB.\n            :return: This object.\n            '
            self._j_builder.setTaskHeapMemoryMB(task_heap_memory_mb)
            return self

        def set_task_off_heap_memory(self, task_off_heap_memory: MemorySize) -> 'SlotSharingGroup.Builder':
            if False:
                i = 10
                return i + 15
            '\n            Sets the task off-heap memory for this SlotSharingGroup.\n\n            :param task_off_heap_memory: The task off-heap memory of the SlotSharingGroup.\n            :return: This object.\n            '
            self._j_builder.setTaskOffHeapMemory(task_off_heap_memory.get_java_memory_size())
            return self

        def set_task_off_heap_memory_mb(self, task_off_heap_memory_mb: int) -> 'SlotSharingGroup.Builder':
            if False:
                while True:
                    i = 10
            '\n            Sets the task off-heap memory for this SlotSharingGroup in MB.\n\n            :param task_off_heap_memory_mb: The task off-heap memory of the SlotSharingGroup in MB.\n            :return: This object.\n            '
            self._j_builder.setTaskOffHeapMemoryMB(task_off_heap_memory_mb)
            return self

        def set_managed_memory(self, managed_memory: MemorySize) -> 'SlotSharingGroup.Builder':
            if False:
                return 10
            '\n            Sets the task managed memory for this SlotSharingGroup.\n\n            :param managed_memory: The task managed memory of the SlotSharingGroup.\n            :return: This object.\n            '
            self._j_builder.setManagedMemory(managed_memory.get_java_memory_size())
            return self

        def set_managed_memory_mb(self, managed_memory_mb: int) -> 'SlotSharingGroup.Builder':
            if False:
                i = 10
                return i + 15
            '\n            Sets the task managed memory for this SlotSharingGroup in MB.\n\n            :param managed_memory_mb: The task managed memory of the SlotSharingGroup in MB.\n            :return: This object.\n            '
            self._j_builder.setManagedMemoryMB(managed_memory_mb)
            return self

        def set_external_resource(self, name: str, value: float) -> 'SlotSharingGroup.Builder':
            if False:
                for i in range(10):
                    print('nop')
            '\n            Adds the given external resource. The old value with the same resource name will be\n            replaced if present.\n\n            :param name: The resource name of the given external resource.\n            :param value: The value of the given external resource.\n            :return: This object.\n            '
            self._j_builder.setExternalResource(name, value)
            return self

        def build(self) -> 'SlotSharingGroup':
            if False:
                i = 10
                return i + 15
            '\n            Builds the SlotSharingGroup.\n\n            :return: The SlotSharingGroup object.\n            '
            return SlotSharingGroup(j_slot_sharing_group=self._j_builder.build())