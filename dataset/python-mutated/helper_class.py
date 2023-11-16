from collections import namedtuple
from enum import Enum
from enum import unique

def gb_to_mb(value: int) -> int:
    if False:
        while True:
            i = 10
    return value << 10

class CustomMachineType:
    """
    Allows to create custom machine types to be used with the VM instances.
    """

    @unique
    class CPUSeries(Enum):
        N1 = 'custom'
        N2 = 'n2-custom'
        N2D = 'n2d-custom'
        E2 = 'e2-custom'
        E2_MICRO = 'e2-custom-micro'
        E2_SMALL = 'e2-custom-small'
        E2_MEDIUM = 'e2-custom-medium'
    TypeLimits = namedtuple('TypeLimits', ['allowed_cores', 'min_mem_per_core', 'max_mem_per_core', 'allow_extra_memory', 'extra_memory_limit'])
    LIMITS = {CPUSeries.E2: TypeLimits(frozenset(range(2, 33, 2)), 512, 8192, False, 0), CPUSeries.E2_MICRO: TypeLimits(frozenset(), 1024, 2048, False, 0), CPUSeries.E2_SMALL: TypeLimits(frozenset(), 2048, 4096, False, 0), CPUSeries.E2_MEDIUM: TypeLimits(frozenset(), 4096, 8192, False, 0), CPUSeries.N2: TypeLimits(frozenset(range(2, 33, 2)).union(set(range(36, 129, 4))), 512, 8192, True, gb_to_mb(624)), CPUSeries.N2D: TypeLimits(frozenset({2, 4, 8, 16, 32, 48, 64, 80, 96}), 512, 8192, True, gb_to_mb(768)), CPUSeries.N1: TypeLimits(frozenset({1}.union(range(2, 97, 2))), 922, 6656, True, gb_to_mb(624))}

    def __init__(self, zone: str, cpu_series: CPUSeries, memory_mb: int, core_count: int=0):
        if False:
            for i in range(10):
                print('nop')
        self.zone = zone
        self.cpu_series = cpu_series
        self.limits = self.LIMITS[self.cpu_series]
        self.core_count = 2 if self.is_shared() else core_count
        self.memory_mb = memory_mb
        self._checked = False
        self._check_parameters()
        self.extra_memory_used = self._check_extra_memory()

    def is_shared(self):
        if False:
            print('Hello World!')
        return self.cpu_series in (CustomMachineType.CPUSeries.E2_SMALL, CustomMachineType.CPUSeries.E2_MICRO, CustomMachineType.CPUSeries.E2_MEDIUM)

    def _check_extra_memory(self) -> bool:
        if False:
            while True:
                i = 10
        if self._checked:
            return self.memory_mb > self.core_count * self.limits.max_mem_per_core
        else:
            raise RuntimeError('You need to call _check_parameters() before calling _check_extra_memory()')

    def _check_parameters(self):
        if False:
            i = 10
            return i + 15
        '\n        Check whether the requested parameters are allowed. Find more information about limitations of custom machine\n        types at: https://cloud.google.com/compute/docs/general-purpose-machines#custom_machine_types\n        '
        if self.limits.allowed_cores and self.core_count not in self.limits.allowed_cores:
            raise RuntimeError(f'Invalid number of cores requested. Allowed number of cores for {self.cpu_series.name} is: {sorted(self.limits.allowed_cores)}')
        if self.memory_mb % 256 != 0:
            raise RuntimeError('Requested memory must be a multiple of 256 MB.')
        if self.memory_mb < self.core_count * self.limits.min_mem_per_core:
            raise RuntimeError(f'Requested memory is too low. Minimal memory for {self.cpu_series.name} is {self.limits.min_mem_per_core} MB per core.')
        if self.memory_mb > self.core_count * self.limits.max_mem_per_core:
            if self.limits.allow_extra_memory:
                if self.memory_mb > self.limits.extra_memory_limit:
                    raise RuntimeError(f'Requested memory is too large.. Maximum memory allowed for {self.cpu_series.name} is {self.limits.extra_memory_limit} MB.')
            else:
                raise RuntimeError(f'Requested memory is too large.. Maximum memory allowed for {self.cpu_series.name} is {self.limits.max_mem_per_core} MB per core.')
        self._checked = True

    def __str__(self) -> str:
        if False:
            i = 10
            return i + 15
        '\n        Return the custom machine type in form of a string acceptable by Compute Engine API.\n        '
        if self.cpu_series in {self.CPUSeries.E2_SMALL, self.CPUSeries.E2_MICRO, self.CPUSeries.E2_MEDIUM}:
            return f'zones/{self.zone}/machineTypes/{self.cpu_series.value}-{self.memory_mb}'
        if self.extra_memory_used:
            return f'zones/{self.zone}/machineTypes/{self.cpu_series.value}-{self.core_count}-{self.memory_mb}-ext'
        return f'zones/{self.zone}/machineTypes/{self.cpu_series.value}-{self.core_count}-{self.memory_mb}'

    def short_type_str(self) -> str:
        if False:
            i = 10
            return i + 15
        '\n        Return machine type in a format without the zone. For example, n2-custom-0-10240.\n        This format is used to create instance templates.\n        '
        return str(self).rsplit('/', maxsplit=1)[1]

    @classmethod
    def from_str(cls, machine_type: str):
        if False:
            i = 10
            return i + 15
        '\n        Construct a new object from a string. The string needs to be a valid custom machine type like:\n         - https://www.googleapis.com/compute/v1/projects/diregapic-mestiv/zones/us-central1-b/machineTypes/e2-custom-4-8192\n         - zones/us-central1-b/machineTypes/e2-custom-4-8192\n         - e2-custom-4-8192 (in this case, the zone parameter will not be set)\n        '
        zone = None
        if machine_type.startswith('http'):
            machine_type = machine_type[machine_type.find('zones/'):]
        if machine_type.startswith('zones/'):
            (_, zone, _, machine_type) = machine_type.split('/')
        extra_mem = machine_type.endswith('-ext')
        if machine_type.startswith('custom'):
            cpu = cls.CPUSeries.N1
            (_, cores, memory) = machine_type.rsplit('-', maxsplit=2)
        else:
            if extra_mem:
                (cpu_series, _, cores, memory, _) = machine_type.split('-')
            else:
                (cpu_series, _, cores, memory) = machine_type.split('-')
            if cpu_series == 'n2':
                cpu = cls.CPUSeries.N2
            elif cpu_series == 'n2d':
                cpu = cls.CPUSeries.N2D
            elif cpu_series == 'e2':
                cpu = cls.CPUSeries.E2
                if cores == 'micro':
                    cpu = cls.CPUSeries.E2_MICRO
                    cores = 2
                elif cores == 'small':
                    cpu = cls.CPUSeries.E2_SMALL
                    cores = 2
                elif cores == 'medium':
                    cpu = cls.CPUSeries.E2_MEDIUM
                    cores = 2
            else:
                raise RuntimeError('Unknown CPU series.')
        cores = int(cores)
        memory = int(memory)
        return cls(zone, cpu, memory, cores)