from __future__ import annotations
from airflow.configuration import conf
from airflow.exceptions import AirflowException
MB = 1
GB = 1024 * MB
TB = 1024 * GB
PB = 1024 * TB
EB = 1024 * PB

class Resource:
    """
    Represents a resource requirement in an execution environment for an operator.

    :param name: Name of the resource
    :param units_str: The string representing the units of a resource (e.g. MB for a CPU
        resource) to be used for display purposes
    :param qty: The number of units of the specified resource that are required for
        execution of the operator.
    """

    def __init__(self, name, units_str, qty):
        if False:
            while True:
                i = 10
        if qty < 0:
            raise AirflowException(f'Received resource quantity {qty} for resource {name}, but resource quantity must be non-negative.')
        self._name = name
        self._units_str = units_str
        self._qty = qty

    def __eq__(self, other):
        if False:
            print('Hello World!')
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.__dict__ == other.__dict__

    def __repr__(self):
        if False:
            while True:
                i = 10
        return str(self.__dict__)

    @property
    def name(self):
        if False:
            for i in range(10):
                print('nop')
        'Name of the resource.'
        return self._name

    @property
    def units_str(self):
        if False:
            for i in range(10):
                print('nop')
        'The string representing the units of a resource.'
        return self._units_str

    @property
    def qty(self):
        if False:
            while True:
                i = 10
        'The number of units of the specified resource that are required for execution of the operator.'
        return self._qty

    def to_dict(self):
        if False:
            print('Hello World!')
        return {'name': self.name, 'qty': self.qty, 'units_str': self.units_str}

class CpuResource(Resource):
    """Represents a CPU requirement in an execution environment for an operator."""

    def __init__(self, qty):
        if False:
            return 10
        super().__init__('CPU', 'core(s)', qty)

class RamResource(Resource):
    """Represents a RAM requirement in an execution environment for an operator."""

    def __init__(self, qty):
        if False:
            return 10
        super().__init__('RAM', 'MB', qty)

class DiskResource(Resource):
    """Represents a disk requirement in an execution environment for an operator."""

    def __init__(self, qty):
        if False:
            for i in range(10):
                print('nop')
        super().__init__('Disk', 'MB', qty)

class GpuResource(Resource):
    """Represents a GPU requirement in an execution environment for an operator."""

    def __init__(self, qty):
        if False:
            i = 10
            return i + 15
        super().__init__('GPU', 'gpu(s)', qty)

class Resources:
    """
    The resources required by an operator.

    Resources that are not specified will use the default values from the airflow config.

    :param cpus: The number of cpu cores that are required
    :param ram: The amount of RAM required
    :param disk: The amount of disk space required
    :param gpus: The number of gpu units that are required
    """

    def __init__(self, cpus=conf.getint('operators', 'default_cpus'), ram=conf.getint('operators', 'default_ram'), disk=conf.getint('operators', 'default_disk'), gpus=conf.getint('operators', 'default_gpus')):
        if False:
            return 10
        self.cpus = CpuResource(cpus)
        self.ram = RamResource(ram)
        self.disk = DiskResource(disk)
        self.gpus = GpuResource(gpus)

    def __eq__(self, other):
        if False:
            return 10
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.__dict__ == other.__dict__

    def __repr__(self):
        if False:
            while True:
                i = 10
        return str(self.__dict__)

    def to_dict(self):
        if False:
            print('Hello World!')
        return {'cpus': self.cpus.to_dict(), 'ram': self.ram.to_dict(), 'disk': self.disk.to_dict(), 'gpus': self.gpus.to_dict()}

    @classmethod
    def from_dict(cls, resources_dict: dict):
        if False:
            print('Hello World!')
        'Create resources from resources dict.'
        cpus = resources_dict['cpus']['qty']
        ram = resources_dict['ram']['qty']
        disk = resources_dict['disk']['qty']
        gpus = resources_dict['gpus']['qty']
        return cls(cpus=cpus, ram=ram, disk=disk, gpus=gpus)