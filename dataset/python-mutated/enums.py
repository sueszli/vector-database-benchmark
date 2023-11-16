from enum import Enum

class StateLists(Enum):
    """
    The set of StateLists tracked in ManticoreBase
    """
    ready = 'READY'
    busy = 'BUSY'
    terminated = 'TERMINATED'
    killed = 'KILLED'

class StateStatus(Enum):
    """
    Statuses that a StateDescriptor can have
    """
    waiting_for_worker = 'waiting_for_worker'
    waiting_for_solver = 'waiting_for_solver'
    running = 'running'
    stopped = 'stopped'
    destroyed = 'destroyed'

class MProcessingType(Enum):
    """Used as configuration constant for choosing multiprocessing flavor"""
    multiprocessing = 'multiprocessing'
    single = 'single'
    threading = 'threading'

    def title(self):
        if False:
            i = 10
            return i + 15
        return self._name_.title()

    @classmethod
    def from_string(cls, name):
        if False:
            i = 10
            return i + 15
        return cls.__members__[name]

    def to_class(self):
        if False:
            for i in range(10):
                print('nop')
        return globals()[f'Manticore{self.title()}']

class Sha3Type(Enum):
    """Used as configuration constant for choosing sha3 flavor"""
    concretize = 'concretize'
    symbolicate = 'symbolicate'
    fake = 'fake'

    def title(self):
        if False:
            while True:
                i = 10
        return self._name_.title()

    @classmethod
    def from_string(cls, name):
        if False:
            print('Hello World!')
        return cls.__members__[name]

class DetectorClassification(Enum):
    """
    Shall be consistent with
    https://github.com/trailofbits/slither/blob/563d5118298e4cae7f0ea5f2a531f0dcdcebd64d/slither/detectors/abstract_detector.py#L11-L15
    """
    HIGH = 0
    MEDIUM = 1
    LOW = 2
    INFORMATIONAL = 3