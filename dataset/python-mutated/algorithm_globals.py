"""Algorithm Globals"""
from typing import Optional
import logging
import numpy as np
from qiskit.tools import parallel
from qiskit.utils.deprecation import deprecate_func
from ..user_config import get_config
from ..exceptions import QiskitError
logger = logging.getLogger(__name__)

class QiskitAlgorithmGlobals:
    """Class for global properties."""
    CPU_COUNT = parallel.local_hardware_info()['cpus']

    @deprecate_func(additional_msg='This algorithm utility has been migrated to an independent package: https://github.com/qiskit-community/qiskit-algorithms. You can run ``pip install qiskit_algorithms`` and import ``from qiskit_algorithms.utils`` instead. ', since='0.45.0')
    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        self._random_seed = None
        self._num_processes = QiskitAlgorithmGlobals.CPU_COUNT
        self._random = None
        self._massive = False
        try:
            settings = get_config()
            self.num_processes = settings.get('num_processes', QiskitAlgorithmGlobals.CPU_COUNT)
        except Exception as ex:
            logger.debug('User Config read error %s', str(ex))

    @property
    @deprecate_func(additional_msg='This algorithm utility has been migrated to an independent package: https://github.com/qiskit-community/qiskit-algorithms. You can run ``pip install qiskit_algorithms`` and import ``from qiskit_algorithms.utils`` instead. ', since='0.45.0', is_property=True)
    def random_seed(self) -> Optional[int]:
        if False:
            for i in range(10):
                print('nop')
        'Return random seed.'
        return self._random_seed

    @random_seed.setter
    @deprecate_func(additional_msg='This algorithm utility has been migrated to an independent package: https://github.com/qiskit-community/qiskit-algorithms. You can run ``pip install qiskit_algorithms`` and import ``from qiskit_algorithms.utils`` instead. ', since='0.45.0', is_property=True)
    def random_seed(self, seed: Optional[int]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Set random seed.'
        self._random_seed = seed
        self._random = None

    @property
    @deprecate_func(additional_msg='This algorithm utility belongs to a legacy workflow and has no replacement.', since='0.45.0', is_property=True)
    def num_processes(self) -> int:
        if False:
            return 10
        'Return num processes.'
        return self._num_processes

    @num_processes.setter
    @deprecate_func(additional_msg='This algorithm utility belongs to a legacy workflow and has no replacement.', since='0.45.0', is_property=True)
    def num_processes(self, num_processes: Optional[int]) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Set num processes.\n        If 'None' is passed, it resets to QiskitAlgorithmGlobals.CPU_COUNT\n        "
        if num_processes is None:
            num_processes = QiskitAlgorithmGlobals.CPU_COUNT
        elif num_processes < 1:
            raise QiskitError(f'Invalid Number of Processes {num_processes}.')
        elif num_processes > QiskitAlgorithmGlobals.CPU_COUNT:
            raise QiskitError('Number of Processes {} cannot be greater than cpu count {}.'.format(num_processes, QiskitAlgorithmGlobals.CPU_COUNT))
        self._num_processes = num_processes
        try:
            parallel.CPU_COUNT = self.num_processes
        except Exception as ex:
            logger.warning("Failed to set qiskit.tools.parallel.CPU_COUNT to value: '%s': Error: '%s'", self.num_processes, str(ex))

    @property
    @deprecate_func(additional_msg='This algorithm utility has been migrated to an independent package: https://github.com/qiskit-community/qiskit-algorithms. You can run ``pip install qiskit_algorithms`` and import ``from qiskit_algorithms.utils`` instead. ', since='0.45.0', is_property=True)
    def random(self) -> np.random.Generator:
        if False:
            i = 10
            return i + 15
        'Return a numpy np.random.Generator (default_rng).'
        if self._random is None:
            self._random = np.random.default_rng(self._random_seed)
        return self._random

    @property
    @deprecate_func(additional_msg='This algorithm utility belongs to a legacy workflow and has no replacement.', since='0.45.0', is_property=True)
    def massive(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Return massive to allow processing of large matrices or vectors.'
        return self._massive

    @massive.setter
    @deprecate_func(additional_msg='This algorithm utility belongs to a legacy workflow and has no replacement.', since='0.45.0', is_property=True)
    def massive(self, massive: bool) -> None:
        if False:
            while True:
                i = 10
        'Set massive to allow processing of large matrices or  vectors.'
        self._massive = massive
algorithm_globals = QiskitAlgorithmGlobals()