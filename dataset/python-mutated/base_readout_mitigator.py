"""
Base class for readout error mitigation.
"""
from abc import ABC, abstractmethod
from typing import Optional, List, Iterable, Tuple, Union, Callable
import numpy as np
from ..distributions.quasi import QuasiDistribution
from ..counts import Counts

class BaseReadoutMitigator(ABC):
    """Base readout error mitigator class."""

    @abstractmethod
    def quasi_probabilities(self, data: Counts, qubits: Iterable[int]=None, clbits: Optional[List[int]]=None, shots: Optional[int]=None) -> QuasiDistribution:
        if False:
            for i in range(10):
                print('nop')
        'Convert counts to a dictionary of quasi-probabilities\n\n        Args:\n            data: Counts to be mitigated.\n            qubits: the physical qubits measured to obtain the counts clbits.\n                If None these are assumed to be qubits [0, ..., N-1]\n                for N-bit counts.\n            clbits: Optional, marginalize counts to just these bits.\n            shots: Optional, the total number of shots, if None shots will\n                be calculated as the sum of all counts.\n\n        Returns:\n            QuasiDistribution: A dictionary containing pairs of [output, mean] where "output"\n                is the key in the dictionaries,\n                which is the length-N bitstring of a measured standard basis state,\n                and "mean" is the mean of non-zero quasi-probability estimates.\n        '

    @abstractmethod
    def expectation_value(self, data: Counts, diagonal: Union[Callable, dict, str, np.ndarray], qubits: Iterable[int]=None, clbits: Optional[List[int]]=None, shots: Optional[int]=None) -> Tuple[float, float]:
        if False:
            i = 10
            return i + 15
        'Calculate the expectation value of a diagonal Hermitian operator.\n\n        Args:\n            data: Counts object to be mitigated.\n            diagonal: the diagonal operator. This may either be specified\n                      as a string containing I,Z,0,1 characters, or as a\n                      real valued 1D array_like object supplying the full diagonal,\n                      or as a dictionary, or as Callable.\n            qubits: the physical qubits measured to obtain the counts clbits.\n                    If None these are assumed to be qubits [0, ..., N-1]\n                    for N-bit counts.\n            clbits: Optional, marginalize counts to just these bits.\n            shots: Optional, the total number of shots, if None shots will\n                be calculated as the sum of all counts.\n\n        Returns:\n            The mean and an upper bound of the standard deviation of operator\n            expectation value calculated from the current counts.\n        '