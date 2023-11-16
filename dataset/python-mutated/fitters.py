"""
Measurement correction fitters.
"""
from typing import List
import copy
import re
import numpy as np
from qiskit import QiskitError
from qiskit.utils.mitigation.circuits import count_keys
from qiskit.utils.mitigation._filters import MeasurementFilter, TensoredFilter
from qiskit.utils.deprecation import deprecate_func

class CompleteMeasFitter:
    """
    Deprecated: Measurement correction fitter for a full calibration
    """

    @deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/qi_migration.')
    def __init__(self, results, state_labels: List[str], qubit_list: List[int]=None, circlabel: str=''):
        if False:
            print('Hello World!')
        "\n        Initialize a measurement calibration matrix from the results of running\n        the circuits returned by `measurement_calibration_circuits`\n\n        A wrapper for the tensored fitter\n\n        .. warning::\n\n            This class is not a public API. The internals are not stable and will\n            likely change. It is used solely for the\n            ``measurement_error_mitigation_cls`` kwarg of the\n            :class:`~qiskit.utils.QuantumInstance` class's constructor (as\n            a class not an instance). Anything outside of that usage does\n            not have the normal user-facing API stability.\n\n        Args:\n            results: the results of running the measurement calibration\n                circuits. If this is `None` the user will set a calibration\n                matrix later.\n            state_labels: list of calibration state labels\n                returned from `measurement_calibration_circuits`.\n                The output matrix will obey this ordering.\n            qubit_list: List of the qubits (for reference and if the\n                subset is needed). If `None`, the qubit_list will be\n                created according to the length of state_labels[0].\n            circlabel: if the qubits were labeled.\n        "
        if qubit_list is None:
            qubit_list = range(len(state_labels[0]))
        self._qubit_list = qubit_list
        self._tens_fitt = TensoredMeasFitter(results, [qubit_list], [state_labels], circlabel)

    @property
    def cal_matrix(self):
        if False:
            while True:
                i = 10
        'Return cal_matrix.'
        return self._tens_fitt.cal_matrices[0]

    @cal_matrix.setter
    def cal_matrix(self, new_cal_matrix):
        if False:
            i = 10
            return i + 15
        'set cal_matrix.'
        self._tens_fitt.cal_matrices = [copy.deepcopy(new_cal_matrix)]

    @property
    def state_labels(self):
        if False:
            i = 10
            return i + 15
        'Return state_labels.'
        return self._tens_fitt.substate_labels_list[0]

    @property
    def qubit_list(self):
        if False:
            print('Hello World!')
        'Return list of qubits.'
        return self._qubit_list

    @state_labels.setter
    def state_labels(self, new_state_labels):
        if False:
            print('Hello World!')
        'Set state label.'
        self._tens_fitt.substate_labels_list[0] = new_state_labels

    @property
    def filter(self):
        if False:
            while True:
                i = 10
        'Return a measurement filter using the cal matrix.'
        return MeasurementFilter(self.cal_matrix, self.state_labels)

    def add_data(self, new_results, rebuild_cal_matrix=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        Add measurement calibration data\n\n        Args:\n            new_results (list or qiskit.result.Result): a single result or list\n                of result objects.\n            rebuild_cal_matrix (bool): rebuild the calibration matrix\n        '
        self._tens_fitt.add_data(new_results, rebuild_cal_matrix)

    def subset_fitter(self, qubit_sublist):
        if False:
            while True:
                i = 10
        '\n        Return a fitter object that is a subset of the qubits in the original\n        list.\n\n        Args:\n            qubit_sublist (list): must be a subset of qubit_list\n\n        Returns:\n            CompleteMeasFitter: A new fitter that has the calibration for a\n                subset of qubits\n\n        Raises:\n            QiskitError: If the calibration matrix is not initialized\n        '
        if self._tens_fitt.cal_matrices is None:
            raise QiskitError('Calibration matrix is not initialized')
        if qubit_sublist is None:
            raise QiskitError('Qubit sublist must be specified')
        for qubit in qubit_sublist:
            if qubit not in self._qubit_list:
                raise QiskitError('Qubit not in the original set of qubits')
        new_state_labels = count_keys(len(qubit_sublist))
        qubit_sublist_ind = []
        for sqb in qubit_sublist:
            for (qbind, qubit) in enumerate(self._qubit_list):
                if qubit == sqb:
                    qubit_sublist_ind.append(qbind)
        q_q_mapping = []
        state_labels_reduced = []
        for label in self.state_labels:
            tmplabel = [label[index] for index in qubit_sublist_ind]
            state_labels_reduced.append(''.join(tmplabel))
        for (sub_lab_ind, _) in enumerate(new_state_labels):
            q_q_mapping.append([])
            for (labelind, label) in enumerate(state_labels_reduced):
                if label == new_state_labels[sub_lab_ind]:
                    q_q_mapping[-1].append(labelind)
        new_fitter = CompleteMeasFitter(results=None, state_labels=new_state_labels, qubit_list=qubit_sublist)
        new_cal_matrix = np.zeros([len(new_state_labels), len(new_state_labels)])
        for i in range(len(new_state_labels)):
            for j in range(len(new_state_labels)):
                for q_q_i_map in q_q_mapping[i]:
                    for q_q_j_map in q_q_mapping[j]:
                        new_cal_matrix[i, j] += self.cal_matrix[q_q_i_map, q_q_j_map]
                new_cal_matrix[i, j] /= len(q_q_mapping[i])
        new_fitter.cal_matrix = new_cal_matrix
        return new_fitter

    def readout_fidelity(self, label_list=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Based on the results, output the readout fidelity which is the\n        normalized trace of the calibration matrix\n\n        Args:\n            label_list (bool): If `None`, returns the average assignment fidelity\n                of a single state. Otherwise it returns the assignment fidelity\n                to be in any one of these states averaged over the second\n                index.\n\n        Returns:\n            numpy.array: readout fidelity (assignment fidelity)\n\n        Additional Information:\n            The on-diagonal elements of the calibration matrix are the\n            probabilities of measuring state 'x' given preparation of state\n            'x' and so the normalized trace is the average assignment fidelity\n        "
        return self._tens_fitt.readout_fidelity(0, label_list)

class TensoredMeasFitter:
    """
    Deprecated: Measurement correction fitter for a tensored calibration.
    """

    @deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/qi_migration.')
    def __init__(self, results, mit_pattern: List[List[int]], substate_labels_list: List[List[str]]=None, circlabel: str=''):
        if False:
            return 10
        "\n        Initialize a measurement calibration matrix from the results of running\n        the circuits returned by `measurement_calibration_circuits`.\n\n        .. warning::\n\n            This class is not a public API. The internals are not stable and will\n            likely change. It is used solely for the\n            ``measurement_error_mitigation_cls`` kwarg of the\n            :class:`~qiskit.utils.QuantumInstance` class's constructor (as\n            a class not an instance). Anything outside of that usage does\n            not have the normal user-facing API stability.\n\n        Args:\n            results: the results of running the measurement calibration\n                circuits. If this is `None`, the user will set calibration\n                matrices later.\n\n            mit_pattern: qubits to perform the\n                measurement correction on, divided to groups according to\n                tensors\n\n            substate_labels_list: for each\n                calibration matrix, the labels of its rows and columns.\n                If `None`, the labels are ordered lexicographically\n\n            circlabel: if the qubits were labeled\n\n        Raises:\n            ValueError: if the mit_pattern doesn't match the\n                substate_labels_list\n        "
        self._result_list = []
        self._cal_matrices = None
        self._circlabel = circlabel
        self._mit_pattern = mit_pattern
        self._qubit_list_sizes = [len(qubit_list) for qubit_list in mit_pattern]
        self._indices_list = []
        if substate_labels_list is None:
            self._substate_labels_list = []
            for list_size in self._qubit_list_sizes:
                self._substate_labels_list.append(count_keys(list_size))
        else:
            self._substate_labels_list = substate_labels_list
            if len(self._qubit_list_sizes) != len(substate_labels_list):
                raise ValueError('mit_pattern does not match substate_labels_list')
        self._indices_list = []
        for (_, sub_labels) in enumerate(self._substate_labels_list):
            self._indices_list.append({lab: ind for (ind, lab) in enumerate(sub_labels)})
        self.add_data(results)

    @property
    def cal_matrices(self):
        if False:
            print('Hello World!')
        'Return cal_matrices.'
        return self._cal_matrices

    @cal_matrices.setter
    def cal_matrices(self, new_cal_matrices):
        if False:
            return 10
        'Set _cal_matrices.'
        self._cal_matrices = copy.deepcopy(new_cal_matrices)

    @property
    def substate_labels_list(self):
        if False:
            while True:
                i = 10
        'Return _substate_labels_list.'
        return self._substate_labels_list

    @property
    def filter(self):
        if False:
            print('Hello World!')
        'Return a measurement filter using the cal matrices.'
        return TensoredFilter(self._cal_matrices, self._substate_labels_list, self._mit_pattern)

    @property
    def nqubits(self):
        if False:
            i = 10
            return i + 15
        'Return _qubit_list_sizes.'
        return sum(self._qubit_list_sizes)

    def add_data(self, new_results, rebuild_cal_matrix=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        Add measurement calibration data\n\n        Args:\n            new_results (list or qiskit.result.Result): a single result or list\n                of Result objects.\n            rebuild_cal_matrix (bool): rebuild the calibration matrix\n        '
        if new_results is None:
            return
        if not isinstance(new_results, list):
            new_results = [new_results]
        for result in new_results:
            self._result_list.append(result)
        if rebuild_cal_matrix:
            self._build_calibration_matrices()

    def readout_fidelity(self, cal_index=0, label_list=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Based on the results, output the readout fidelity, which is the average\n        of the diagonal entries in the calibration matrices.\n\n        Args:\n            cal_index(integer): readout fidelity for this index in _cal_matrices\n            label_list (list):  Returns the average fidelity over of the groups\n                f states. In the form of a list of lists of states. If `None`,\n                then each state used in the construction of the calibration\n                matrices forms a group of size 1\n\n        Returns:\n            numpy.array: The readout fidelity (assignment fidelity)\n\n        Raises:\n            QiskitError: If the calibration matrix has not been set for the\n                object.\n\n        Additional Information:\n            The on-diagonal elements of the calibration matrices are the\n            probabilities of measuring state 'x' given preparation of state\n            'x'.\n        "
        if self._cal_matrices is None:
            raise QiskitError('Cal matrix has not been set')
        if label_list is None:
            label_list = [[label] for label in self._substate_labels_list[cal_index]]
        state_labels = self._substate_labels_list[cal_index]
        fidelity_label_list = []
        if label_list is None:
            fidelity_label_list = [[label] for label in state_labels]
        else:
            for fid_sublist in label_list:
                fidelity_label_list.append([])
                for fid_statelabl in fid_sublist:
                    for (label_idx, label) in enumerate(state_labels):
                        if fid_statelabl == label:
                            fidelity_label_list[-1].append(label_idx)
                            continue
        assign_fid_list = []
        for fid_label_sublist in fidelity_label_list:
            assign_fid_list.append(0)
            for state_idx_i in fid_label_sublist:
                for state_idx_j in fid_label_sublist:
                    assign_fid_list[-1] += self._cal_matrices[cal_index][state_idx_i][state_idx_j]
            assign_fid_list[-1] /= len(fid_label_sublist)
        return np.mean(assign_fid_list)

    def _build_calibration_matrices(self):
        if False:
            while True:
                i = 10
        '\n        Build the measurement calibration matrices from the results of running\n        the circuits returned by `measurement_calibration`.\n        '
        self._cal_matrices = []
        for list_size in self._qubit_list_sizes:
            self._cal_matrices.append(np.zeros([2 ** list_size, 2 ** list_size], dtype=float))
        for result in self._result_list:
            for experiment in result.results:
                circ_name = experiment.header.name
                circ_search = re.search('(?<=' + self._circlabel + 'cal_)\\w+', circ_name)
                if circ_search is None:
                    continue
                state = circ_search.group(0)
                state_cnts = result.get_counts(circ_name)
                for (measured_state, counts) in state_cnts.items():
                    end_index = self.nqubits
                    for (cal_ind, cal_mat) in enumerate(self._cal_matrices):
                        start_index = end_index - self._qubit_list_sizes[cal_ind]
                        substate_index = self._indices_list[cal_ind][state[start_index:end_index]]
                        measured_substate_index = self._indices_list[cal_ind][measured_state[start_index:end_index]]
                        end_index = start_index
                        cal_mat[measured_substate_index][substate_index] += counts
        for (mat_index, _) in enumerate(self._cal_matrices):
            sums_of_columns = np.sum(self._cal_matrices[mat_index], axis=0)
            self._cal_matrices[mat_index] = np.divide(self._cal_matrices[mat_index], sums_of_columns, out=np.zeros_like(self._cal_matrices[mat_index]), where=sums_of_columns != 0)

    def subset_fitter(self, qubit_sublist):
        if False:
            i = 10
            return i + 15
        'Return a fitter object that is a subset of the qubits in the original list.\n\n        This is only a partial implementation of the ``subset_fitter`` method since only\n        mitigation patterns of length 1 are supported. This corresponds to patterns of the\n        form ``[[0], [1], [2], ...]``. Note however, that such patterns are a good first\n        approximation to mitigate readout errors on large quantum circuits.\n\n        Args:\n            qubit_sublist (list): must be a subset of qubit_list\n\n        Returns:\n            TensoredMeasFitter: A new fitter that has the calibration for a\n                subset of qubits\n\n        Raises:\n            QiskitError: If the calibration matrix is not initialized\n            QiskitError: If the mit pattern is not a tensor of single-qubit\n                measurement error mitigation.\n            QiskitError: If a qubit in the given ``qubit_sublist`` is not in the list of\n                qubits in the mit. pattern.\n        '
        if self._cal_matrices is None:
            raise QiskitError('Calibration matrices are not initialized.')
        if qubit_sublist is None:
            raise QiskitError('Qubit sublist must be specified.')
        if not all((len(tensor) == 1 for tensor in self._mit_pattern)):
            raise QiskitError(f'Each element in the mit pattern should have length 1. Found {self._mit_pattern}.')
        supported_qubits = {tensor[0] for tensor in self._mit_pattern}
        for qubit in qubit_sublist:
            if qubit not in supported_qubits:
                raise QiskitError(f'Qubit {qubit} is not in the mit pattern {self._mit_pattern}.')
        new_mit_pattern = [[idx] for idx in qubit_sublist]
        new_substate_labels_list = [self._substate_labels_list[idx] for idx in qubit_sublist]
        new_fitter = TensoredMeasFitter(results=None, mit_pattern=new_mit_pattern, substate_labels_list=new_substate_labels_list)
        new_fitter.cal_matrices = [self._cal_matrices[idx] for idx in qubit_sublist]
        return new_fitter