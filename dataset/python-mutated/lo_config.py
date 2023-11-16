"""Helper class used to convert a user LO configuration into a list of frequencies."""
from qiskit.pulse.channels import DriveChannel, MeasureChannel
from qiskit.pulse.configuration import LoConfig
from qiskit.exceptions import QiskitError

class LoConfigConverter:
    """This class supports to convert LoConfig into ~`lo_freq` attribute of configs.
    The format of LO frequency setup can be easily modified by replacing
    ``get_qubit_los`` and ``get_meas_los`` to align with your backend.
    """

    def __init__(self, qobj_model, qubit_lo_freq=None, meas_lo_freq=None, qubit_lo_range=None, meas_lo_range=None, **run_config):
        if False:
            print('Hello World!')
        'Create new converter.\n\n        Args:\n            qobj_model (Union[PulseQobjExperimentConfig, QasmQobjExperimentConfig): qobj model for\n                experiment config.\n            qubit_lo_freq (Optional[List[float]]): List of default qubit LO frequencies in Hz.\n            meas_lo_freq (Optional[List[float]]): List of default meas LO frequencies in Hz.\n            qubit_lo_range (Optional[List[List[float]]]): List of qubit LO ranges,\n                each of form ``[range_min, range_max]`` in Hz.\n            meas_lo_range (Optional[List[List[float]]]): List of measurement LO ranges,\n                each of form ``[range_min, range_max]`` in Hz.\n            n_qubits (int): Number of qubits in the system.\n            run_config (dict): experimental configuration.\n        '
        self.qobj_model = qobj_model
        self.qubit_lo_freq = qubit_lo_freq
        self.meas_lo_freq = meas_lo_freq
        self.run_config = run_config
        self.n_qubits = self.run_config.get('n_qubits', None)
        self.default_lo_config = LoConfig()
        if qubit_lo_range:
            for (i, lo_range) in enumerate(qubit_lo_range):
                self.default_lo_config.add_lo_range(DriveChannel(i), lo_range)
        if meas_lo_range:
            for (i, lo_range) in enumerate(meas_lo_range):
                self.default_lo_config.add_lo_range(MeasureChannel(i), lo_range)

    def __call__(self, user_lo_config):
        if False:
            return 10
        'Return experiment config w/ LO values property configured.\n\n        Args:\n            user_lo_config (LoConfig): A dictionary of LOs to format.\n\n        Returns:\n            Union[PulseQobjExperimentConfig, QasmQobjExperimentConfig]: Qobj experiment config.\n        '
        lo_config = {}
        q_los = self.get_qubit_los(user_lo_config)
        if q_los:
            lo_config['qubit_lo_freq'] = [freq / 1000000000.0 for freq in q_los]
        m_los = self.get_meas_los(user_lo_config)
        if m_los:
            lo_config['meas_lo_freq'] = [freq / 1000000000.0 for freq in m_los]
        return self.qobj_model(**lo_config)

    def get_qubit_los(self, user_lo_config):
        if False:
            i = 10
            return i + 15
        'Set experiment level qubit LO frequencies. Use default values from job level if\n        experiment level values not supplied. If experiment level and job level values not supplied,\n        raise an error. If configured LO frequency is the same as default, this method returns\n        ``None``.\n\n        Args:\n            user_lo_config (LoConfig): A dictionary of LOs to format.\n\n        Returns:\n            List[float]: A list of qubit LOs.\n\n        Raises:\n            QiskitError: When LO frequencies are missing and no default is set at job level.\n        '
        _q_los = None
        if self.qubit_lo_freq:
            _q_los = self.qubit_lo_freq.copy()
        elif self.n_qubits:
            _q_los = [None] * self.n_qubits
        if _q_los:
            for (channel, lo_freq) in user_lo_config.qubit_los.items():
                self.default_lo_config.check_lo(channel, lo_freq)
                _q_los[channel.index] = lo_freq
            if _q_los == self.qubit_lo_freq:
                return None
            if None in _q_los:
                raise QiskitError("Invalid experiment level qubit LO's. Must either pass values for all drive channels or pass 'default_qubit_los'.")
        return _q_los

    def get_meas_los(self, user_lo_config):
        if False:
            print('Hello World!')
        'Set experiment level meas LO frequencies. Use default values from job level if experiment\n        level values not supplied. If experiment level and job level values not supplied, raise an\n        error. If configured LO frequency is the same as default, this method returns ``None``.\n\n        Args:\n            user_lo_config (LoConfig): A dictionary of LOs to format.\n\n        Returns:\n            List[float]: A list of measurement LOs.\n\n        Raises:\n            QiskitError: When LO frequencies are missing and no default is set at job level.\n        '
        _m_los = None
        if self.meas_lo_freq:
            _m_los = self.meas_lo_freq.copy()
        elif self.n_qubits:
            _m_los = [None] * self.n_qubits
        if _m_los:
            for (channel, lo_freq) in user_lo_config.meas_los.items():
                self.default_lo_config.check_lo(channel, lo_freq)
                _m_los[channel.index] = lo_freq
            if _m_los == self.meas_lo_freq:
                return None
            if None in _m_los:
                raise QiskitError("Invalid experiment level measurement LO's. Must either pass values for all measurement channels or pass 'default_meas_los'.")
        return _m_los