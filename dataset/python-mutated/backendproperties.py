"""Backend Properties classes."""
import copy
import datetime
from typing import Any, Iterable, Tuple, Union
import dateutil.parser
from qiskit.providers.exceptions import BackendPropertyError
from qiskit.utils.units import apply_prefix

class Nduv:
    """Class representing name-date-unit-value

    Attributes:
        date: date.
        name: name.
        unit: unit.
        value: value.
    """

    def __init__(self, date, name, unit, value):
        if False:
            print('Hello World!')
        'Initialize a new name-date-unit-value object\n\n        Args:\n            date (datetime.datetime): Date field\n            name (str): Name field\n            unit (str): Nduv unit\n            value (float): The value of the Nduv\n        '
        self.date = date
        self.name = name
        self.unit = unit
        self.value = value

    @classmethod
    def from_dict(cls, data):
        if False:
            for i in range(10):
                print('nop')
        'Create a new Nduv object from a dictionary.\n\n        Args:\n            data (dict): A dictionary representing the Nduv to create.\n                         It will be in the same format as output by\n                         :func:`to_dict`.\n\n        Returns:\n            Nduv: The Nduv from the input dictionary.\n        '
        return cls(**data)

    def to_dict(self):
        if False:
            while True:
                i = 10
        'Return a dictionary format representation of the object.\n\n        Returns:\n            dict: The dictionary form of the Nduv.\n        '
        out_dict = {'date': self.date, 'name': self.name, 'unit': self.unit, 'value': self.value}
        return out_dict

    def __eq__(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, Nduv):
            if self.to_dict() == other.to_dict():
                return True
        return False

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'Nduv({repr(self.date)}, {self.name}, {self.unit}, {self.value})'

class GateProperties:
    """Class representing a gate's properties

    Attributes:
        qubits: qubits.
        gate: gate.
        parameters: parameters.
    """
    _data = {}

    def __init__(self, qubits, gate, parameters, **kwargs):
        if False:
            return 10
        'Initialize a new :class:`GateProperties` object\n\n        Args:\n            qubits (list): A list of integers representing qubits\n            gate (str): The gates name\n            parameters (list): List of :class:`Nduv` objects for the\n                name-date-unit-value for the gate\n            kwargs: Optional additional fields\n        '
        self._data = {}
        self.qubits = qubits
        self.gate = gate
        self.parameters = parameters
        self._data.update(kwargs)

    def __getattr__(self, name):
        if False:
            i = 10
            return i + 15
        try:
            return self._data[name]
        except KeyError as ex:
            raise AttributeError(f'Attribute {name} is not defined') from ex

    @classmethod
    def from_dict(cls, data):
        if False:
            return 10
        'Create a new Gate object from a dictionary.\n\n        Args:\n            data (dict): A dictionary representing the Gate to create.\n                         It will be in the same format as output by\n                         :func:`to_dict`.\n\n        Returns:\n            GateProperties: The Nduv from the input dictionary.\n        '
        in_data = {}
        for (key, value) in data.items():
            if key == 'parameters':
                in_data[key] = list(map(Nduv.from_dict, value))
            else:
                in_data[key] = value
        return cls(**in_data)

    def to_dict(self):
        if False:
            while True:
                i = 10
        'Return a dictionary format representation of the BackendStatus.\n\n        Returns:\n            dict: The dictionary form of the Gate.\n        '
        out_dict = {}
        out_dict['qubits'] = self.qubits
        out_dict['gate'] = self.gate
        out_dict['parameters'] = [x.to_dict() for x in self.parameters]
        out_dict.update(self._data)
        return out_dict

    def __eq__(self, other):
        if False:
            return 10
        if isinstance(other, GateProperties):
            if self.to_dict() == other.to_dict():
                return True
        return False
Gate = GateProperties

class BackendProperties:
    """Class representing backend properties

    This holds backend properties measured by the provider. All properties
    which are provided optionally. These properties may describe qubits, gates,
    or other general properties of the backend.
    """
    _data = {}

    def __init__(self, backend_name, backend_version, last_update_date, qubits, gates, general, **kwargs):
        if False:
            while True:
                i = 10
        'Initialize a BackendProperties instance.\n\n        Args:\n            backend_name (str): Backend name.\n            backend_version (str): Backend version in the form X.Y.Z.\n            last_update_date (datetime.datetime or str): Last date/time that a property was\n                updated. If specified as a ``str``, it must be in ISO format.\n            qubits (list): System qubit parameters as a list of lists of\n                           :class:`Nduv` objects\n            gates (list): System gate parameters as a list of :class:`GateProperties`\n                          objects\n            general (list): General parameters as a list of :class:`Nduv`\n                            objects\n            kwargs: optional additional fields\n        '
        self._data = {}
        self.backend_name = backend_name
        self.backend_version = backend_version
        if isinstance(last_update_date, str):
            last_update_date = dateutil.parser.isoparse(last_update_date)
        self.last_update_date = last_update_date
        self.general = general
        self.qubits = qubits
        self.gates = gates
        self._qubits = {}
        for (qubit, props) in enumerate(qubits):
            formatted_props = {}
            for prop in props:
                value = self._apply_prefix(prop.value, prop.unit)
                formatted_props[prop.name] = (value, prop.date)
                self._qubits[qubit] = formatted_props
        self._gates = {}
        for gate in gates:
            if gate.gate not in self._gates:
                self._gates[gate.gate] = {}
            formatted_props = {}
            for param in gate.parameters:
                value = self._apply_prefix(param.value, param.unit)
                formatted_props[param.name] = (value, param.date)
            self._gates[gate.gate][tuple(gate.qubits)] = formatted_props
        self._data.update(kwargs)

    def __getattr__(self, name):
        if False:
            while True:
                i = 10
        try:
            return self._data[name]
        except KeyError as ex:
            raise AttributeError(f'Attribute {name} is not defined') from ex

    @classmethod
    def from_dict(cls, data):
        if False:
            print('Hello World!')
        'Create a new BackendProperties object from a dictionary.\n\n        Args:\n            data (dict): A dictionary representing the BackendProperties to create.  It will be in\n                the same format as output by :meth:`to_dict`.\n\n        Returns:\n            BackendProperties: The BackendProperties from the input dictionary.\n        '
        in_data = copy.copy(data)
        backend_name = in_data.pop('backend_name')
        backend_version = in_data.pop('backend_version')
        last_update_date = in_data.pop('last_update_date')
        qubits = []
        for qubit in in_data.pop('qubits'):
            nduvs = []
            for nduv in qubit:
                nduvs.append(Nduv.from_dict(nduv))
            qubits.append(nduvs)
        gates = [GateProperties.from_dict(x) for x in in_data.pop('gates')]
        general = [Nduv.from_dict(x) for x in in_data.pop('general')]
        return cls(backend_name, backend_version, last_update_date, qubits, gates, general, **in_data)

    def to_dict(self):
        if False:
            while True:
                i = 10
        'Return a dictionary format representation of the BackendProperties.\n\n        Returns:\n            dict: The dictionary form of the BackendProperties.\n        '
        out_dict = {'backend_name': self.backend_name, 'backend_version': self.backend_version, 'last_update_date': self.last_update_date}
        out_dict['qubits'] = []
        for qubit in self.qubits:
            qubit_props = []
            for item in qubit:
                qubit_props.append(item.to_dict())
            out_dict['qubits'].append(qubit_props)
        out_dict['gates'] = [x.to_dict() for x in self.gates]
        out_dict['general'] = [x.to_dict() for x in self.general]
        out_dict.update(self._data)
        return out_dict

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        if isinstance(other, BackendProperties):
            if self.to_dict() == other.to_dict():
                return True
        return False

    def gate_property(self, gate: str, qubits: Union[int, Iterable[int]]=None, name: str=None) -> Tuple[Any, datetime.datetime]:
        if False:
            return 10
        '\n        Return the property of the given gate.\n\n        Args:\n            gate: Name of the gate.\n            qubits: The qubit to find the property for.\n            name: Optionally used to specify which gate property to return.\n\n        Returns:\n            Gate property as a tuple of the value and the time it was measured.\n\n        Raises:\n            BackendPropertyError: If the property is not found or name is\n                                  specified but qubit is not.\n        '
        try:
            result = self._gates[gate]
            if qubits is not None:
                if isinstance(qubits, int):
                    qubits = (qubits,)
                result = result[tuple(qubits)]
                if name:
                    result = result[name]
            elif name:
                raise BackendPropertyError(f'Provide qubits to get {name} of {gate}')
        except KeyError as ex:
            raise BackendPropertyError(f'Could not find the desired property for {gate}') from ex
        return result

    def faulty_qubits(self):
        if False:
            print('Hello World!')
        'Return a list of faulty qubits.'
        faulty = []
        for qubit in self._qubits:
            if not self.is_qubit_operational(qubit):
                faulty.append(qubit)
        return faulty

    def faulty_gates(self):
        if False:
            i = 10
            return i + 15
        'Return a list of faulty gates.'
        faulty = []
        for gate in self.gates:
            if not self.is_gate_operational(gate.gate, gate.qubits):
                faulty.append(gate)
        return faulty

    def is_gate_operational(self, gate: str, qubits: Union[int, Iterable[int]]=None) -> bool:
        if False:
            print('Hello World!')
        '\n        Return the operational status of the given gate.\n\n        Args:\n            gate: Name of the gate.\n            qubits: The qubit to find the operational status for.\n\n        Returns:\n            bool: Operational status of the given gate. True if the gate is operational,\n            False otherwise.\n        '
        properties = self.gate_property(gate, qubits)
        if 'operational' in properties:
            return bool(properties['operational'][0])
        return True

    def gate_error(self, gate: str, qubits: Union[int, Iterable[int]]) -> float:
        if False:
            print('Hello World!')
        '\n        Return gate error estimates from backend properties.\n\n        Args:\n            gate: The gate for which to get the error.\n            qubits: The specific qubits for the gate.\n\n        Returns:\n            Gate error of the given gate and qubit(s).\n        '
        return self.gate_property(gate, qubits, 'gate_error')[0]

    def gate_length(self, gate: str, qubits: Union[int, Iterable[int]]) -> float:
        if False:
            while True:
                i = 10
        '\n        Return the duration of the gate in units of seconds.\n\n        Args:\n            gate: The gate for which to get the duration.\n            qubits: The specific qubits for the gate.\n\n        Returns:\n            Gate length of the given gate and qubit(s).\n        '
        return self.gate_property(gate, qubits, 'gate_length')[0]

    def qubit_property(self, qubit: int, name: str=None) -> Tuple[Any, datetime.datetime]:
        if False:
            while True:
                i = 10
        '\n        Return the property of the given qubit.\n\n        Args:\n            qubit: The property to look for.\n            name: Optionally used to specify within the hierarchy which property to return.\n\n        Returns:\n            Qubit property as a tuple of the value and the time it was measured.\n\n        Raises:\n            BackendPropertyError: If the property is not found.\n        '
        try:
            result = self._qubits[qubit]
            if name is not None:
                result = result[name]
        except KeyError as ex:
            raise BackendPropertyError("Couldn't find the propert{name} for qubit {qubit}.".format(name="y '" + name + "'" if name else 'ies', qubit=qubit)) from ex
        return result

    def t1(self, qubit: int) -> float:
        if False:
            while True:
                i = 10
        '\n        Return the T1 time of the given qubit.\n\n        Args:\n            qubit: Qubit for which to return the T1 time of.\n\n        Returns:\n            T1 time of the given qubit.\n        '
        return self.qubit_property(qubit, 'T1')[0]

    def t2(self, qubit: int) -> float:
        if False:
            print('Hello World!')
        '\n        Return the T2 time of the given qubit.\n\n        Args:\n            qubit: Qubit for which to return the T2 time of.\n\n        Returns:\n            T2 time of the given qubit.\n        '
        return self.qubit_property(qubit, 'T2')[0]

    def frequency(self, qubit: int) -> float:
        if False:
            i = 10
            return i + 15
        '\n        Return the frequency of the given qubit.\n\n        Args:\n            qubit: Qubit for which to return frequency of.\n\n        Returns:\n            Frequency of the given qubit.\n        '
        return self.qubit_property(qubit, 'frequency')[0]

    def readout_error(self, qubit: int) -> float:
        if False:
            return 10
        '\n        Return the readout error of the given qubit.\n\n        Args:\n            qubit: Qubit for which to return the readout error of.\n\n        Return:\n            Readout error of the given qubit.\n        '
        return self.qubit_property(qubit, 'readout_error')[0]

    def readout_length(self, qubit: int) -> float:
        if False:
            print('Hello World!')
        '\n        Return the readout length [sec] of the given qubit.\n\n        Args:\n            qubit: Qubit for which to return the readout length of.\n\n        Return:\n            Readout length of the given qubit.\n        '
        return self.qubit_property(qubit, 'readout_length')[0]

    def is_qubit_operational(self, qubit: int) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the operational status of the given qubit.\n\n        Args:\n            qubit: Qubit for which to return operational status of.\n\n        Returns:\n            Operational status of the given qubit.\n        '
        properties = self.qubit_property(qubit)
        if 'operational' in properties:
            return bool(properties['operational'][0])
        return True

    def _apply_prefix(self, value: float, unit: str) -> float:
        if False:
            for i in range(10):
                print('nop')
        "\n        Given a SI unit prefix and value, apply the prefix to convert to\n        standard SI unit.\n\n        Args:\n            value: The number to apply prefix to.\n            unit: String prefix.\n\n        Returns:\n            Converted value.\n\n        Raises:\n            BackendPropertyError: If the units aren't recognized.\n        "
        try:
            return apply_prefix(value, unit)
        except Exception as ex:
            raise BackendPropertyError(f'Could not understand units: {unit}') from ex