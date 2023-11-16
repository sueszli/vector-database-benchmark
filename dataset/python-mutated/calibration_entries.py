"""Internal format of calibration data in target."""
import inspect
from abc import ABCMeta, abstractmethod
from enum import IntEnum
from typing import Callable, List, Union, Optional, Sequence, Any
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.schedule import Schedule, ScheduleBlock
from qiskit.qobj.converters import QobjToInstructionConverter
from qiskit.qobj.pulse_qobj import PulseQobjInstruction

class CalibrationPublisher(IntEnum):
    """Defines who defined schedule entry."""
    BACKEND_PROVIDER = 0
    QISKIT = 1
    EXPERIMENT_SERVICE = 2

class CalibrationEntry(metaclass=ABCMeta):
    """A metaclass of a calibration entry.

    This class defines a standard model of Qiskit pulse program that is
    agnostic to the underlying in-memory representation.

    This entry distinguishes whether this is provided by end-users or a backend
    by :attr:`.user_provided` attribute which may be provided when
    the actual calibration data is provided to the entry with by :meth:`define`.

    Note that a custom entry provided by an end-user may appear in the wire-format
    as an inline calibration, e.g. :code:`defcal` of the QASM3,
    that may update the backend instruction set architecture for execution.

    .. note::

        This and built-in subclasses are expected to be private without stable user-facing API.
        The purpose of this class is to wrap different
        in-memory pulse program representations in Qiskit, so that it can provide
        the standard data model and API which are primarily used by the transpiler ecosystem.
        It is assumed that end-users will never directly instantiate this class,
        but :class:`.Target` or :class:`.InstructionScheduleMap` internally use this data model
        to avoid implementing a complicated branching logic to
        manage different calibration data formats.

    """

    @abstractmethod
    def define(self, definition: Any, user_provided: bool):
        if False:
            for i in range(10):
                print('nop')
        'Attach definition to the calibration entry.\n\n        Args:\n            definition: Definition of this entry.\n            user_provided: If this entry is defined by user.\n                If the flag is set, this calibration may appear in the wire format\n                as an inline calibration, to override the backend instruction set architecture.\n        '
        pass

    @abstractmethod
    def get_signature(self) -> inspect.Signature:
        if False:
            return 10
        'Return signature object associated with entry definition.\n\n        Returns:\n            Signature object.\n        '
        pass

    @abstractmethod
    def get_schedule(self, *args, **kwargs) -> Union[Schedule, ScheduleBlock]:
        if False:
            print('Hello World!')
        'Generate schedule from entry definition.\n\n        If the pulse program is templated with :class:`.Parameter` objects,\n        you can provide corresponding parameter values for this method\n        to get a particular pulse program with assigned parameters.\n\n        Args:\n            args: Command parameters.\n            kwargs: Command keyword parameters.\n\n        Returns:\n            Pulse schedule with assigned parameters.\n        '
        pass

    @property
    @abstractmethod
    def user_provided(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Return if this entry is user defined.'
        pass

class ScheduleDef(CalibrationEntry):
    """In-memory Qiskit Pulse representation.

    A pulse schedule must provide signature with the .parameters attribute.
    This entry can be parameterized by a Qiskit Parameter object.
    The .get_schedule method returns a parameter-assigned pulse program.

    .. see_also::
        :class:`.CalibrationEntry` for the purpose of this class.

    """

    def __init__(self, arguments: Optional[Sequence[str]]=None):
        if False:
            while True:
                i = 10
        'Define an empty entry.\n\n        Args:\n            arguments: User provided argument names for this entry, if parameterized.\n\n        Raises:\n            PulseError: When `arguments` is not a sequence of string.\n        '
        if arguments and (not all((isinstance(arg, str) for arg in arguments))):
            raise PulseError(f'Arguments must be name of parameters. Not {arguments}.')
        if arguments:
            arguments = list(arguments)
        self._user_arguments = arguments
        self._definition = None
        self._signature = None
        self._user_provided = None

    @property
    def user_provided(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self._user_provided

    def _parse_argument(self):
        if False:
            i = 10
            return i + 15
        'Generate signature from program and user provided argument names.'
        all_argnames = {x.name for x in self._definition.parameters}
        if self._user_arguments:
            if set(self._user_arguments) != all_argnames:
                raise PulseError(f"Specified arguments don't match with schedule parameters. {self._user_arguments} != {self._definition.parameters}.")
            argnames = list(self._user_arguments)
        else:
            argnames = sorted(all_argnames)
        params = []
        for argname in argnames:
            param = inspect.Parameter(argname, kind=inspect.Parameter.POSITIONAL_OR_KEYWORD)
            params.append(param)
        signature = inspect.Signature(parameters=params, return_annotation=type(self._definition))
        self._signature = signature

    def define(self, definition: Union[Schedule, ScheduleBlock], user_provided: bool=True):
        if False:
            i = 10
            return i + 15
        self._definition = definition
        self._parse_argument()
        self._user_provided = user_provided

    def get_signature(self) -> inspect.Signature:
        if False:
            while True:
                i = 10
        return self._signature

    def get_schedule(self, *args, **kwargs) -> Union[Schedule, ScheduleBlock]:
        if False:
            while True:
                i = 10
        if not args and (not kwargs):
            out = self._definition
        else:
            try:
                to_bind = self.get_signature().bind_partial(*args, **kwargs)
            except TypeError as ex:
                raise PulseError("Assigned parameter doesn't match with schedule parameters.") from ex
            value_dict = {}
            for param in self._definition.parameters:
                try:
                    value_dict[param] = to_bind.arguments[param.name]
                except KeyError:
                    pass
            out = self._definition.assign_parameters(value_dict, inplace=False)
        if 'publisher' not in out.metadata:
            if self.user_provided:
                out.metadata['publisher'] = CalibrationPublisher.QISKIT
            else:
                out.metadata['publisher'] = CalibrationPublisher.BACKEND_PROVIDER
        return out

    def __eq__(self, other):
        if False:
            return 10
        if hasattr(other, '_definition'):
            return self._definition == other._definition
        return False

    def __str__(self):
        if False:
            print('Hello World!')
        out = f'Schedule {self._definition.name}'
        params_str = ', '.join(self.get_signature().parameters.keys())
        if params_str:
            out += f'({params_str})'
        return out

class CallableDef(CalibrationEntry):
    """Python callback function that generates Qiskit Pulse program.

    A callable is inspected by the python built-in inspection module and
    provide the signature. This entry is parameterized by the function signature
    and .get_schedule method returns a non-parameterized pulse program
    by consuming the provided arguments and keyword arguments.

    .. see_also::
        :class:`.CalibrationEntry` for the purpose of this class.

    """

    def __init__(self):
        if False:
            while True:
                i = 10
        'Define an empty entry.'
        self._definition = None
        self._signature = None
        self._user_provided = None

    @property
    def user_provided(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self._user_provided

    def define(self, definition: Callable, user_provided: bool=True):
        if False:
            while True:
                i = 10
        self._definition = definition
        self._signature = inspect.signature(definition)
        self._user_provided = user_provided

    def get_signature(self) -> inspect.Signature:
        if False:
            i = 10
            return i + 15
        return self._signature

    def get_schedule(self, *args, **kwargs) -> Union[Schedule, ScheduleBlock]:
        if False:
            while True:
                i = 10
        try:
            to_bind = self._signature.bind(*args, **kwargs)
            to_bind.apply_defaults()
        except TypeError as ex:
            raise PulseError("Assigned parameter doesn't match with function signature.") from ex
        out = self._definition(**to_bind.arguments)
        if 'publisher' not in out.metadata:
            if self.user_provided:
                out.metadata['publisher'] = CalibrationPublisher.QISKIT
            else:
                out.metadata['publisher'] = CalibrationPublisher.BACKEND_PROVIDER
        return out

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        if hasattr(other, '_definition'):
            return self._definition == other._definition
        return False

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        params_str = ', '.join(self.get_signature().parameters.keys())
        return f'Callable {self._definition.__name__}({params_str})'

class PulseQobjDef(ScheduleDef):
    """Qobj JSON serialized format instruction sequence.

    A JSON serialized program can be converted into Qiskit Pulse program with
    the provided qobj converter. Because the Qobj JSON doesn't provide signature,
    conversion process occurs when the signature is requested for the first time
    and the generated pulse program is cached for performance.

    .. see_also::
        :class:`.CalibrationEntry` for the purpose of this class.

    """

    def __init__(self, arguments: Optional[Sequence[str]]=None, converter: Optional[QobjToInstructionConverter]=None, name: Optional[str]=None):
        if False:
            return 10
        'Define an empty entry.\n\n        Args:\n            arguments: User provided argument names for this entry, if parameterized.\n            converter: Optional. Qobj to Qiskit converter.\n            name: Name of schedule.\n        '
        super().__init__(arguments=arguments)
        self._converter = converter or QobjToInstructionConverter(pulse_library=[])
        self._name = name
        self._source = None

    def _build_schedule(self):
        if False:
            i = 10
            return i + 15
        'Build pulse schedule from cmd-def sequence.'
        schedule = Schedule(name=self._name)
        for qobj_inst in self._source:
            for qiskit_inst in self._converter._get_sequences(qobj_inst):
                schedule.insert(qobj_inst.t0, qiskit_inst, inplace=True)
        self._definition = schedule
        self._parse_argument()

    def define(self, definition: List[PulseQobjInstruction], user_provided: bool=False):
        if False:
            return 10
        self._source = definition
        self._user_provided = user_provided

    def get_signature(self) -> inspect.Signature:
        if False:
            for i in range(10):
                print('nop')
        if self._definition is None:
            self._build_schedule()
        return super().get_signature()

    def get_schedule(self, *args, **kwargs) -> Union[Schedule, ScheduleBlock]:
        if False:
            while True:
                i = 10
        if self._definition is None:
            self._build_schedule()
        return super().get_schedule(*args, **kwargs)

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other, PulseQobjDef):
            return self._source == other._source
        if isinstance(other, ScheduleDef) and self._definition is None:
            self._build_schedule()
        if hasattr(other, '_definition'):
            return self._definition == other._definition
        return False

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        if self._definition is None:
            return 'PulseQobj'
        return super().__str__()