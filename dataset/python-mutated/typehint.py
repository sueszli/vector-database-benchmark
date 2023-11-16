import logging
from typing import TypedDict, Any, Type, TypeVar, get_origin
from typing_extensions import Literal, TypeGuard
_logger = logging.getLogger(__name__)
T = TypeVar('T')

def typed_dict_validation(typ: Type[T], instance: Any) -> TypeGuard[T]:
    if False:
        print('Hello World!')
    if not isinstance(instance, dict):
        _logger.error('Validation failed for %s. Instance is not a dict: %s', typ, type(instance))
        return False
    for (property_name, property_type) in typ.__annotations__.items():
        if property_name not in instance:
            _logger.error('Validation failed for %s. Missing key: %s', typ, property_name)
            return False
        value = instance[property_name]
        if property_type in (int, float, bool, str):
            if not isinstance(value, property_type):
                _logger.error('Validation failed for %s. Wrong type: %s. Expected %s, got %s', typ, property_name, property_type, type(value))
                return False
        elif get_origin(property_type) == Literal:
            if value not in property_type.__args__:
                _logger.error('Validation failed for %s. Expect literal to be one of %s, got %s', typ, property_type.__args__, value)
                return False
        else:
            result = typed_dict_validation(property_type, value)
            if result is False:
                return False
    return True

class Trial(TypedDict):
    id: str
    sequence: int
    experiment: str
    command: str
    parameter: str
UpstreamCommandType = Literal['create', 'kill', 'wakeup']
DownstreamCommandType = Literal['metric', 'status', 'awake']
Status = Literal['waiting', 'running', 'succeeded', 'failed', 'interrupted']

class CreateCommand(TypedDict):
    command_type: Literal['create']
    trial: Trial

class KillCommand(TypedDict):
    command_type: Literal['kill']
    id: str

class MetricCommand(TypedDict):
    command_type: Literal['metric']
    id: str
    metric: str

class TrialStatusCommand(TypedDict):
    command_type: Literal['status']
    id: str
    status: Status

class WakeUpCommand(TypedDict):
    command_type: Literal['wakeup']

class ReportAwakeCommand(TypedDict):
    command_type: Literal['awake']
    time: float
    idle: bool