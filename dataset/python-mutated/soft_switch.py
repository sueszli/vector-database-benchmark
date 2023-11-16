import enum
from golem.model import GenericKeyValue
from golem.rpc import utils as rpc_utils

class SoftSwitch(enum.Enum):
    ON: str = 'on'
    OFF: str = 'off'
SOFT_SWITCH_KEY = 'concent_soft_switch'
REQUIRED_KEY = 'concent_required'

def _is_on(key: str, default: SoftSwitch) -> bool:
    if False:
        for i in range(10):
            print('nop')
    query = GenericKeyValue.select().where(GenericKeyValue.key == key).limit(1)
    try:
        value = SoftSwitch(query[0].value)
    except IndexError:
        value = default
    return value == SoftSwitch.ON

def _turn(key: str, on: bool) -> None:
    if False:
        print('Hello World!')
    (entry, _) = GenericKeyValue.get_or_create(key=key)
    entry.value = SoftSwitch.ON.value if on else SoftSwitch.OFF.value
    entry.save()

@rpc_utils.expose('golem.concent.switch')
def concent_is_on() -> bool:
    if False:
        while True:
            i = 10
    '\n    Verify if the Concent is marked as enabled within the Golem Node\n    '
    return _is_on(SOFT_SWITCH_KEY, default=SoftSwitch.OFF)

@rpc_utils.expose('golem.concent.switch.turn')
def concent_turn(on: bool) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Mark Concent as enabled/disabled within the Golem Node\n    '
    _turn(SOFT_SWITCH_KEY, on)

@rpc_utils.expose('golem.concent.required_as_provider')
def is_required_as_provider() -> bool:
    if False:
        for i in range(10):
            print('nop')
    '\n    Verify if the Concent is required for tasks accepted for computation\n    as a Provider\n    '
    return _is_on(REQUIRED_KEY, default=SoftSwitch.ON)

@rpc_utils.expose('golem.concent.required_as_provider.turn')
def required_as_provider_turn(on: bool) -> None:
    if False:
        return 10
    '\n    Mark Concent as required/not-required for tasks accepted for computation\n    '
    _turn(REQUIRED_KEY, on)