from abc import ABC, abstractmethod
from contextlib import contextmanager
from enum import Enum, unique
from functools import total_ordering, wraps
from itertools import count
from .mycroft_skill import MycroftSkill
from mycroft.messagebus.message import Message, dig_for_message
ENTITY = 'ENTITY'
SCENE = 'SCENE'
IOT_REQUEST_ID = 'iot_request_id'
_counter = count()

def auto():
    if False:
        print('Hello World!')
    '\n    Indefinitely return the next number in sequence from 0.\n\n    This can be replaced with enum.auto when we no longer\n    need to support python3.4.\n    '
    return next(_counter)

class _BusKeys:
    """
    This class contains some strings used to identify
    messages on the messagebus. They are used in in
    CommonIoTSkill and the IoTController skill, but
    are not intended to be used elsewhere.
    """
    BASE = 'iot'
    TRIGGER = BASE + ':trigger'
    RESPONSE = TRIGGER + '.response'
    RUN = BASE + ':run.'
    REGISTER = BASE + 'register'
    CALL_FOR_REGISTRATION = REGISTER + '.request'
    SPEAK = BASE + ':speak'

@unique
class Thing(Enum):
    """
    This class represents 'Things' which may be controlled
    by IoT Skills. This is intended to be used with the
    IoTRequest class. See that class for more details.
    """
    LIGHT = auto()
    THERMOSTAT = auto()
    DOOR = auto()
    LOCK = auto()
    PLUG = auto()
    SWITCH = auto()
    TEMPERATURE = auto()
    HEAT = auto()
    AIR_CONDITIONING = auto()

@unique
class Attribute(Enum):
    """
    This class represents 'Attributes' of 'Things'.
    """
    BRIGHTNESS = auto()
    COLOR = auto()
    COLOR_TEMPERATURE = auto()
    TEMPERATURE = auto()

@unique
class State(Enum):
    """
    This class represents 'States' of 'Things'.

    These are generally intended to handle binary
    queries, such as "is the door locked?" or
    "is the heat on?" where 'locked' and 'on'
    are the state values. The special value
    'STATE' can be used for more general queries
    capable of providing more detailed in formation,
    for example, "what is the state of the lamp?"
    could produce state information that includes
    brightness or color.
    """
    STATE = auto()
    POWERED = auto()
    UNPOWERED = auto()
    LOCKED = auto()
    UNLOCKED = auto()
    OCCUPIED = auto()
    UNOCCUPIED = auto()

@unique
class Action(Enum):
    """
    This class represents 'Actions' that can be applied to
    'Things,' e.d. a LIGHT can be turned ON. It is intended
    to be used with the IoTRequest class. See that class
    for more details.
    """
    ON = auto()
    OFF = auto()
    TOGGLE = auto()
    ADJUST = auto()
    SET = auto()
    INCREASE = auto()
    DECREASE = auto()
    TRIGGER = auto()
    BINARY_QUERY = auto()
    INFORMATION_QUERY = auto()
    LOCATE = auto()
    LOCK = auto()
    UNLOCK = auto()

@total_ordering
class IoTRequestVersion(Enum):
    """
    Enum indicating support IoTRequest fields

    This class allows us to extend the request without
    requiring that all existing skills are updated to
    handle the new fields. Skills will simply not respond
    to requests that contain fields they are not aware of.

    CommonIoTSkill subclasses should override
    CommonIoTSkill.supported_request_version to indicate
    their level of support. For backward compatibility,
    the default is V1.

    Note that this is an attempt to avoid false positive
    matches (i.e. prevent skills from reporting that they
    can handle a request that contains fields they don't
    know anything about). To avoid any possibility of
    false negatives, however, skills should always try to
    support the latest version.

    Version to supported fields (provided only for reference - always use the
    latest version available, and account for all fields):

    V1 = {'action', 'thing', 'attribute', 'entity', 'scene'}
    V2 = V1 | {'value'}
    V3 = V2 | {'state'}
    """

    def __lt__(self, other):
        if False:
            return 10
        return self.name < other.name
    V1 = {'action', 'thing', 'attribute', 'entity', 'scene'}
    V2 = V1 | {'value'}
    V3 = V2 | {'state'}

class IoTRequest:
    """
    This class represents a request from a user to control
    an IoT device. It contains all of the information an IoT
    skill should need in order to determine if it can handle
    a user's request. The information is supplied as properties
    on the request. At present, those properties are:

    action (see the Action enum)
    thing (see the Thing enum)
    state (see the State enum)
    attribute (see the Attribute enum)
    value
    entity
    scene

    The 'action' is mandatory, and will always be not None. The
    other fields may be None.

    The 'entity' is intended to be used for user-defined values
    specific to a skill. For example, in a skill controlling Lights,
    an 'entity' might represent a group of lights. For a smart-lock
    skill, it might represent a specific lock, e.g. 'front door.'

    The 'scene' value is also intended to to be used for user-defined
    values. Skills that extend CommonIotSkill are expected to register
    their own scenes. The controller skill will have the ability to
    trigger multiple skills, so common scene names may trigger many
    skills, for a coherent experience.

    The 'value' property will be a number value. This is intended to
    be used for requests such as "set the heat to 70 degrees" and
    "set the lights to 50% brightness."

    Skills that extend CommonIotSkill will be expected to register
    their own entities. See the documentation in CommonIotSkill for
    more details.
    """

    def __init__(self, action: Action, thing: Thing=None, attribute: Attribute=None, entity: str=None, scene: str=None, value: int=None, state: State=None):
        if False:
            print('Hello World!')
        if not thing and (not entity) and (not scene):
            raise Exception('At least one of thing, entity, or scene must be present!')
        self.action = action
        self.thing = thing
        self.attribute = attribute
        self.entity = entity
        self.scene = scene
        self.value = value
        self.state = state

    def __repr__(self):
        if False:
            return 10
        template = 'IoTRequest(action={action}, thing={thing}, attribute={attribute}, entity={entity}, scene={scene}, value={value}, state={state})'
        entity = '"{}"'.format(self.entity) if self.entity else None
        scene = '"{}"'.format(self.scene) if self.scene else None
        value = '"{}"'.format(self.value) if self.value is not None else None
        return template.format(action=self.action, thing=self.thing, attribute=self.attribute, entity=entity, scene=scene, value=value, state=self.state)

    @property
    def version(self):
        if False:
            for i in range(10):
                print('nop')
        if self.state is not None:
            return IoTRequestVersion.V3
        if self.value is not None:
            return IoTRequestVersion.V2
        return IoTRequestVersion.V1

    def to_dict(self):
        if False:
            print('Hello World!')
        return {'action': self.action.name, 'thing': self.thing.name if self.thing else None, 'attribute': self.attribute.name if self.attribute else None, 'entity': self.entity, 'scene': self.scene, 'value': self.value, 'state': self.state.name if self.state else None}

    @classmethod
    def from_dict(cls, data: dict):
        if False:
            print('Hello World!')
        data = data.copy()
        data['action'] = Action[data['action']]
        if data.get('thing') not in (None, ''):
            data['thing'] = Thing[data['thing']]
        if data.get('attribute') not in (None, ''):
            data['attribute'] = Attribute[data['attribute']]
        if data.get('state') not in (None, ''):
            data['state'] = State[data['state']]
        return cls(**data)

def _track_request(func):
    if False:
        print('Hello World!')
    '\n    Used within the CommonIoT skill to track IoT requests.\n\n    The primary purpose of tracking the reqeust is determining\n    if the skill is currently handling an IoT request, or is\n    running a standard intent. While running IoT requests, certain\n    methods defined on MycroftSkill should behave differently than\n    under normal circumstances. In particular, speech related methods\n    should not actually trigger speech, but instead pass the message\n    to the IoT control skill, which will handle deconfliction (in the\n    event multiple skills want to respond verbally to the same request).\n\n    Args:\n        func: Callable\n\n    Returns:\n        Callable\n\n    '

    @wraps(func)
    def tracking_function(self, message: Message):
        if False:
            while True:
                i = 10
        with self._current_request(message.data.get(IOT_REQUEST_ID)):
            func(self, message)
    return tracking_function

class CommonIoTSkill(MycroftSkill, ABC):
    """
    Skills that want to work with the CommonIoT system should
    extend this class. Subclasses will be expected to implement
    two methods, `can_handle` and `run_request`. See the
    documentation for those functions for more details on how
    they are expected to behave.

    Subclasses may also register their own entities and scenes.
    See the register_entities and register_scenes methods for
    details.

    This class works in conjunction with a controller skill.
    The controller registers vocabulary and intents to capture
    IoT related requests. It then emits messages on the messagebus
    that will be picked up by all skills that extend this class.
    Each skill will have the opportunity to declare whether or not
    it can handle the given request. Skills that acknowledge that
    they are capable of handling the request will be considered
    candidates, and after a short timeout, a winner, or winners,
    will be chosen. With this setup, a user can have several IoT
    systems, and control them all without worry that skills will
    step on each other.
    """

    @wraps(MycroftSkill.__init__)
    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
        self._current_iot_request = None

    def bind(self, bus):
        if False:
            while True:
                i = 10
        '\n        Overrides MycroftSkill.bind.\n\n        This is called automatically during setup, and\n        need not otherwise be used.\n\n        Subclasses that override this method must call this\n        via super in their implementation.\n\n        Args:\n            bus:\n        '
        if bus:
            super().bind(bus)
            self.add_event(_BusKeys.TRIGGER, self._handle_trigger)
            self.add_event(_BusKeys.RUN + self.skill_id, self._run_request)
            self.add_event(_BusKeys.CALL_FOR_REGISTRATION, self._handle_call_for_registration)

    @contextmanager
    def _current_request(self, id: str):
        if False:
            i = 10
            return i + 15
        self._current_iot_request = id
        yield id
        self._current_iot_request = None

    @_track_request
    def _handle_trigger(self, message: Message):
        if False:
            print('Hello World!')
        '\n        Given a message, determines if this skill can\n        handle the request. If it can, it will emit\n        a message on the bus indicating that.\n\n        Args:\n            message: Message\n        '
        data = message.data
        request = IoTRequest.from_dict(data[IoTRequest.__name__])
        if request.version > self.supported_request_version:
            return
        (can_handle, callback_data) = self.can_handle(request)
        if can_handle:
            data.update({'skill_id': self.skill_id, 'callback_data': callback_data})
            self.bus.emit(message.response(data))

    @_track_request
    def _run_request(self, message: Message):
        if False:
            while True:
                i = 10
        '\n        Given a message, extracts the IoTRequest and\n        callback_data and sends them to the run_request\n        method.\n\n        Args:\n            message: Message\n        '
        request = IoTRequest.from_dict(message.data[IoTRequest.__name__])
        callback_data = message.data['callback_data']
        self.run_request(request, callback_data)

    def speak(self, utterance, *args, **kwargs):
        if False:
            while True:
                i = 10
        if self._current_iot_request:
            message = dig_for_message()
            self.bus.emit(message.forward(_BusKeys.SPEAK, data={'skill_id': self.skill_id, IOT_REQUEST_ID: self._current_iot_request, 'speak_args': args, 'speak_kwargs': kwargs, 'speak': utterance}))
        else:
            super().speak(utterance, *args, **kwargs)

    def _handle_call_for_registration(self, _: Message):
        if False:
            for i in range(10):
                print('nop')
        "\n        Register this skill's scenes and entities when requested.\n\n        Args:\n            _: Message. This is ignored.\n        "
        self.register_entities_and_scenes()

    def _register_words(self, words: [str], word_type: str):
        if False:
            i = 10
            return i + 15
        '\n        Emit a message to the controller skill to register vocab.\n\n        Emits a message on the bus containing the type and\n        the words. The message will be picked up by the\n        controller skill, and the vocabulary will be registered\n        to that skill.\n\n        Args:\n            words:\n            word_type:\n        '
        if words:
            self.bus.emit(Message(_BusKeys.REGISTER, data={'skill_id': self.skill_id, 'type': word_type, 'words': list(words)}))

    def register_entities_and_scenes(self):
        if False:
            return 10
        "\n        This method will register this skill's scenes and entities.\n\n        This should be called in the skill's `initialize` method,\n        at some point after `get_entities` and `get_scenes` can\n        be expected to return correct results.\n\n        "
        self._register_words(self.get_entities(), ENTITY)
        self._register_words(self.get_scenes(), SCENE)

    @property
    def supported_request_version(self) -> IoTRequestVersion:
        if False:
            while True:
                i = 10
        '\n        Get the supported IoTRequestVersion\n\n        By default, this returns IoTRequestVersion.V1. Subclasses\n        should override this to indicate higher levels of support.\n\n        The documentation for IoTRequestVersion provides a reference\n        indicating which fields are included in each version. Note\n        that you should always take the latest, and account for all\n        request fields.\n        '
        return IoTRequestVersion.V1

    def get_entities(self) -> [str]:
        if False:
            print('Hello World!')
        '\n        Get a list of custom entities.\n\n        This is intended to be overridden by subclasses, though it\n        it not required (the default implementation will return an\n        empty list).\n\n        The strings returned by this function will be registered\n        as ENTITY values with the intent parser. Skills should provide\n        group names, user aliases for specific devices, or anything\n        else that might represent a THING or a set of THINGs, e.g.\n        \'bedroom\', \'lamp\', \'front door.\' This allows commands that\n        don\'t explicitly include a THING to still be handled, e.g.\n        "bedroom off" as opposed to "bedroom lights off."\n        '
        return []

    def get_scenes(self) -> [str]:
        if False:
            i = 10
            return i + 15
        '\n        Get a list of custom scenes.\n\n        This method is intended to be overridden by subclasses, though\n        it is not required. The strings returned by this function will\n        be registered as SCENE values with the intent parser. Skills\n        should provide user defined scene names that they are aware of\n        and capable of handling, e.g. "relax," "movie time," etc.\n        '
        return []

    @abstractmethod
    def can_handle(self, request: IoTRequest):
        if False:
            i = 10
            return i + 15
        "\n        Determine if an IoTRequest can be handled by this skill.\n\n        This method must be implemented by all subclasses.\n\n        An IoTRequest contains several properties (see the\n        documentation for that class). This method should return\n        True if and only if this skill can take the appropriate\n        'action' when considering all other properties\n        of the request. In other words, a partial match, one in which\n        any piece of the IoTRequest is not known to this skill,\n        and is not None, this should return (False, None).\n\n        Args:\n            request: IoTRequest\n\n        Returns: (boolean, dict)\n            True if and only if this skill knows about all the\n            properties set on the IoTRequest, and a dict containing\n            callback_data. If this skill is chosen to handle the\n            request, this dict will be supplied to `run_request`.\n\n            Note that the dictionary will be sent over the bus, and thus\n            must be JSON serializable.\n        "
        return (False, None)

    @abstractmethod
    def run_request(self, request: IoTRequest, callback_data: dict):
        if False:
            return 10
        '\n        Handle an IoT Request.\n\n        All subclasses must implement this method.\n\n        When this skill is chosen as a winner, this function will be called.\n        It will be passed an IoTRequest equivalent to the one that was\n        supplied to `can_handle`, as well as the `callback_data` returned by\n        `can_handle`.\n\n        Args:\n            request: IoTRequest\n            callback_data: dict\n        '
        pass