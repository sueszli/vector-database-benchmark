import logging
from threading import Event, Lock
from time import sleep, monotonic
from behave.contrib.scenario_autoretry import patch_scenario_with_autoretry
from msm import MycroftSkillsManager
from mycroft.audio import wait_while_speaking
from mycroft.configuration import Configuration
from mycroft.messagebus.client import MessageBusClient
from mycroft.messagebus import Message
from mycroft.util import create_daemon

def create_voight_kampff_logger():
    if False:
        for i in range(10):
            print('nop')
    fmt = logging.Formatter('{asctime} | {name} | {levelname} | {message}', style='{')
    handler = logging.StreamHandler()
    handler.setFormatter(fmt)
    log = logging.getLogger('Voight Kampff')
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False
    return log

class InterceptAllBusClient(MessageBusClient):
    """Bus Client storing all messages received.

    This allows read back of older messages and non-event-driven operation.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.messages = []
        self.message_lock = Lock()
        self.new_message_available = Event()
        self._processed_messages = 0

    def on_message(self, _, message):
        if False:
            for i in range(10):
                print('nop')
        'Extends normal operation by storing the received message.\n\n        Args:\n            message (Message): message from the Mycroft bus\n        '
        with self.message_lock:
            self.messages.append(Message.deserialize(message))
        self.new_message_available.set()
        super().on_message(_, message)

    def get_messages(self, msg_type):
        if False:
            for i in range(10):
                print('nop')
        'Get messages from received list of messages.\n\n        Args:\n            msg_type (None,str): string filter for the message type to extract.\n                                 if None all messages will be returned.\n        '
        with self.message_lock:
            self._processed_messages = len(self.messages)
            if msg_type is None:
                return [m for m in self.messages]
            else:
                return [m for m in self.messages if m.msg_type == msg_type]

    def remove_message(self, msg):
        if False:
            i = 10
            return i + 15
        'Remove a specific message from the list of messages.\n\n        Args:\n            msg (Message): message to remove from the list\n        '
        with self.message_lock:
            if msg not in self.messages:
                raise ValueError(f'{msg.msg_type} was not found in the list of messages.')
            if self.messages.index(msg) < self._processed_messages:
                self._processed_messages -= 1
            self.messages.remove(msg)

    def clear_messages(self):
        if False:
            for i in range(10):
                print('nop')
        'Clear all messages that has been fetched at least once.'
        with self.message_lock:
            self.messages = self.messages[self._processed_messages:]
            self._processed_messages = 0

    def clear_all_messages(self):
        if False:
            for i in range(10):
                print('nop')
        'Clear all messages.'
        with self.message_lock:
            self.messages = []
            self._processed_messages = 0

def before_all(context):
    if False:
        for i in range(10):
            print('nop')
    log = create_voight_kampff_logger()
    bus = InterceptAllBusClient()
    bus_connected = Event()
    bus.once('open', bus_connected.set)
    create_daemon(bus.run_forever)
    context.msm = MycroftSkillsManager()
    log.info('Waiting for messagebus connection...')
    bus_connected.wait()
    log.info('Waiting for skills to be loaded...')
    start = monotonic()
    while True:
        response = bus.wait_for_response(Message('mycroft.skills.all_loaded'))
        if response and response.data['status']:
            break
        elif monotonic() - start >= 2 * 60:
            raise Exception('Timeout waiting for skills to become ready.')
        else:
            sleep(1)
    context.bus = bus
    context.step_timeout = 10
    context.matched_message = None
    context.log = log
    context.config = Configuration.get()
    Configuration.set_config_update_handlers(bus)

def before_feature(context, feature):
    if False:
        print('Hello World!')
    context.log.info('Starting tests for {}'.format(feature.name))
    for scenario in feature.scenarios:
        patch_scenario_with_autoretry(scenario, max_attempts=2)

def after_all(context):
    if False:
        return 10
    context.bus.close()

def after_feature(context, feature):
    if False:
        return 10
    context.log.info('Result: {} ({:.2f}s)'.format(str(feature.status.name), feature.duration))

def after_scenario(context, scenario):
    if False:
        while True:
            i = 10
    'Wait for mycroft completion and reset any changed state.'
    wait_while_speaking()
    context.bus.clear_all_messages()
    context.matched_message = None
    context.step_timeout = 10