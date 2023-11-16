from collections import namedtuple
from enum import IntEnum
import json
import logging
import signal as sig
import sys
from threading import Event, Thread
from time import sleep
from .log import LOG

def reset_sigint_handler():
    if False:
        return 10
    'Reset the sigint handler to the default.\n\n    This fixes KeyboardInterrupt not getting raised when started via\n    start-mycroft.sh\n    '
    sig.signal(sig.SIGINT, sig.default_int_handler)

def create_daemon(target, args=(), kwargs=None):
    if False:
        for i in range(10):
            print('nop')
    'Helper to quickly create and start a thread with daemon = True'
    t = Thread(target=target, args=args, kwargs=kwargs)
    t.daemon = True
    t.start()
    return t

def wait_for_exit_signal():
    if False:
        for i in range(10):
            print('nop')
    'Blocks until KeyboardInterrupt is received.'
    try:
        while True:
            sleep(100)
    except KeyboardInterrupt:
        pass
_log_all_bus_messages = False

def bus_logging_status():
    if False:
        print('Hello World!')
    global _log_all_bus_messages
    return _log_all_bus_messages

def _update_log_level(msg, name):
    if False:
        return 10
    'Update log level for process.\n\n    Args:\n        msg (Message): Message sent to trigger the log level change\n        name (str): Name of the current process\n    '
    global _log_all_bus_messages
    lvl = msg['data'].get('level', '').upper()
    if lvl in ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']:
        LOG.level = lvl
        LOG(name).info('Changing log level to: {}'.format(lvl))
        try:
            logging.getLogger().setLevel(lvl)
            logging.getLogger('urllib3').setLevel(lvl)
        except Exception:
            pass
    else:
        LOG(name).info('Invalid level provided: {}'.format(lvl))
    log_bus = msg['data'].get('bus', None)
    if log_bus is not None:
        LOG(name).info('Bus logging: {}'.format(log_bus))
        _log_all_bus_messages = log_bus

def create_echo_function(name, whitelist=None):
    if False:
        i = 10
        return i + 15
    'Standard logging mechanism for Mycroft processes.\n\n    This handles the setup of the basic logging for all Mycroft\n    messagebus-based processes.\n    TODO 20.08: extract log level setting thing completely from this function\n\n    Args:\n        name (str): Reference name of the process\n        whitelist (list, optional): List of "type" strings. If defined, only\n                                    messages in this list will be logged.\n\n    Returns:\n        func: The echo function\n    '
    from mycroft.configuration import Configuration
    blacklist = Configuration.get().get('ignore_logs')
    if whitelist:
        whitelist.append('mycroft.debug.log')

    def echo(message):
        if False:
            return 10
        global _log_all_bus_messages
        try:
            msg = json.loads(message)
            msg_type = msg.get('type', '')
            if whitelist and (not any([msg_type.startswith(e) for e in whitelist])):
                return
            if blacklist and msg_type in blacklist:
                return
            if msg_type == 'mycroft.debug.log':
                _update_log_level(msg, name)
            elif msg_type == 'registration':
                msg['data']['token'] = None
                message = json.dumps(msg)
        except Exception as e:
            LOG.info('Error: {}'.format(repr(e)), exc_info=True)
        if _log_all_bus_messages:
            LOG(name).info('BUS: {}'.format(message))
    return echo

def start_message_bus_client(service, bus=None, whitelist=None):
    if False:
        while True:
            i = 10
    'Start the bus client daemon and wait for connection.\n\n    Args:\n        service (str): name of the service starting the connection\n        bus (MessageBusClient): an instance of the Mycroft MessageBusClient\n        whitelist (list, optional): List of "type" strings. If defined, only\n                                    messages in this list will be logged.\n    Returns:\n        A connected instance of the MessageBusClient\n    '
    from mycroft.messagebus.client import MessageBusClient
    from mycroft.configuration import Configuration
    if bus is None:
        bus = MessageBusClient()
    Configuration.set_config_update_handlers(bus)
    bus_connected = Event()
    bus.on('message', create_echo_function(service, whitelist))
    bus.once('open', bus_connected.set)
    create_daemon(bus.run_forever)
    bus_connected.wait()
    LOG.info('Connected to messagebus')
    return bus

class ProcessState(IntEnum):
    """Oredered enum to make state checks easy.

    For example Alive can be determined using >= ProcessState.ALIVE,
    which will return True if the state is READY as well as ALIVE.
    """
    NOT_STARTED = 0
    STARTED = 1
    ERROR = 2
    STOPPING = 3
    ALIVE = 4
    READY = 5
_STATUS_CALLBACKS = ['on_started', 'on_alive', 'on_ready', 'on_error', 'on_stopping']
if sys.version_info < (3, 7):
    StatusCallbackMap = namedtuple('CallbackMap', _STATUS_CALLBACKS)
    StatusCallbackMap.__new__.__defaults__ = (None,) * 5
else:
    StatusCallbackMap = namedtuple('CallbackMap', _STATUS_CALLBACKS, defaults=(None,) * len(_STATUS_CALLBACKS))

class ProcessStatus:
    """Process status tracker.

    The class tracks process status and execute callback methods on
    state changes as well as replies to messagebus queries of the
    process status.

    Args:
        name (str): process name, will be used to create the messagebus
                    messagetype "mycroft.{name}...".
        bus (MessageBusClient): Connection to the Mycroft messagebus.
        callback_map (StatusCallbackMap): optionally, status callbacks for the
                                          various status changes.
    """

    def __init__(self, name, bus, callback_map=None):
        if False:
            return 10
        self.bus = bus
        self.name = name
        self.callbacks = callback_map or StatusCallbackMap()
        self.state = ProcessState.NOT_STARTED
        self._register_handlers()

    def _register_handlers(self):
        if False:
            print('Hello World!')
        'Register messagebus handlers for status queries.'
        self.bus.on('mycroft.{}.is_alive'.format(self.name), self.check_alive)
        self.bus.on('mycroft.{}.is_ready'.format(self.name), self.check_ready)
        self.bus.on('mycroft.{}.all_loaded'.format(self.name), self.check_ready)

    def check_alive(self, message=None):
        if False:
            for i in range(10):
                print('nop')
        'Respond to is_alive status request.\n\n        Args:\n            message: Optional message to respond to, if omitted no message\n                     is sent.\n\n        Returns:\n            bool, True if process is alive.\n        '
        is_alive = self.state >= ProcessState.ALIVE
        if message:
            status = {'status': is_alive}
            self.bus.emit(message.response(data=status))
        return is_alive

    def check_ready(self, message=None):
        if False:
            i = 10
            return i + 15
        'Respond to all_loaded status request.\n\n        Args:\n            message: Optional message to respond to, if omitted no message\n                     is sent.\n\n        Returns:\n            bool, True if process is ready.\n        '
        is_ready = self.state >= ProcessState.READY
        if message:
            status = {'status': is_ready}
            self.bus.emit(message.response(data=status))
        return is_ready

    def set_started(self):
        if False:
            i = 10
            return i + 15
        'Process is started.'
        self.state = ProcessState.STARTED
        if self.callbacks.on_started:
            self.callbacks.on_started()

    def set_alive(self):
        if False:
            for i in range(10):
                print('nop')
        'Basic loading is done.'
        self.state = ProcessState.ALIVE
        if self.callbacks.on_alive:
            self.callbacks.on_alive()

    def set_ready(self):
        if False:
            i = 10
            return i + 15
        'All loading is done.'
        self.state = ProcessState.READY
        if self.callbacks.on_ready:
            self.callbacks.on_ready()

    def set_stopping(self):
        if False:
            print('Hello World!')
        'Process shutdown has started.'
        self.state = ProcessState.STOPPING
        if self.callbacks.on_stopping:
            self.callbacks.on_stopping()

    def set_error(self, err=''):
        if False:
            while True:
                i = 10
        'An error has occured and the process is non-functional.'
        self.state = ProcessState.ERROR
        if self.callbacks.on_error:
            self.callbacks.on_error(err)