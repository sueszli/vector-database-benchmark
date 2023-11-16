from inspect import signature
from mycroft.messagebus import Message
from mycroft.metrics import Stopwatch, report_timing
from mycroft.util.log import LOG
from ..skill_data import to_alnum

def unmunge_message(message, skill_id):
    if False:
        for i in range(10):
            print('nop')
    'Restore message keywords by removing the Letterified skill ID.\n    Args:\n        message (Message): Intent result message\n        skill_id (str): skill identifier\n    Returns:\n        Message without clear keywords\n    '
    if isinstance(message, Message) and isinstance(message.data, dict):
        skill_id = to_alnum(skill_id)
        for key in list(message.data.keys()):
            if key.startswith(skill_id):
                new_key = key[len(skill_id):]
                message.data[new_key] = message.data.pop(key)
    return message

def get_handler_name(handler):
    if False:
        for i in range(10):
            print('nop')
    'Name (including class if available) of handler function.\n\n    Args:\n        handler (function): Function to be named\n\n    Returns:\n        string: handler name as string\n    '
    if '__self__' in dir(handler) and 'name' in dir(handler.__self__):
        return handler.__self__.name + '.' + handler.__name__
    else:
        return handler.__name__

def create_wrapper(handler, skill_id, on_start, on_end, on_error):
    if False:
        while True:
            i = 10
    'Create the default skill handler wrapper.\n\n    This wrapper handles things like metrics, reporting handler start/stop\n    and errors.\n        handler (callable): method/function to call\n        skill_id: skill_id for associated skill\n        on_start (function): function to call before executing the handler\n        on_end (function): function to call after executing the handler\n        on_error (function): function to call for error reporting\n    '

    def wrapper(message):
        if False:
            print('Hello World!')
        stopwatch = Stopwatch()
        try:
            message = Message(message.msg_type, data=message.data, context=message.context)
            message = unmunge_message(message, skill_id)
            if on_start:
                on_start(message)
            with stopwatch:
                if len(signature(handler).parameters) == 0:
                    handler()
                else:
                    handler(message)
        except Exception as e:
            if on_error:
                on_error(e)
        finally:
            if on_end:
                on_end(message)
            context = message.context
            if context and 'ident' in context:
                report_timing(context['ident'], 'skill_handler', stopwatch, {'handler': handler.__name__, 'skill_id': skill_id})
    return wrapper

def create_basic_wrapper(handler, on_error=None):
    if False:
        for i in range(10):
            print('nop')
    'Create the default skill handler wrapper.\n\n    This wrapper handles things like metrics, reporting handler start/stop\n    and errors.\n\n    Args:\n        handler (callable): method/function to call\n        on_error (function): function to call to report error.\n\n    Returns:\n        Wrapped callable\n    '

    def wrapper(message):
        if False:
            i = 10
            return i + 15
        try:
            if len(signature(handler).parameters) == 0:
                handler()
            else:
                handler(message)
        except Exception as e:
            if on_error:
                on_error(e)
    return wrapper

class EventContainer:
    """Container tracking messagbus handlers.

    This container tracks events added by a skill, allowing unregistering
    all events on shutdown.
    """

    def __init__(self, bus=None):
        if False:
            for i in range(10):
                print('nop')
        self.bus = bus
        self.events = []

    def set_bus(self, bus):
        if False:
            for i in range(10):
                print('nop')
        self.bus = bus

    def add(self, name, handler, once=False):
        if False:
            i = 10
            return i + 15
        'Create event handler for executing intent or other event.\n\n        Args:\n            name (string): IntentParser name\n            handler (func): Method to call\n            once (bool, optional): Event handler will be removed after it has\n                                   been run once.\n        '

        def once_wrapper(message):
            if False:
                return 10
            self.remove(name)
            handler(message)
        if handler:
            if once:
                self.bus.once(name, once_wrapper)
                self.events.append((name, once_wrapper))
            else:
                self.bus.on(name, handler)
                self.events.append((name, handler))
            LOG.debug('Added event: {}'.format(name))

    def remove(self, name):
        if False:
            print('Hello World!')
        'Removes an event from bus emitter and events list.\n\n        Args:\n            name (string): Name of Intent or Scheduler Event\n        Returns:\n            bool: True if found and removed, False if not found\n        '
        LOG.debug('Removing event {}'.format(name))
        removed = False
        for (_name, _handler) in list(self.events):
            if name == _name:
                try:
                    self.events.remove((_name, _handler))
                except ValueError:
                    LOG.error('Failed to remove event {}'.format(name))
                    pass
                removed = True
        if removed:
            self.bus.remove_all_listeners(name)
        return removed

    def __iter__(self):
        if False:
            print('Hello World!')
        return iter(self.events)

    def clear(self):
        if False:
            while True:
                i = 10
        'Unregister all registered handlers and clear the list of registered\n        events.\n        '
        for (e, f) in self.events:
            self.bus.remove(e, f)
        self.events = []