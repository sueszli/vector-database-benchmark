"""Event scheduler system for calling skill (and other) methods at a specific
times.
"""
import json
import shutil
import time
from datetime import datetime, timedelta
from threading import Thread, Lock
from os.path import isfile, join, expanduser
import xdg.BaseDirectory
from mycroft.configuration import Configuration
from mycroft.messagebus.message import Message
from mycroft.util.log import LOG
from .mycroft_skill.event_container import EventContainer, create_basic_wrapper

def repeat_time(sched_time, repeat):
    if False:
        i = 10
        return i + 15
    'Next scheduled time for repeating event. Guarantees that the\n    time is not in the past (but could skip interim events)\n\n    Args:\n        sched_time (float): Scheduled unix time for the event\n        repeat (float):     Repeat period in seconds\n\n    Returns: (float) time for next event\n    '
    next_time = sched_time + repeat
    while next_time < time.time():
        next_time = time.time() + abs(repeat)
    return next_time

class EventScheduler(Thread):
    """Create an event scheduler thread. Will send messages at a
     predetermined time to the registered targets.

    Args:
        bus:            Mycroft messagebus (mycroft.messagebus)
        schedule_file:  File to store pending events to on shutdown
    """

    def __init__(self, bus, schedule_file='schedule.json'):
        if False:
            return 10
        super().__init__()
        self.events = {}
        self.event_lock = Lock()
        self.bus = bus
        self.is_running = True
        old_schedule_path = join(expanduser(Configuration.get()['data_dir']), schedule_file)
        new_schedule_path = join(xdg.BaseDirectory.load_first_config('mycroft'), schedule_file)
        if isfile(old_schedule_path):
            shutil.move(old_schedule_path, new_schedule_path)
        self.schedule_file = new_schedule_path
        if self.schedule_file:
            self.load()
        self.bus.on('mycroft.scheduler.schedule_event', self.schedule_event_handler)
        self.bus.on('mycroft.scheduler.remove_event', self.remove_event_handler)
        self.bus.on('mycroft.scheduler.update_event', self.update_event_handler)
        self.bus.on('mycroft.scheduler.get_event', self.get_event_handler)
        self.start()

    def load(self):
        if False:
            return 10
        'Load json data with active events from json file.'
        if isfile(self.schedule_file):
            json_data = {}
            with open(self.schedule_file) as f:
                try:
                    json_data = json.load(f)
                except Exception as e:
                    LOG.error(e)
            current_time = time.time()
            with self.event_lock:
                for key in json_data:
                    event_list = json_data[key]
                    self.events[key] = [tuple(e) for e in event_list if e[0] > current_time or e[1]]

    def run(self):
        if False:
            return 10
        while self.is_running:
            self.check_state()
            time.sleep(0.5)

    def check_state(self):
        if False:
            i = 10
            return i + 15
        'Check if an event should be triggered.'
        with self.event_lock:
            pending_messages = []
            for event in self.events:
                current_time = time.time()
                e = self.events[event]
                passed = [(t, r, d, c) for (t, r, d, c) in e if t <= current_time]
                remaining = [(t, r, d, c) for (t, r, d, c) in e if t > current_time]
                for (sched_time, repeat, data, context) in passed:
                    pending_messages.append(Message(event, data, context))
                    if repeat:
                        next_time = repeat_time(sched_time, repeat)
                        remaining.append((next_time, repeat, data, context))
                self.events[event] = remaining
        self.clear_empty()
        for msg in pending_messages:
            self.bus.emit(msg)

    def schedule_event(self, event, sched_time, repeat=None, data=None, context=None):
        if False:
            i = 10
            return i + 15
        'Add event to pending event schedule.\n\n        Args:\n            event (str): Handler for the event\n            sched_time ([type]): [description]\n            repeat ([type], optional): Defaults to None. [description]\n            data ([type], optional): Defaults to None. [description]\n            context (dict, optional): context (dict, optional): message\n                                      context to send when the\n                                      handler is called\n        '
        data = data or {}
        with self.event_lock:
            event_list = self.events.get(event, [])
            if repeat and event in self.events:
                LOG.debug('Repeating event {} is already scheduled, discarding'.format(event))
            else:
                event_list.append((sched_time, repeat, data, context))
                self.events[event] = event_list

    def schedule_event_handler(self, message):
        if False:
            while True:
                i = 10
        'Messagebus interface to the schedule_event method.\n        Required data in the message envelope is\n            event: event to emit\n            time:  time to emit the event\n\n        Optional data is\n            repeat: repeat interval\n            data:   data to send along with the event\n        '
        event = message.data.get('event')
        sched_time = message.data.get('time')
        repeat = message.data.get('repeat')
        data = message.data.get('data')
        context = message.context
        if event and sched_time:
            self.schedule_event(event, sched_time, repeat, data, context)
        elif not event:
            LOG.error('Scheduled event name not provided')
        else:
            LOG.error('Scheduled event time not provided')

    def remove_event(self, event):
        if False:
            while True:
                i = 10
        'Remove an event from the list of scheduled events.\n\n        Args:\n            event (str): event identifier\n        '
        with self.event_lock:
            if event in self.events:
                self.events.pop(event)

    def remove_event_handler(self, message):
        if False:
            for i in range(10):
                print('nop')
        'Messagebus interface to the remove_event method.'
        event = message.data.get('event')
        self.remove_event(event)

    def update_event(self, event, data):
        if False:
            for i in range(10):
                print('nop')
        'Change an existing events data.\n\n        This will only update the first call if multiple calls are registered\n        to the same event identifier.\n\n        Args:\n            event (str): event identifier\n            data (dict): new data\n        '
        with self.event_lock:
            if len(self.events.get(event, [])) > 0:
                (time, repeat, _, context) = self.events[event][0]
                self.events[event][0] = (time, repeat, data, context)

    def update_event_handler(self, message):
        if False:
            for i in range(10):
                print('nop')
        'Messagebus interface to the update_event method.'
        event = message.data.get('event')
        data = message.data.get('data')
        self.update_event(event, data)

    def get_event_handler(self, message):
        if False:
            print('Hello World!')
        'Messagebus interface to get_event.\n\n        Emits another event sending event status.\n        '
        event_name = message.data.get('name')
        event = None
        with self.event_lock:
            if event_name in self.events:
                event = self.events[event_name]
        emitter_name = 'mycroft.event_status.callback.{}'.format(event_name)
        self.bus.emit(message.reply(emitter_name, data=event))

    def store(self):
        if False:
            while True:
                i = 10
        'Write current schedule to disk.'
        with self.event_lock:
            with open(self.schedule_file, 'w') as f:
                json.dump(self.events, f)

    def clear_repeating(self):
        if False:
            for i in range(10):
                print('nop')
        'Remove repeating events from events dict.'
        with self.event_lock:
            for e in self.events:
                self.events[e] = [i for i in self.events[e] if i[1] is None]

    def clear_empty(self):
        if False:
            while True:
                i = 10
        'Remove empty event entries from events dict.'
        with self.event_lock:
            self.events = {k: self.events[k] for k in self.events if self.events[k] != []}

    def shutdown(self):
        if False:
            i = 10
            return i + 15
        'Stop the running thread.'
        self.is_running = False
        self.bus.remove_all_listeners('mycroft.scheduler.schedule_event')
        self.bus.remove_all_listeners('mycroft.scheduler.remove_event')
        self.bus.remove_all_listeners('mycroft.scheduler.update_event')
        self.join()
        self.clear_repeating()
        self.clear_empty()
        self.store()

class EventSchedulerInterface:
    """Interface for accessing the event scheduler over the message bus."""

    def __init__(self, name, sched_id=None, bus=None):
        if False:
            i = 10
            return i + 15
        self.name = name
        self.sched_id = sched_id
        self.bus = bus
        self.events = EventContainer(bus)
        self.scheduled_repeats = []

    def set_bus(self, bus):
        if False:
            while True:
                i = 10
        self.bus = bus
        self.events.set_bus(bus)

    def set_id(self, sched_id):
        if False:
            return 10
        self.sched_id = sched_id

    def _create_unique_name(self, name):
        if False:
            print('Hello World!')
        'Return a name unique to this skill using the format\n        [skill_id]:[name].\n\n        Args:\n            name:   Name to use internally\n\n        Returns:\n            str: name unique to this skill\n        '
        return str(self.sched_id) + ':' + (name or '')

    def _schedule_event(self, handler, when, data, name, repeat_interval=None, context=None):
        if False:
            return 10
        'Underlying method for schedule_event and schedule_repeating_event.\n\n        Takes scheduling information and sends it off on the message bus.\n\n        Args:\n            handler:                method to be called\n            when (datetime):        time (in system timezone) for first\n                                    calling the handler, or None to\n                                    initially trigger <frequency> seconds\n                                    from now\n            data (dict, optional):  data to send when the handler is called\n            name (str, optional):   reference name, must be unique\n            repeat_interval (float/int):  time in seconds between calls\n            context (dict, optional): message context to send\n                                      when the handler is called\n        '
        if isinstance(when, (int, float)) and when >= 0:
            when = datetime.now() + timedelta(seconds=when)
        if not name:
            name = self.name + handler.__name__
        unique_name = self._create_unique_name(name)
        if repeat_interval:
            self.scheduled_repeats.append(name)
        data = data or {}

        def on_error(e):
            if False:
                while True:
                    i = 10
            LOG.exception('An error occured executing the scheduled event {}'.format(repr(e)))
        wrapped = create_basic_wrapper(handler, on_error)
        self.events.add(unique_name, wrapped, once=not repeat_interval)
        event_data = {'time': time.mktime(when.timetuple()), 'event': unique_name, 'repeat': repeat_interval, 'data': data}
        self.bus.emit(Message('mycroft.scheduler.schedule_event', data=event_data, context=context))

    def schedule_event(self, handler, when, data=None, name=None, context=None):
        if False:
            return 10
        'Schedule a single-shot event.\n\n        Args:\n            handler:               method to be called\n            when (datetime/int/float):   datetime (in system timezone) or\n                                   number of seconds in the future when the\n                                   handler should be called\n            data (dict, optional): data to send when the handler is called\n            name (str, optional):  reference name\n                                   NOTE: This will not warn or replace a\n                                   previously scheduled event of the same\n                                   name.\n            context (dict, optional): message context to send\n                                      when the handler is called\n        '
        self._schedule_event(handler, when, data, name, context=context)

    def schedule_repeating_event(self, handler, when, interval, data=None, name=None, context=None):
        if False:
            print('Hello World!')
        'Schedule a repeating event.\n\n        Args:\n            handler:                method to be called\n            when (datetime):        time (in system timezone) for first\n                                    calling the handler, or None to\n                                    initially trigger <frequency> seconds\n                                    from now\n            interval (float/int):   time in seconds between calls\n            data (dict, optional):  data to send when the handler is called\n            name (str, optional):   reference name, must be unique\n            context (dict, optional): message context to send\n                                      when the handler is called\n        '
        if name not in self.scheduled_repeats:
            if not when:
                when = datetime.now() + timedelta(seconds=interval)
            self._schedule_event(handler, when, data, name, interval, context=context)
        else:
            LOG.debug('The event is already scheduled, cancel previous event if this scheduling should replace the last.')

    def update_scheduled_event(self, name, data=None):
        if False:
            return 10
        'Change data of event.\n\n        Args:\n            name (str): reference name of event (from original scheduling)\n        '
        data = data or {}
        data = {'event': self._create_unique_name(name), 'data': data}
        self.bus.emit(Message('mycroft.schedule.update_event', data=data))

    def cancel_scheduled_event(self, name):
        if False:
            print('Hello World!')
        'Cancel a pending event. The event will no longer be scheduled\n        to be executed\n\n        Args:\n            name (str): reference name of event (from original scheduling)\n        '
        unique_name = self._create_unique_name(name)
        data = {'event': unique_name}
        if name in self.scheduled_repeats:
            self.scheduled_repeats.remove(name)
        if self.events.remove(unique_name):
            self.bus.emit(Message('mycroft.scheduler.remove_event', data=data))

    def get_scheduled_event_status(self, name):
        if False:
            while True:
                i = 10
        'Get scheduled event data and return the amount of time left\n\n        Args:\n            name (str): reference name of event (from original scheduling)\n\n        Returns:\n            int: the time left in seconds\n\n        Raises:\n            Exception: Raised if event is not found\n        '
        event_name = self._create_unique_name(name)
        data = {'name': event_name}
        reply_name = 'mycroft.event_status.callback.{}'.format(event_name)
        msg = Message('mycroft.scheduler.get_event', data=data)
        status = self.bus.wait_for_response(msg, reply_type=reply_name)
        if status:
            event_time = int(status.data[0][0])
            current_time = int(time.time())
            time_left_in_seconds = event_time - current_time
            LOG.info(time_left_in_seconds)
            return time_left_in_seconds
        else:
            raise Exception('Event Status Messagebus Timeout')

    def cancel_all_repeating_events(self):
        if False:
            for i in range(10):
                print('nop')
        'Cancel any repeating events started by the skill.'
        for e in list(self.scheduled_repeats):
            self.cancel_scheduled_event(e)

    def shutdown(self):
        if False:
            for i in range(10):
                print('nop')
        'Shutdown the interface unregistering any event handlers.'
        self.cancel_all_repeating_events()
        self.events.clear()