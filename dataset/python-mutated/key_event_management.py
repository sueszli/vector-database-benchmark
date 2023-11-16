"""
Receives events from /dev/input/by-id/somedevice

Events are 24 bytes long and are emitted from a character device in 24 byte chunks.
As the keyboards have "2" keyboard interfaces we need to listen on both of them in case some n00b
switches to game mode.

Each event is in the format of
* signed int of seconds
* signed int of microseconds
* unsigned short of event type
* unsigned short code
* signed int value
"""
import datetime
import fcntl
import json
import logging
import os
import random
import select
import struct
import threading
import time
from openrazer_daemon.keyboard import KEY_MAPPING, TARTARUS_KEY_MAPPING, EVENT_MAPPING, TARTARUS_EVENT_MAPPING, NAGA_HEX_V2_EVENT_MAPPING, NAGA_HEX_V2_KEY_MAPPING, ORBWEAVER_EVENT_MAPPING, ORBWEAVER_KEY_MAPPING
from .macro import MacroKey, MacroRunner, macro_dict_to_obj
EVENT_FORMAT = '@llHHI'
EVENT_SIZE = struct.calcsize(EVENT_FORMAT)
EPOLL_TIMEOUT = 0.01
SPIN_SLEEP = 0.01
EVIOCGRAB = 1074021776
COLOUR_CHOICES = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255))

def random_colour_picker(last_choice, iterable):
    if False:
        while True:
            i = 10
    '\n    Chose a random choice but not the last one\n\n    :param last_choice: Last choice\n    :type last_choice: object\n\n    :param iterable: Iterable object\n    :type iterable: iterable\n\n    :return: Choice\n    :rtype: object\n    '
    result = random.choice(iterable)
    while result == last_choice:
        result = random.choice(iterable)
    return result

class KeyWatcher(threading.Thread):
    """
    Thread to watch keyboard event files and return keypresses
    """

    @staticmethod
    def parse_event_record(data):
        if False:
            print('Hello World!')
        '\n        Parse Input event record\n\n        :param data: Binary data\n        :type data: bytes\n\n        :return: Tuple of event time, key_action, key_code\n        :rtype: tuple\n        '
        (ev_sec, ev_usec, ev_type, ev_code, ev_value) = struct.unpack(EVENT_FORMAT, data)
        if ev_type != 1:
            return (None, None, None)
        if ev_value == 0:
            key_action = 'release'
        elif ev_value == 1:
            key_action = 'press'
        elif ev_value == 2:
            key_action = 'autorepeat'
        else:
            key_action = 'unknown'
        seconds = ev_sec + ev_usec * 1e-06
        date = datetime.datetime.fromtimestamp(seconds)
        result = (date, key_action, ev_code)
        if ev_type == ev_code == ev_value == 0:
            return (None, None, None)
        return result

    def __init__(self, device_id, event_files, parent, use_epoll=True):
        if False:
            print('Hello World!')
        super().__init__()
        self._logger = logging.getLogger('razer.device{0}.keywatcher'.format(device_id))
        self._event_files = event_files
        self._shutdown = False
        self._use_epoll = use_epoll
        self._parent = parent
        self.open_event_files = [open(event_file, 'rb') for event_file in self._event_files]
        for event_file in self.open_event_files:
            flags = fcntl.fcntl(event_file.fileno(), fcntl.F_GETFL)
            fcntl.fcntl(event_file.fileno(), fcntl.F_SETFL, flags | os.O_NONBLOCK)

    def run(self):
        if False:
            return 10
        '\n        Main event loop\n        '
        event_file_map = {event_file.fileno(): event_file for event_file in self.open_event_files}
        poll_object = select.epoll()
        for event_fd in event_file_map.keys():
            poll_object.register(event_fd, select.EPOLLIN | select.EPOLLPRI)
        while not self._shutdown:
            try:
                if self._use_epoll:
                    self._poll_epoll(poll_object, event_file_map)
                else:
                    self._poll_read()
            except (IOError, OSError):
                pass
            time.sleep(SPIN_SLEEP)
        for (event_fd, event_file) in event_file_map.items():
            poll_object.unregister(event_fd)
            event_file.close()
        poll_object.close()

    def _poll_epoll(self, poll_object, event_file_map):
        if False:
            print('Hello World!')
        events = poll_object.poll(EPOLL_TIMEOUT)
        if len(events) != 0:
            for (event_fd, mask) in events:
                while True:
                    key_data = event_file_map[event_fd].read(EVENT_SIZE)
                    if not key_data:
                        break
                    (date, key_action, key_code) = self.parse_event_record(key_data)
                    if date is None:
                        continue
                    self._parent.key_action(date, key_code, key_action)

    def _poll_read(self):
        if False:
            print('Hello World!')
        for event_file in self.open_event_files:
            key_data = event_file.read(EVENT_SIZE)
            if key_data is None:
                continue
            (date, key_action, key_code) = self.parse_event_record(key_data)
            if date is None:
                continue
            self._parent.key_action(date, key_code, key_action)

    @property
    def shutdown(self):
        if False:
            print('Hello World!')
        '\n        Thread shutdown condition\n\n        :return: Shutdown condition\n        :rtype: bool\n        '
        return self._shutdown

    @shutdown.setter
    def shutdown(self, value):
        if False:
            return 10
        '\n        Set thread shutdown condition\n\n        :param value: Boolean, normally only True would be used\n        :type value: str\n        '
        self._shutdown = value

class KeyboardKeyManager(object):
    """
    Key management class.

    This class deals with anything to do with keypresses. Currently it does:
    * Receiving keypresses from the KeyWatcher
    * Logic to deal with GameMode shortcut not working when macro's not enabled
    * Logic to deal with recording on the fly macros and replaying them

    It will be used to store keypresses in a list (for at most 2 seconds) if enabled for the ripple effect, when I
    get round to making the effect.
    """
    KEY_MAP = KEY_MAPPING
    EVENT_MAP = EVENT_MAPPING

    def __init__(self, device_id, event_files, parent, use_epoll=False, testing=False, should_grab_event_files=False):
        if False:
            i = 10
            return i + 15
        self._device_id = device_id
        self._logger = logging.getLogger('razer.device{0}.keymanager'.format(device_id))
        self._parent = parent
        self._parent.register_observer(self)
        self._testing = testing
        self._event_files = event_files
        self._access_lock = threading.Lock()
        self._keywatcher = KeyWatcher(device_id, event_files, self, use_epoll=use_epoll)
        self._open_event_files = self._keywatcher.open_event_files
        if len(event_files) > 0:
            self._logger.debug('Starting KeyWatcher')
            self._keywatcher.start()
        else:
            self._logger.warning('No event files for KeyWatcher')
        self._recording_macro = False
        self._macros = {}
        self._current_macro_bind_key = None
        self._current_macro_combo = []
        self._threads = set()
        self._clean_counter = 0
        self._temp_key_store_active = False
        self._temp_key_store = []
        self._temp_expire_time = datetime.timedelta(seconds=2)
        self._last_colour_choice = None
        self._should_grab_event_files = should_grab_event_files
        self._event_files_locked = False
        if self._should_grab_event_files:
            self.grab_event_files(True)

    @property
    def temp_key_store(self):
        if False:
            return 10
        '\n        Get the temporary key store\n\n        :return: List of keys\n        :rtype: list\n        '
        self._access_lock.acquire()
        now = datetime.datetime.now()
        try:
            while self._temp_key_store[0][0] < now:
                self._temp_key_store.pop(0)
        except IndexError:
            pass
        result = self._temp_key_store[:]
        self._access_lock.release()
        return result

    @property
    def temp_key_store_state(self):
        if False:
            i = 10
            return i + 15
        '\n        Get the state of the temporary key store\n\n        :return: Active state\n        :rtype: bool\n        '
        return self._temp_key_store_active

    @temp_key_store_state.setter
    def temp_key_store_state(self, value):
        if False:
            while True:
                i = 10
        '\n        Set the state of the temporary key store\n\n        :param value: Active state\n        :type value: bool\n        '
        self._temp_key_store_active = value

    def grab_event_files(self, grab):
        if False:
            while True:
                i = 10
        '\n        Grab the event files exclusively\n\n        :param grab: True to grab, False to release\n        :type grab: bool\n        '
        if not self._testing:
            for event_file in self._open_event_files:
                fcntl.ioctl(event_file.fileno(), EVIOCGRAB, int(grab))
        self._event_files_locked = grab

    def key_action(self, event_time, key_id, key_press='press'):
        if False:
            while True:
                i = 10
        '\n        Process a key press event\n\n        Ok an attempt to explain the logic\n        * The function sets a value _fn_down depending on the state of FN.\n        * Adds keypress and release events to a macro list if recording a macro.\n        * Pressing FN+F9 starts recording a macro, then selecting any key marks that as a macro key,\n          then it will record keys, then pressing FN+F9 will save macro.\n        * Pressing any macro key will run macro.\n        * Pressing FN+F10 will toggle game mode.\n        :param event_time: Time event occurred\n        :type event_time: datetime.datetime\n\n        :param key_id: Key Event ID\n        :type key_id: int\n\n        :param key_press: Can either be press, release, autorepeat\n        :type key_press: bool\n        '
        if not self._event_files_locked and self._should_grab_event_files:
            self.grab_event_files(True)
        if key_press == 'autorepeat':
            if key_id in (683, 682):
                key_press = 'press'
            else:
                return
        now = datetime.datetime.now()
        try:
            while self._temp_key_store[0][0] < now:
                self._temp_key_store.pop(0)
        except IndexError:
            pass
        if self._clean_counter > 20 and len(self._threads) > 0:
            self._clean_counter = 0
            self.clean_macro_threads()
        try:
            key_name = self.EVENT_MAP[key_id]
            if key_press == 'release':
                if self._recording_macro:
                    if key_name not in (self._current_macro_bind_key, 'MACROMODE'):
                        self._current_macro_combo.append((event_time, key_name, 'UP'))
            else:
                if self._temp_key_store_active:
                    colour = random_colour_picker(self._last_colour_choice, COLOUR_CHOICES)
                    self._last_colour_choice = colour
                    self._temp_key_store.append((now + self._temp_expire_time, self.KEY_MAP[key_name], colour))
                if key_name == 'MACROMODE':
                    self._logger.info('Got macro combo')
                    if not self._recording_macro:
                        self._recording_macro = True
                        self._current_macro_bind_key = None
                        self._current_macro_combo = []
                        self._parent.setMacroEffect(1)
                        self._parent.setMacroMode(True)
                    else:
                        self._logger.debug('Finished recording macro')
                        if self._current_macro_bind_key is not None:
                            if len(self._current_macro_combo) > 0:
                                self.add_kb_macro()
                            else:
                                self.dbus_delete_macro(self._current_macro_bind_key)
                        self._recording_macro = False
                        self._parent.setMacroEffect(0)
                        self._parent.setMacroMode(False)
                elif key_name == 'GAMEMODE':
                    self._logger.info('Got game mode combo')
                    game_mode = self._parent.getGameMode()
                    self._parent.setGameMode(not game_mode)
                elif key_name == 'BRIGHTNESSDOWN':
                    current_brightness = self._parent.method_args.get('brightness', None)
                    if current_brightness is None:
                        current_brightness = self._parent.getBrightness()
                    if current_brightness > 0:
                        current_brightness -= 10
                        if current_brightness < 0:
                            current_brightness = 0
                        self._parent.setBrightness(current_brightness)
                elif key_name == 'BRIGHTNESSUP':
                    current_brightness = self._parent.method_args.get('brightness', None)
                    if current_brightness is None:
                        current_brightness = self._parent.getBrightness()
                    if current_brightness < 100:
                        current_brightness += 10
                        if current_brightness > 100:
                            current_brightness = 100
                        self._parent.setBrightness(current_brightness)
                elif self._recording_macro:
                    if self._current_macro_bind_key is None:
                        if key_name not in ('M1', 'M2', 'M3', 'M4', 'M5'):
                            self._logger.warning('Macros are only for M1-M5 for now.')
                            self._recording_macro = False
                            self._parent.setMacroMode(False)
                        else:
                            self._current_macro_bind_key = key_name
                            self._parent.setMacroEffect(0)
                    elif key_name == self._current_macro_bind_key:
                        self._logger.warning('Skipping macro assignment as would cause recursion')
                        self._recording_macro = False
                        self._parent.setMacroMode(False)
                    else:
                        self._current_macro_combo.append((event_time, key_name, 'DOWN'))
                elif key_name in self._macros:
                    self.play_macro(key_name)
        except KeyError as err:
            self._logger.exception("Got key error. Couldn't convert event to key name", exc_info=err)

    def add_kb_macro(self):
        if False:
            print('Hello World!')
        '\n        Tidy up the recorded macro and add it to the store\n\n        Goes through the macro and generated relative delays between key events\n        '
        new_macro = []
        start_time = self._current_macro_combo[0][0]
        for (event_time, key, state) in self._current_macro_combo:
            delay = (event_time - start_time).microseconds
            start_time = event_time
            new_macro.append(MacroKey(key, delay, state))
        self._macros[self._current_macro_bind_key] = new_macro

    def clean_macro_threads(self):
        if False:
            i = 10
            return i + 15
        '\n        Threadless-threadpool\n\n        Goes though the threads (macro play jobs) and removed the threads if they have finished.\n        #SetMagic\n        '
        self._logger.debug('Cleaning up macro threads')
        to_remove = set()
        for macro_thread in self._threads:
            macro_thread.join(timeout=0.05)
            if not macro_thread.is_alive():
                to_remove.add(macro_thread)
        self._threads -= to_remove

    def play_macro(self, macro_key):
        if False:
            i = 10
            return i + 15
        '\n        Play macro for a given key\n\n        Launches a thread and adds it to the pool\n        :param macro_key: Macro Key\n        :type macro_key: str\n        '
        self._logger.info('Running Macro %s:%s', macro_key, str(self._macros[macro_key]))
        macro_thread = MacroRunner(self._device_id, macro_key, self._macros[macro_key])
        macro_thread.start()
        self._threads.add(macro_thread)

    def dbus_delete_macro(self, key_name):
        if False:
            return 10
        '\n        Delete a macro from a key\n\n        :param key_name: Key Name\n        :type key_name: str\n        '
        try:
            del self._macros[key_name]
        except KeyError:
            pass

    def dbus_get_macros(self):
        if False:
            while True:
                i = 10
        '\n        Get macros in JSON format\n\n        Returns a JSON blob of all active macros in the format of\n        {BIND_KEY: [MACRO_DICT...]}\n\n        MACRO_DICT is a dict representation of an action that can be performed. The dict will have a\n        type key which determines what type of action it will perform.\n        For example there are key press macros, URL opening macros, Script running macros etc...\n        :return: JSON of macros\n        :rtype: str\n        '
        result_dict = {}
        for (macro_key, macro_combo) in self._macros.items():
            str_combo = [value.to_dict() for value in macro_combo]
            result_dict[macro_key] = str_combo
        return json.dumps(result_dict)

    def dbus_add_macro(self, macro_key, macro_json):
        if False:
            return 10
        '\n        Add macro from JSON\n\n        The macro_json will be a list of macro objects which is then converted into JSON\n        :param macro_key: Macro bind key\n        :type macro_key: str\n\n        :param macro_json: Macro JSON\n        :type macro_json: str\n        '
        macro_list = [macro_dict_to_obj(macro_object_dict) for macro_object_dict in json.loads(macro_json)]
        self._macros[macro_key] = macro_list

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Cleanup function\n        '
        if self._keywatcher.is_alive():
            self._parent.remove_observer(self)
            self._logger.debug('Stopping key manager')
            self._keywatcher.shutdown = True
            self._keywatcher.join(timeout=2)
            if self._keywatcher.is_alive():
                self._logger.error('Could not stop KeyWatcher thread')

    def __del__(self):
        if False:
            i = 10
            return i + 15
        self.close()

    def notify(self, msg):
        if False:
            i = 10
            return i + 15
        '\n        Receive notificatons from the device (we only care about effects)\n\n        :param msg: Notification\n        :type msg: tuple\n        '
        if not isinstance(msg, tuple):
            self._logger.warning('Got msg that was not a tuple')
        elif msg[0] == 'effect':
            if msg[2] != 'setRipple':
                pass

class NagaHexV2KeyManager(KeyboardKeyManager):
    KEY_MAP = NAGA_HEX_V2_KEY_MAPPING
    EVENT_MAP = NAGA_HEX_V2_EVENT_MAPPING

class GamepadKeyManager(KeyboardKeyManager):
    GAMEPAD_EVENT_MAPPING = TARTARUS_EVENT_MAPPING
    GAMEPAD_KEY_MAPPING = TARTARUS_KEY_MAPPING

    def __init__(self, device_id, event_files, parent, use_epoll=True, testing=False):
        if False:
            while True:
                i = 10
        super().__init__(device_id, event_files, parent, use_epoll, testing=testing)
        self._mode_modifier = False
        self._mode_modifier_combo = []
        self._mode_modifier_key_down = False

    def key_action(self, event_time, key_id, key_press=True):
        if False:
            while True:
                i = 10
        '\n        Process a key press event\n\n        Ok an attempt to explain the logic\n        * The function sets a value _fn_down depending on the state of FN.\n        * Adds keypress and release events to a macro list if recording a macro.\n        * Pressing FN+F9 starts recording a macro, then selecting any key marks that as a macro key,\n          then it will record keys, then pressing FN+F9 will save macro.\n        * Pressing any macro key will run macro.\n        * Pressing FN+F10 will toggle game mode.\n        :param event_time: Time event occurred\n        :type event_time: datetime.datetime\n\n        :param key_id: Key Event ID\n        :type key_id: int\n\n        :param key_press: If true then its a press, else its a release\n        :type key_press: bool\n        '
        self._access_lock.acquire()
        if not self._event_files_locked:
            self.grab_event_files(True)
        now = datetime.datetime.now()
        try:
            while self._temp_key_store[0][0] < now:
                self._temp_key_store.pop(0)
        except IndexError:
            pass
        if self._clean_counter > 20 and len(self._threads) > 0:
            self._clean_counter = 0
            self.clean_macro_threads()
        try:
            key_name = self.GAMEPAD_EVENT_MAPPING[key_id]
            if self._temp_key_store_active:
                colour = random_colour_picker(self._last_colour_choice, COLOUR_CHOICES)
                self._last_colour_choice = colour
                self._temp_key_store.append((now + self._temp_expire_time, self.GAMEPAD_KEY_MAPPING[key_name], colour))
            if self._mode_modifier:
                if key_name == 'MODE_SWITCH' and key_press:
                    self._mode_modifier_key_down = True
                    self._mode_modifier_combo.clear()
                    self._mode_modifier_combo.append('MODE')
                elif key_name == 'MODE_SWITCH' and (not key_press):
                    self._mode_modifier_key_down = False
                elif key_press and self._mode_modifier_key_down:
                    self._mode_modifier_combo.append(key_name)
                    key_name = '+'.join(self._mode_modifier_combo)
            self._logger.debug('Macro String: {0}'.format(key_name))
            if key_name in self._macros and key_press:
                self.play_macro(key_name)
        except KeyError as err:
            self._logger.exception("Got key error. Couldn't convert event to key name", exc_info=err)
        self._access_lock.release()

    @property
    def mode_modifier(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get if the MODE_SWTICH key is to act as a modifier\n\n        :return: True if a modifier, false if not\n        :rtype: bool\n        '
        return self._mode_modifier

    @mode_modifier.setter
    def mode_modifier(self, value):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set MODE_SWITCH modifier state\n\n        :param value: Modifier state\n        :type value: bool\n        '
        self._mode_modifier = True if value else False

class OrbweaverKeyManager(GamepadKeyManager):
    GAMEPAD_EVENT_MAPPING = ORBWEAVER_EVENT_MAPPING
    GAMEPAD_KEY_MAPPING = ORBWEAVER_KEY_MAPPING