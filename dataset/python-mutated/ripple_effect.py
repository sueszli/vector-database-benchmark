"""
Contains the functions and classes to perform ripple effects
"""
import datetime
import logging
import math
import threading
import time
from openrazer_daemon.keyboard import KeyboardColour

class RippleEffectThread(threading.Thread):
    """
    Ripple thread.

    This thread contains the run loop which performs all the circle calculations and generating of the binary payload
    """

    def __init__(self, parent, device_number):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self._logger = logging.getLogger('razer.device{0}.ripplethread'.format(device_number))
        self._parent = parent
        self._colour = (0, 255, 0)
        self._refresh_rate = 0.04
        self._shutdown = False
        self._active = False
        (self._rows, self._cols) = self._parent._parent.MATRIX_DIMS
        self._keyboard_grid = KeyboardColour(self._rows, self._cols)

    @property
    def shutdown(self):
        if False:
            return 10
        '\n        Get the shutdown flag\n        '
        return self._shutdown

    @shutdown.setter
    def shutdown(self, value):
        if False:
            return 10
        '\n        Set the shutdown flag\n\n        :param value: Shutdown\n        :type value: bool\n        '
        self._shutdown = value

    @property
    def active(self):
        if False:
            return 10
        '\n        Get if the thread is active\n\n        :return: Active\n        :rtype: bool\n        '
        return self._active

    @property
    def key_list(self):
        if False:
            print('Hello World!')
        '\n        Get key list\n\n        :return: Key list\n        :rtype: list\n        '
        return self._parent.key_list

    def enable(self, colour, refresh_rate):
        if False:
            return 10
        '\n        Enable the ripple effect\n\n        If the colour tuple contains None then it will set the ripple to random colours\n        :param colour: Colour tuple like (0, 255, 255)\n        :type colour: tuple\n\n        :param refresh_rate: Refresh rate in seconds\n        :type refresh_rate: float\n        '
        if colour[0] is None:
            self._colour = None
        else:
            self._colour = colour
        self._refresh_rate = refresh_rate
        self._active = True

    def disable(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Disable the ripple effect\n        '
        self._active = False

    def run(self):
        if False:
            print('Hello World!')
        '\n        Event loop\n        '
        expire_diff = datetime.timedelta(seconds=2)
        if self._rows == 6 and self._cols == 22:
            needslogohandling = True
            self._rows += 1
        else:
            needslogohandling = False
        while not self._shutdown:
            if self._active:
                self._keyboard_grid.reset_rows()
                now = datetime.datetime.now()
                radiuses = []
                for (expire_time, (key_row, key_col), colour) in self.key_list:
                    event_time = expire_time - expire_diff
                    now_diff = now - event_time
                    if self._colour is not None:
                        colour = self._colour
                    radiuses.append((key_row, key_col, now_diff.total_seconds() * 24, colour))
                for row in range(0, self._rows):
                    for col in range(0, self._cols):
                        if needslogohandling and row == 0 and (col == 20):
                            continue
                        if needslogohandling and row == 6:
                            if col != 11:
                                continue
                            for (cirlce_centre_row, circle_centre_col, rad, colour) in radiuses:
                                radius = math.sqrt(math.pow(cirlce_centre_row - row, 2) + math.pow(circle_centre_col - col, 2))
                                if rad >= radius >= rad - 2:
                                    self._keyboard_grid.set_key_colour(0, 20, colour)
                                    break
                        else:
                            for (cirlce_centre_row, circle_centre_col, rad, colour) in radiuses:
                                radius = math.sqrt(math.pow(cirlce_centre_row - row, 2) + math.pow(circle_centre_col - col, 2))
                                if rad >= radius >= rad - 2:
                                    self._keyboard_grid.set_key_colour(row, col, colour)
                                    break
                payload = self._keyboard_grid.get_total_binary()
                self._parent.set_rgb_matrix(payload)
                self._parent.refresh_keyboard()
            time.sleep(self._refresh_rate)

class RippleManager(object):
    """
    Class which manages the overall process of performing a ripple effect
    """

    def __init__(self, parent, device_number):
        if False:
            for i in range(10):
                print('nop')
        self._logger = logging.getLogger('razer.device{0}.ripplemanager'.format(device_number))
        self._parent = parent
        self._parent.register_observer(self)
        self._is_closed = False
        self._ripple_thread = RippleEffectThread(self, device_number)
        self._ripple_thread.start()

    @property
    def key_list(self):
        if False:
            return 10
        '\n        Get the list of keys from the key manager\n\n        :return: List of tuples (expire_time, (key_row, key_col), random_colour)\n        :rtype: list of tuple\n        '
        result = []
        if hasattr(self._parent, 'key_manager'):
            result = self._parent.key_manager.temp_key_store
        return result

    def set_rgb_matrix(self, payload):
        if False:
            i = 10
            return i + 15
        '\n        Set the LED matrix on the keyboard\n\n        :param payload: Binary payload\n        :type payload: bytes\n        '
        self._parent._set_key_row(payload)

    def refresh_keyboard(self):
        if False:
            print('Hello World!')
        '\n        Refresh the keyboard\n        '
        self._parent._set_custom_effect()

    def notify(self, msg):
        if False:
            print('Hello World!')
        '\n        Receive notificatons from the device (we only care about effects)\n\n        :param msg: Notification\n        :type msg: tuple\n        '
        if not isinstance(msg, tuple):
            self._logger.warning('Got msg that was not a tuple')
        elif msg[0] == 'effect':
            if msg[2] == 'setRipple':
                self._parent.key_manager.temp_key_store_state = True
                self._ripple_thread.enable(msg[3:6], msg[6])
            else:
                self._ripple_thread.disable()
                self._parent.key_manager.temp_key_store_state = False

    def close(self):
        if False:
            return 10
        '\n        Close the manager, stop ripple thread\n        '
        if not self._is_closed:
            self._logger.debug('Closing Ripple Manager')
            self._is_closed = True
            self._ripple_thread.shutdown = True
            self._ripple_thread.join(timeout=2)
            if self._ripple_thread.is_alive():
                self._logger.error('Could not stop RippleEffect thread')

    def __del__(self):
        if False:
            while True:
                i = 10
        self.close()