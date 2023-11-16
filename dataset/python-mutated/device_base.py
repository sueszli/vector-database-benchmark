"""
Hardware base class
"""
import configparser
import re
import os
import types
import inspect
import logging
import time
import json
import random
from openrazer_daemon.dbus_services.service import DBusService
import openrazer_daemon.dbus_services.dbus_methods
from openrazer_daemon.misc import effect_sync
from openrazer_daemon.misc.battery_notifier import BatteryManager as _BatteryManager

class RazerDevice(DBusService):
    """
    Base class

    Sets up the logger, sets up DBus
    """
    OBJECT_PATH = '/org/razer/device/'
    METHODS = []
    EVENT_FILE_REGEX = None
    USB_VID = None
    USB_PID = None
    HAS_MATRIX = False
    DEDICATED_MACRO_KEYS = False
    MATRIX_DIMS = None
    POLL_RATES = None
    DPI_MAX = None
    WAVE_DIRS = (1, 2)
    ZONES = ('backlight', 'logo', 'scroll', 'left', 'right', 'charging', 'fast_charging', 'fully_charged', 'channel1', 'channel2', 'channel3', 'channel4', 'channel5', 'channel6')
    DEVICE_IMAGE = None

    def __init__(self, device_path, device_number, config, persistence, testing, additional_interfaces, additional_methods):
        if False:
            return 10
        self.logger = logging.getLogger('razer.device{0}'.format(device_number))
        self.logger.info('Initialising device.%d %s', device_number, self.__class__.__name__)
        self._serial = None
        self.storage_name = 'UnknownDevice'
        self._observer_list = []
        self._effect_sync_propagate_up = False
        self._disable_notifications = False
        self._disable_persistence = False
        self.additional_interfaces = []
        if additional_interfaces is not None:
            self.additional_interfaces.extend(additional_interfaces)
        self._battery_manager = None
        self.config = config
        self.persistence = persistence
        self._testing = testing
        self._parent = None
        self._device_path = device_path
        self._device_number = device_number
        self.serial = self.get_serial()
        if self.USB_PID == 3847:
            self.storage_name = 'ChromaMug'
        elif self.USB_PID == 19:
            self.storage_name = 'Orochi2011'
        elif self.USB_PID == 22:
            self.storage_name = 'DeathAdder35G'
        elif self.USB_PID == 41:
            self.storage_name = 'DeathAdder35GBlack'
        elif self.USB_PID == 36 or self.USB_PID == 37:
            self.storage_name = 'Mamba2012'
        else:
            self.storage_name = self.serial
        self.zone = dict()
        for i in self.ZONES:
            self.zone[i] = {'present': False, 'active': True, 'brightness': 75.0, 'effect': 'spectrum', 'colors': [0, 255, 0, 0, 255, 255, 0, 0, 255], 'speed': 1, 'wave_dir': 1}
        if 'available_dpi' in self.METHODS:
            self.dpi = [1800, 0]
        else:
            self.dpi = [1800, 1800]
        self.poll_rate = 500
        if 'set_poll_rate' in self.METHODS and (not self.POLL_RATES):
            self.POLL_RATES = [125, 500, 1000]
        self._effect_sync = effect_sync.EffectSync(self, device_number)
        self._is_closed = False
        self.methods_internal = ['get_firmware', 'get_matrix_dims', 'has_matrix', 'get_device_name']
        self.methods_internal.extend(additional_methods)
        self.event_files = []
        if self._testing:
            search_dir = os.path.join(device_path, 'input')
        else:
            search_dir = '/dev/input/by-id/'
        if os.path.exists(search_dir):
            for event_file in os.listdir(search_dir):
                if self.EVENT_FILE_REGEX is not None and self.EVENT_FILE_REGEX.match(event_file) is not None:
                    self.event_files.append(os.path.join(search_dir, event_file))
        object_path = os.path.join(self.OBJECT_PATH, self.serial)
        super().__init__(object_path)
        self.suspend_args = {}
        self.method_args = {}
        methods = {('razer.device.misc', 'getSerial', self.get_serial, None, 's'), ('razer.device.misc', 'suspendDevice', self.suspend_device, None, None), ('razer.device.misc', 'getDeviceMode', self.get_device_mode, None, 's'), ('razer.device.misc', 'getDeviceImage', self.get_device_image, None, 's'), ('razer.device.misc', 'setDeviceMode', self.set_device_mode, 'yy', None), ('razer.device.misc', 'resumeDevice', self.resume_device, None, None), ('razer.device.misc', 'getVidPid', self.get_vid_pid, None, 'ai'), ('razer.device.misc', 'getDriverVersion', openrazer_daemon.dbus_services.dbus_methods.version, None, 's'), ('razer.device.misc', 'hasDedicatedMacroKeys', self.dedicated_macro_keys, None, 'b'), ('razer.device.misc', 'getRazerUrls', self.get_image_json, None, 's'), ('razer.device.lighting.chroma', 'restoreLastEffect', self.restore_effect, None, None)}
        effect_methods = {'backlight_chroma': {('razer.device.lighting.chroma', 'getEffect', self.get_current_effect, None, 's'), ('razer.device.lighting.chroma', 'getEffectColors', self.get_current_effect_colors, None, 'ay'), ('razer.device.lighting.chroma', 'getEffectSpeed', self.get_current_effect_speed, None, 'i'), ('razer.device.lighting.chroma', 'getWaveDir', self.get_current_wave_dir, None, 'i')}, 'backlight': {('razer.device.lighting.backlight', 'getBacklightEffect', self.get_current_effect, None, 's'), ('razer.device.lighting.backlight', 'getBacklightEffectColors', self.get_current_effect_colors, None, 'ay'), ('razer.device.lighting.backlight', 'getBacklightEffectSpeed', self.get_current_effect_speed, None, 'i'), ('razer.device.lighting.backlight', 'getBacklightWaveDir', self.get_current_wave_dir, None, 'i')}, 'logo': {('razer.device.lighting.logo', 'getLogoEffect', self.get_current_logo_effect, None, 's'), ('razer.device.lighting.logo', 'getLogoEffectColors', self.get_current_logo_effect_colors, None, 'ay'), ('razer.device.lighting.logo', 'getLogoEffectSpeed', self.get_current_logo_effect_speed, None, 'i'), ('razer.device.lighting.logo', 'getLogoWaveDir', self.get_current_logo_wave_dir, None, 'i')}, 'scroll': {('razer.device.lighting.scroll', 'getScrollEffect', self.get_current_scroll_effect, None, 's'), ('razer.device.lighting.scroll', 'getScrollEffectColors', self.get_current_scroll_effect_colors, None, 'ay'), ('razer.device.lighting.scroll', 'getScrollEffectSpeed', self.get_current_scroll_effect_speed, None, 'i'), ('razer.device.lighting.scroll', 'getScrollWaveDir', self.get_current_scroll_wave_dir, None, 'i')}, 'left': {('razer.device.lighting.left', 'getLeftEffect', self.get_current_left_effect, None, 's'), ('razer.device.lighting.left', 'getLeftEffectColors', self.get_current_left_effect_colors, None, 'ay'), ('razer.device.lighting.left', 'getLeftEffectSpeed', self.get_current_left_effect_speed, None, 'i'), ('razer.device.lighting.left', 'getLeftWaveDir', self.get_current_left_wave_dir, None, 'i')}, 'right': {('razer.device.lighting.right', 'getRightEffect', self.get_current_right_effect, None, 's'), ('razer.device.lighting.right', 'getRightEffectColors', self.get_current_right_effect_colors, None, 'ay'), ('razer.device.lighting.right', 'getRightEffectSpeed', self.get_current_right_effect_speed, None, 'i'), ('razer.device.lighting.right', 'getRightWaveDir', self.get_current_right_wave_dir, None, 'i')}, 'charging': {('razer.device.lighting.charging', 'getChargingEffect', self.get_current_charging_effect, None, 's'), ('razer.device.lighting.charging', 'getChargingEffectColors', self.get_current_charging_effect_colors, None, 'ay'), ('razer.device.lighting.charging', 'getChargingEffectSpeed', self.get_current_charging_effect_speed, None, 'i'), ('razer.device.lighting.charging', 'getChargingWaveDir', self.get_current_charging_wave_dir, None, 'i')}, 'fast_charging': {('razer.device.lighting.fast_charging', 'getFastChargingEffect', self.get_current_fast_charging_effect, None, 's'), ('razer.device.lighting.fast_charging', 'getFastChargingEffectColors', self.get_current_fast_charging_effect_colors, None, 'ay'), ('razer.device.lighting.fast_charging', 'getFastChargingEffectSpeed', self.get_current_fast_charging_effect_speed, None, 'i'), ('razer.device.lighting.fast_charging', 'getFastChargingWaveDir', self.get_current_fast_charging_wave_dir, None, 'i')}, 'fully_charged': {('razer.device.lighting.fully_charged', 'getFullyChargedEffect', self.get_current_fully_charged_effect, None, 's'), ('razer.device.lighting.fully_charged', 'getFullyChargedEffectColors', self.get_current_fully_charged_effect_colors, None, 'ay'), ('razer.device.lighting.fully_charged', 'getFullyChargedEffectSpeed', self.get_current_fully_charged_effect_speed, None, 'i'), ('razer.device.lighting.fully_charged', 'getFullyChargedWaveDir', self.get_current_fully_charged_wave_dir, None, 'i')}}
        for m in methods:
            self.logger.debug('Adding {}.{} method to DBus'.format(m[0], m[1]))
            self.add_dbus_method(m[0], m[1], m[2], in_signature=m[3], out_signature=m[4])
        if 'set_static_effect' in self.METHODS or 'bw_set_static' in self.METHODS:
            self.zone['backlight']['present'] = True
            for m in effect_methods['backlight_chroma']:
                self.logger.debug('Adding {}.{} method to DBus'.format(m[0], m[1]))
                self.add_dbus_method(m[0], m[1], m[2], in_signature=m[3], out_signature=m[4])
        for i in self.ZONES:
            if 'set_' + i + '_static_classic' in self.METHODS or 'set_' + i + '_static' in self.METHODS or 'set_' + i + '_active' in self.METHODS or ('set_' + i + '_on' in self.METHODS):
                self.zone[i]['present'] = True
                for m in effect_methods[i]:
                    self.logger.debug('Adding {}.{} method to DBus'.format(m[0], m[1]))
                    self.add_dbus_method(m[0], m[1], m[2], in_signature=m[3], out_signature=m[4])
        self.load_methods()
        if self.persistence.has_section(self.storage_name):
            if 'set_dpi_xy' in self.METHODS or 'set_dpi_xy_byte' in self.METHODS:
                try:
                    self.dpi[0] = int(self.persistence[self.storage_name]['dpi_x'])
                    self.dpi[1] = int(self.persistence[self.storage_name]['dpi_y'])
                except (KeyError, configparser.NoOptionError):
                    self.logger.info('Failed to get DPI from persistence storage, using default.')
            if 'set_poll_rate' in self.METHODS:
                try:
                    self.poll_rate = int(self.persistence[self.storage_name]['poll_rate'])
                except (KeyError, configparser.NoOptionError):
                    self.logger.info('Failed to get poll rate from persistence storage, using default.')
        for i in self.ZONES:
            if self.zone[i]['present']:
                if self.persistence.has_section(self.storage_name):
                    try:
                        self.zone[i]['effect'] = self.persistence[self.storage_name][i + '_effect']
                    except (KeyError, configparser.NoOptionError):
                        self.logger.info('Failed to get ' + i + ' effect from persistence storage, using default.')
                    try:
                        self.zone[i]['active'] = self.persistence.getboolean(self.storage_name, i + '_active')
                    except (KeyError, configparser.NoOptionError):
                        self.logger.info('Failed to get ' + i + ' active from persistence storage, using default.')
                    try:
                        self.zone[i]['brightness'] = float(self.persistence[self.storage_name][i + '_brightness'])
                    except (KeyError, configparser.NoOptionError):
                        self.logger.info('Failed to get ' + i + ' brightness from persistence storage, using default.')
                    try:
                        for (index, item) in enumerate(self.persistence[self.storage_name][i + '_colors'].split(' ')):
                            self.zone[i]['colors'][index] = int(item)
                            if not 0 <= self.zone[i]['colors'][index] <= 255:
                                raise ValueError('Color out of range')
                        if len(self.zone[i]['colors']) != 9:
                            raise ValueError('There must be exactly 9 colors')
                    except ValueError:
                        self.zone[i]['colors'] = [0, 255, 0, 0, 255, 255, 0, 0, 255]
                        self.logger.info('%s: Invalid colors; restoring to defaults.', self.__class__.__name__)
                    except (KeyError, configparser.NoOptionError):
                        self.logger.info('Failed to get ' + i + ' colors from persistence storage, using default.')
                    try:
                        self.zone[i]['speed'] = int(self.persistence[self.storage_name][i + '_speed'])
                    except (KeyError, configparser.NoOptionError):
                        self.logger.info('Failed to get ' + i + ' speed from persistence storage, using default.')
                    try:
                        self.zone[i]['wave_dir'] = int(self.persistence[self.storage_name][i + '_wave_dir'])
                    except (KeyError, configparser.NoOptionError):
                        self.logger.info('Failed to get ' + i + ' wave direction from persistence storage, using default.')
        if 'get_battery' in self.METHODS:
            self._init_battery_manager()
        self.restore_dpi_poll_rate()
        self.restore_brightness()
        if self.config.getboolean('Startup', 'restore_persistence') is True:
            self.restore_effect()

    def send_effect_event(self, effect_name, *args):
        if False:
            return 10
        '\n        Send effect event\n\n        :param effect_name: Effect name\n        :type effect_name: str\n\n        :param args: Effect arguments\n        :type args: list\n        '
        payload = ['effect', self, effect_name]
        payload.extend(args)
        self.notify_observers(tuple(payload))

    def dedicated_macro_keys(self):
        if False:
            while True:
                i = 10
        '\n        Returns if the device has dedicated macro keys\n\n        :return: Macro keys\n        :rtype: bool\n        '
        return self.DEDICATED_MACRO_KEYS

    def restore_dpi_poll_rate(self):
        if False:
            while True:
                i = 10
        '\n        Set the device DPI & poll rate to the saved value\n        '
        dpi_func = getattr(self, 'setDPI', None)
        if dpi_func is not None:
            if self.dpi[0] > self.DPI_MAX:
                self.logger.warning('Constraining DPI X to maximum of ' + str(self.DPI_MAX) + ' because stored value ' + str(self.dpi[0]) + ' is larger.')
                self.dpi[0] = self.DPI_MAX
            if self.dpi[1] > self.DPI_MAX:
                self.logger.warning('Constraining DPI Y to maximum of ' + str(self.DPI_MAX) + ' because stored value ' + str(self.dpi[1]) + ' is larger.')
                self.dpi[1] = self.DPI_MAX
            dpi_func(self.dpi[0], self.dpi[1])
        poll_rate_func = getattr(self, 'setPollRate', None)
        if poll_rate_func is not None:
            if self.poll_rate not in self.POLL_RATES:
                self.logger.warning('Constraining poll rate because stored value ' + str(self.poll_rate) + ' is not available.')
                self.poll_rate = min(self.POLL_RATES, key=lambda x: abs(x - self.poll_rate))
            poll_rate_func(self.poll_rate)

    def restore_brightness(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the device to the current brightness/active state.\n\n        This is used at launch time.\n        '
        for i in self.ZONES:
            if self.zone[i]['present']:
                if 'set_' + i + '_active' in self.METHODS:
                    active_func = getattr(self, 'set' + self.capitalize_first_char(i) + 'Active', None)
                    if active_func is not None:
                        active_func(self.zone[i]['active'])
                bright_func = None
                if i == 'backlight':
                    bright_func = getattr(self, 'setBrightness', None)
                elif 'set_' + i + '_brightness' in self.METHODS:
                    bright_func = getattr(self, 'set' + self.capitalize_first_char(i) + 'Brightness', None)
                if bright_func is not None:
                    bright_func(self.zone[i]['brightness'])

    def disable_brightness(self):
        if False:
            while True:
                i = 10
        '\n        Set brightness to 0 and/or active state to false.\n        '
        for i in self.ZONES:
            if self.zone[i]['present']:
                if 'set_' + i + '_active' in self.METHODS:
                    active_func = getattr(self, 'set' + self.capitalize_first_char(i) + 'Active', None)
                    if active_func is not None:
                        active_func(False)
                bright_func = None
                if i == 'backlight':
                    bright_func = getattr(self, 'setBrightness', None)
                elif 'set_' + i + '_brightness' in self.METHODS:
                    bright_func = getattr(self, 'set' + self.capitalize_first_char(i) + 'Brightness', None)
                if bright_func is not None:
                    bright_func(0)

    def restore_effect(self):
        if False:
            while True:
                i = 10
        '\n        Set the device to the current effect\n\n        This is used at launch time and can be called by applications\n        that use custom matrix frames after they exit\n        '
        for i in self.ZONES:
            if self.zone[i]['present']:
                if i == 'backlight':
                    effect_func_name = 'set' + self.capitalize_first_char(self.zone[i]['effect'])
                else:
                    effect_func_name = 'set' + self.handle_underscores(self.capitalize_first_char(i)) + self.capitalize_first_char(self.zone[i]['effect'])
                effect_func = getattr(self, effect_func_name, None)
                if effect_func == None and (not self.zone[i]['effect'] == 'spectrum'):
                    self.logger.info('%s: Invalid effect name %s; restoring to Spectrum.', self.__class__.__name__, effect_func_name)
                    self.zone[i]['effect'] = 'spectrum'
                    if i == 'backlight':
                        effect_func_name = 'setSpectrum'
                    else:
                        effect_func_name = 'set' + self.capitalize_first_char(i) + 'Spectrum'
                    effect_func = getattr(self, effect_func_name, None)
                if effect_func is not None:
                    effect = self.zone[i]['effect']
                    colors = self.zone[i]['colors']
                    speed = self.zone[i]['speed']
                    wave_dir = self.zone[i]['wave_dir']
                    if self.get_num_arguments(effect_func) == 0:
                        effect_func()
                    elif self.get_num_arguments(effect_func) == 1:
                        if effect == 'starlightRandom':
                            effect_func(speed)
                        elif effect == 'wave':
                            effect_func(wave_dir)
                        elif effect == 'wheel':
                            effect_func(wave_dir)
                        elif effect == 'rippleRandomColour':
                            pass
                        else:
                            self.logger.error("%s: Effect requires 1 argument but don't know how to handle it!", self.__class__.__name__)
                    elif self.get_num_arguments(effect_func) == 3:
                        effect_func(colors[0], colors[1], colors[2])
                    elif self.get_num_arguments(effect_func) == 4:
                        if effect == 'starlightSingle' or effect == 'reactive':
                            effect_func(colors[0], colors[1], colors[2], speed)
                        elif effect == 'ripple':
                            pass
                        else:
                            self.logger.error("%s: Effect requires 4 arguments but don't know how to handle it!", self.__class__.__name__)
                    elif self.get_num_arguments(effect_func) == 6:
                        effect_func(colors[0], colors[1], colors[2], colors[3], colors[4], colors[5])
                    elif self.get_num_arguments(effect_func) == 7:
                        effect_func(colors[0], colors[1], colors[2], colors[3], colors[4], colors[5], speed)
                    elif self.get_num_arguments(effect_func) == 9:
                        effect_func(colors[0], colors[1], colors[2], colors[3], colors[4], colors[5], colors[6], colors[7], colors[8])
                    else:
                        self.logger.error("%s: Couldn't detect effect argument count!", self.__class__.__name__)

    def set_persistence(self, zone, key, value):
        if False:
            for i in range(10):
                print('nop')
        "\n        Set a device's current state for persisting across sessions.\n\n        :param zone: Zone\n        :type zone: string\n\n        :param key: Key\n        :type key: string\n\n        :param value: Value\n        :type value: string\n        "
        if self._disable_persistence:
            return
        self.logger.debug('Set persistence (%s, %s, %s)', zone, key, value)
        self.persistence.status['changed'] = True
        if zone:
            self.zone[zone][key] = value
        else:
            self.zone[key] = value

    def get_current_effect(self):
        if False:
            return 10
        "\n        Get the device's current effect\n\n        :return: Effect\n        :rtype: string\n        "
        self.logger.debug('DBus call get_current_effect')
        return self.zone['backlight']['effect']

    def get_current_effect_colors(self):
        if False:
            while True:
                i = 10
        "\n        Get the device's current effect's colors\n\n        :return: 3 colors\n        :rtype: list of byte\n        "
        self.logger.debug('DBus call get_current_effect_colors')
        return self.zone['backlight']['colors']

    def get_current_effect_speed(self):
        if False:
            return 10
        "\n        Get the device's current effect's speed\n\n        :return: Speed\n        :rtype: int\n        "
        self.logger.debug('DBus call get_current_effect_speed')
        return self.zone['backlight']['speed']

    def get_current_wave_dir(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Get the device's current wave direction\n\n        :return: Direction\n        :rtype: int\n        "
        self.logger.debug('DBus call get_current_wave_dir')
        return self.zone['backlight']['wave_dir']

    def get_current_logo_effect(self):
        if False:
            i = 10
            return i + 15
        "\n        Get the device's current logo effect\n\n        :return: Effect\n        :rtype: string\n        "
        self.logger.debug('DBus call get_current_logo_effect')
        return self.zone['logo']['effect']

    def get_current_logo_effect_colors(self):
        if False:
            return 10
        "\n        Get the device's current logo effect's colors\n\n        :return: 3 colors\n        :rtype: list of byte\n        "
        self.logger.debug('DBus call get_current_logo_effect_colors')
        return self.zone['logo']['colors']

    def get_current_logo_effect_speed(self):
        if False:
            return 10
        "\n        Get the device's current logo effect's speed\n\n        :return: Speed\n        :rtype: int\n        "
        self.logger.debug('DBus call get_current_logo_effect_speed')
        return self.zone['logo']['speed']

    def get_current_logo_wave_dir(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Get the device's current logo wave direction\n\n        :return: Direction\n        :rtype: int\n        "
        self.logger.debug('DBus call get_current_logo_wave_dir')
        return self.zone['logo']['wave_dir']

    def get_current_scroll_effect(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Get the device's current scroll effect\n\n        :return: Effect\n        :rtype: string\n        "
        self.logger.debug('DBus call get_current_scroll_effect')
        return self.zone['scroll']['effect']

    def get_current_scroll_effect_colors(self):
        if False:
            while True:
                i = 10
        "\n        Get the device's current scroll effect's colors\n\n        :return: 3 colors\n        :rtype: list of byte\n        "
        self.logger.debug('DBus call get_current_scroll_effect_colors')
        return self.zone['scroll']['colors']

    def get_current_scroll_effect_speed(self):
        if False:
            while True:
                i = 10
        "\n        Get the device's current scroll effect's speed\n\n        :return: Speed\n        :rtype: int\n        "
        self.logger.debug('DBus call get_current_scroll_effect_speed')
        return self.zone['scroll']['speed']

    def get_current_scroll_wave_dir(self):
        if False:
            while True:
                i = 10
        "\n        Get the device's current scroll wave direction\n\n        :return: Direction\n        :rtype: int\n        "
        self.logger.debug('DBus call get_current_scroll_wave_dir')
        return self.zone['scroll']['wave_dir']

    def get_current_left_effect(self):
        if False:
            print('Hello World!')
        "\n        Get the device's current left effect\n\n        :return: Effect\n        :rtype: string\n        "
        self.logger.debug('DBus call get_current_left_effect')
        return self.zone['left']['effect']

    def get_current_left_effect_colors(self):
        if False:
            return 10
        "\n        Get the device's current left effect's colors\n\n        :return: 3 colors\n        :rtype: list of byte\n        "
        self.logger.debug('DBus call get_current_left_effect_colors')
        return self.zone['left']['colors']

    def get_current_left_effect_speed(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Get the device's current left effect's speed\n\n        :return: Speed\n        :rtype: int\n        "
        self.logger.debug('DBus call get_current_left_effect_speed')
        return self.zone['left']['speed']

    def get_current_left_wave_dir(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Get the device's current left wave direction\n\n        :return: Direction\n        :rtype: int\n        "
        self.logger.debug('DBus call get_current_left_wave_dir')
        return self.zone['left']['wave_dir']

    def get_current_right_effect(self):
        if False:
            while True:
                i = 10
        "\n        Get the device's current right effect\n\n        :return: Effect\n        :rtype: string\n        "
        self.logger.debug('DBus call get_current_right_effect')
        return self.zone['right']['effect']

    def get_current_right_effect_colors(self):
        if False:
            print('Hello World!')
        "\n        Get the device's current right effect's colors\n\n        :return: 3 colors\n        :rtype: list of byte\n        "
        self.logger.debug('DBus call get_current_right_effect_colors')
        return self.zone['right']['colors']

    def get_current_right_effect_speed(self):
        if False:
            return 10
        "\n        Get the device's current right effect's speed\n\n        :return: Speed\n        :rtype: int\n        "
        self.logger.debug('DBus call get_current_right_effect_speed')
        return self.zone['right']['speed']

    def get_current_right_wave_dir(self):
        if False:
            print('Hello World!')
        "\n        Get the device's current right wave direction\n\n        :return: Direction\n        :rtype: int\n        "
        self.logger.debug('DBus call get_current_right_wave_dir')
        return self.zone['right']['wave_dir']

    def get_current_charging_effect(self):
        if False:
            print('Hello World!')
        "\n        Get the device's current charging effect\n\n        :return: Effect\n        :rtype: string\n        "
        self.logger.debug('DBus call get_current_charging_effect')
        return self.zone['charging']['effect']

    def get_current_charging_effect_colors(self):
        if False:
            return 10
        "\n        Get the device's current charging effect's colors\n\n        :return: 3 colors\n        :rtype: list of byte\n        "
        self.logger.debug('DBus call get_current_charging_effect_colors')
        return self.zone['charging']['colors']

    def get_current_charging_effect_speed(self):
        if False:
            print('Hello World!')
        "\n        Get the device's current charging effect's speed\n\n        :return: Speed\n        :rtype: int\n        "
        self.logger.debug('DBus call get_current_charging_effect_speed')
        return self.zone['charging']['speed']

    def get_current_charging_wave_dir(self):
        if False:
            i = 10
            return i + 15
        "\n        Get the device's current charging wave direction\n\n        :return: Direction\n        :rtype: int\n        "
        self.logger.debug('DBus call get_current_charging_wave_dir')
        return self.zone['charging']['wave_dir']

    def get_current_fast_charging_effect(self):
        if False:
            i = 10
            return i + 15
        "\n        Get the device's current fast_charging effect\n\n        :return: Effect\n        :rtype: string\n        "
        self.logger.debug('DBus call get_current_fast_charging_effect')
        return self.zone['fast_charging']['effect']

    def get_current_fast_charging_effect_colors(self):
        if False:
            print('Hello World!')
        "\n        Get the device's current fast_charging effect's colors\n\n        :return: 3 colors\n        :rtype: list of byte\n        "
        self.logger.debug('DBus call get_current_fast_charging_effect_colors')
        return self.zone['fast_charging']['colors']

    def get_current_fast_charging_effect_speed(self):
        if False:
            print('Hello World!')
        "\n        Get the device's current fast_charging effect's speed\n\n        :return: Speed\n        :rtype: int\n        "
        self.logger.debug('DBus call get_current_fast_charging_effect_speed')
        return self.zone['fast_charging']['speed']

    def get_current_fast_charging_wave_dir(self):
        if False:
            i = 10
            return i + 15
        "\n        Get the device's current fast_charging wave direction\n\n        :return: Direction\n        :rtype: int\n        "
        self.logger.debug('DBus call get_current_fast_charging_wave_dir')
        return self.zone['fast_charging']['wave_dir']

    def get_current_fully_charged_effect(self):
        if False:
            i = 10
            return i + 15
        "\n        Get the device's current fully_charged effect\n\n        :return: Effect\n        :rtype: string\n        "
        self.logger.debug('DBus call get_current_fully_charged_effect')
        return self.zone['fully_charged']['effect']

    def get_current_fully_charged_effect_colors(self):
        if False:
            i = 10
            return i + 15
        "\n        Get the device's current fully_charged effect's colors\n\n        :return: 3 colors\n        :rtype: list of byte\n        "
        self.logger.debug('DBus call get_current_fully_charged_effect_colors')
        return self.zone['fully_charged']['colors']

    def get_current_fully_charged_effect_speed(self):
        if False:
            i = 10
            return i + 15
        "\n        Get the device's current fully_charged effect's speed\n\n        :return: Speed\n        :rtype: int\n        "
        self.logger.debug('DBus call get_current_fully_charged_effect_speed')
        return self.zone['fully_charged']['speed']

    def get_current_fully_charged_wave_dir(self):
        if False:
            while True:
                i = 10
        "\n        Get the device's current fully_charged wave direction\n\n        :return: Direction\n        :rtype: int\n        "
        self.logger.debug('DBus call get_current_fully_charged_wave_dir')
        return self.zone['fully_charged']['wave_dir']

    @property
    def effect_sync(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Propagate the obsever call upwards, used for syncing effects\n\n        :return: Effects sync flag\n        :rtype: bool\n        '
        return self._effect_sync_propagate_up

    @effect_sync.setter
    def effect_sync(self, value):
        if False:
            for i in range(10):
                print('nop')
        '\n        Setting to true will propagate observer events upwards\n\n        :param value: Effect sync\n        :type value: bool\n        '
        self._effect_sync_propagate_up = value

    @property
    def disable_notify(self):
        if False:
            while True:
                i = 10
        '\n        Disable notifications flag\n\n        :return: Flag\n        :rtype: bool\n        '
        return self._disable_notifications

    @disable_notify.setter
    def disable_notify(self, value):
        if False:
            print('Hello World!')
        '\n        Set the disable notifications flag\n\n        :param value: Disable\n        :type value: bool\n        '
        self._disable_notifications = value

    @property
    def disable_persistence(self):
        if False:
            while True:
                i = 10
        '\n        Disable persistence flag\n\n        :return: Flag\n        :rtype: bool\n        '
        return self._disable_persistence

    @disable_persistence.setter
    def disable_persistence(self, value):
        if False:
            i = 10
            return i + 15
        '\n        Set the disable persistence flag\n\n        :param value: Disable\n        :type value: bool\n        '
        self._disable_persistence = value

    def get_driver_path(self, driver_filename):
        if False:
            print('Hello World!')
        '\n        Get the path to a driver file\n\n        :param driver_filename: Name of driver file\n        :type driver_filename: str\n\n        :return: Full path to driver\n        :rtype: str\n        '
        return os.path.join(self._device_path, driver_filename)

    def get_serial(self):
        if False:
            print('Hello World!')
        '\n        Get serial number for device\n\n        :return: String of the serial number\n        :rtype: str\n        '
        if self._serial is None:
            serial_path = os.path.join(self._device_path, 'device_serial')
            count = 0
            serial = ''
            while len(serial) == 0:
                if count >= 5:
                    break
                try:
                    with open(serial_path, 'r') as f:
                        serial = f.read().strip()
                except (PermissionError, OSError) as err:
                    self.logger.warning('getting serial: {0}'.format(err))
                    serial = ''
                except UnicodeDecodeError as err:
                    self.logger.warning('malformed serial: {0}'.format(err))
                    serial = ''
                count += 1
                if len(serial) == 0:
                    time.sleep(0.1)
                    self.logger.debug('getting serial: {0} count:{1}'.format(serial, count))
            if serial == '' or serial == 'Default string' or serial == 'empty (NULL)' or (serial == 'As printed in the D cover'):
                serial = 'UNKWN{0:012}'.format(random.randint(0, 4096))
            self._serial = serial.replace(' ', '_')
        return self._serial

    def get_device_mode(self):
        if False:
            i = 10
            return i + 15
        '\n        Get device mode\n\n        :return: String of device mode and arg separated by colon, e.g. 0:0 or 3:0\n        :rtype: str\n        '
        device_mode_path = os.path.join(self._device_path, 'device_mode')
        with open(device_mode_path, 'rb') as mode_file:
            count = 0
            mode = mode_file.read().strip()
            while len(mode) == 0:
                if count >= 3:
                    break
                mode = mode_file.read().strip()
                count += 1
                time.sleep(0.1)
            return '{0}:{1}'.format(mode[0], mode[1])

    def set_device_mode(self, mode_id, param):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set device mode\n\n        :param mode_id: Device mode ID\n        :type mode_id: int\n\n        :param param: Device mode parameter\n        :type param: int\n        '
        device_mode_path = os.path.join(self._device_path, 'device_mode')
        with open(device_mode_path, 'wb') as mode_file:
            if mode_id not in (0, 3):
                mode_id = 0
            if param != 0:
                param = 0
            mode_file.write(bytes([mode_id, param]))

    def _set_custom_effect(self):
        if False:
            while True:
                i = 10
        '\n        Set the device to use custom LED matrix\n        '
        driver_path = self.get_driver_path('matrix_effect_custom')
        payload = b'1'
        with open(driver_path, 'wb') as driver_file:
            driver_file.write(payload)

    def _set_key_row(self, payload):
        if False:
            while True:
                i = 10
        '\n        Set the RGB matrix on the device\n\n        Byte array like\n        [1, 255, 255, 00, 255, 255, 00, 255, 255, 00, 255, 255, 00, 255, 255, 00, 255, 255, 00, 255, 255, 00, 255, 255, 00,\n            255, 255, 00, 255, 255, 00, 255, 255, 00, 255, 255, 00, 255, 255, 00, 255, 255, 00, 255, 00, 00]\n\n        First byte is row, on firefly its always 1, on keyboard its 0-5\n        Then its 3byte groups of RGB\n        :param payload: Binary payload\n        :type payload: bytes\n        '
        driver_path = self.get_driver_path('matrix_custom_frame')
        with open(driver_path, 'wb') as driver_file:
            driver_file.write(payload)

    def _init_battery_manager(self):
        if False:
            return 10
        '\n        Initializes the BatteryManager using the provided name\n        '
        self._battery_manager = _BatteryManager(self, self._device_number, self.getDeviceName())
        self._battery_manager.active = self.config.getboolean('Startup', 'battery_notifier', fallback=False)
        self._battery_manager.frequency = self.config.getint('Startup', 'battery_notifier_freq', fallback=10 * 60)
        self._battery_manager.percent = self.config.getint('Startup', 'battery_notifier_percent', fallback=33)

    def get_vid_pid(self):
        if False:
            return 10
        '\n        Get the usb VID PID\n\n        :return: List of VID PID\n        :rtype: list of int\n        '
        result = [self.USB_VID, self.USB_PID]
        return result

    def get_image_json(self):
        if False:
            i = 10
            return i + 15
        return json.dumps({'top_img': self.get_device_image(), 'side_img': self.get_device_image(), 'perspective_img': self.get_device_image()})

    def get_device_image(self):
        if False:
            i = 10
            return i + 15
        return self.DEVICE_IMAGE

    def load_methods(self):
        if False:
            i = 10
            return i + 15
        '\n        Load DBus methods\n\n        Goes through the list in self.methods_internal and self.METHODS and loads each effect and adds it to DBus\n        '
        available_functions = {}
        methods = dir(openrazer_daemon.dbus_services.dbus_methods)
        for method in methods:
            potential_function = getattr(openrazer_daemon.dbus_services.dbus_methods, method)
            if isinstance(potential_function, types.FunctionType) and hasattr(potential_function, 'endpoint') and potential_function.endpoint:
                available_functions[potential_function.__name__] = potential_function
        self.methods_internal.extend(self.METHODS)
        for method_name in self.methods_internal:
            try:
                new_function = available_functions[method_name]
                self.logger.debug('Adding %s.%s method to DBus', new_function.interface, new_function.name)
                self.add_dbus_method(new_function.interface, new_function.name, new_function, new_function.in_sig, new_function.out_sig, new_function.byte_arrays)
            except KeyError as e:
                raise RuntimeError("Couldn't add method to DBus: " + str(e)) from None

    def suspend_device(self):
        if False:
            i = 10
            return i + 15
        '\n        Suspend device\n        '
        self.logger.info('Suspending %s', self.__class__.__name__)
        self.disable_notify = True
        self.disable_persistence = True
        self.disable_brightness()
        self._suspend_device()
        self.disable_notify = False
        self.disable_persistence = False

    def resume_device(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Resume device\n        '
        self.logger.info('Resuming %s', self.__class__.__name__)
        self.disable_notify = True
        self.disable_persistence = True
        self.restore_brightness()
        self._resume_device()
        self.disable_notify = False
        self.disable_persistence = False

    def _suspend_device(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Override to implement custom suspend behavior\n        '

    def _resume_device(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Override to implement custom resume behavior\n        '

    def _close(self):
        if False:
            while True:
                i = 10
        '\n        To be overridden by any subclasses to do cleanup\n        '
        self._observer_list.clear()
        if self._battery_manager:
            self._battery_manager.close()

    def close(self):
        if False:
            while True:
                i = 10
        '\n        Close any resources opened by subclasses\n        '
        if not self._is_closed:
            if 'get_dpi_xy' in self.METHODS:
                dpi_func = getattr(self, 'getDPI', None)
                if dpi_func is not None:
                    self.dpi = dpi_func()
            self._close()
            self._is_closed = True

    def register_observer(self, observer):
        if False:
            while True:
                i = 10
        '\n        Observer design pattern, register\n\n        :param observer: Observer\n        :type observer: object\n        '
        if observer not in self._observer_list:
            self._observer_list.append(observer)

    def register_parent(self, parent):
        if False:
            print('Hello World!')
        '\n        Register the parent as an observer to be optionally notified (sends to other devices)\n\n        :param parent: Observer\n        :type parent: object\n        '
        self._parent = parent

    def remove_observer(self, observer):
        if False:
            print('Hello World!')
        '\n        Obsever design pattern, remove\n\n        :param observer: Observer\n        :type observer: object\n        '
        try:
            self._observer_list.remove(observer)
        except ValueError:
            pass

    def notify_observers(self, msg):
        if False:
            print('Hello World!')
        '\n        Notify observers with msg\n\n        :param msg: Tuple with first element a string\n        :type msg: tuple\n        '
        if not self._disable_notifications:
            self.logger.debug('Sending observer message: %s', str(msg))
            if self._effect_sync_propagate_up and self._parent is not None:
                self._parent.notify_parent(msg)
            for observer in self._observer_list:
                observer.notify(msg)

    def notify(self, msg):
        if False:
            for i in range(10):
                print('nop')
        '\n        Receive observer messages\n\n        :param msg: Tuple with first element a string\n        :type msg: tuple\n        '
        self.logger.debug('Got observer message: %s', str(msg))
        for observer in self._observer_list:
            observer.notify(msg)

    @classmethod
    def match(cls, device_id, dev_path):
        if False:
            print('Hello World!')
        "\n        Match against the device ID\n\n        :param device_id: Device ID like 0000:0000:0000.0000\n        :type device_id: str\n\n        :param dev_path: Device path. Normally '/sys/bus/hid/devices/0000:0000:0000.0000'\n        :type dev_path: str\n\n        :return: True if its the correct device ID\n        :rtype: bool\n        "
        pattern = '^[0-9A-F]{4}:' + '{0:04X}'.format(cls.USB_VID) + ':' + '{0:04X}'.format(cls.USB_PID) + '\\.[0-9A-F]{4}$'
        if re.match(pattern, device_id) is not None:
            if 'device_type' in os.listdir(dev_path):
                return True
        return False

    @staticmethod
    def get_num_arguments(func):
        if False:
            while True:
                i = 10
        '\n        Get number of arguments in a function\n\n        :param func: Function\n        :type func: callable\n\n        :return: Number of arguments\n        :rtype: int\n        '
        func_sig = inspect.signature(func)
        return len(func_sig.parameters)

    @staticmethod
    def handle_underscores(string):
        if False:
            for i in range(10):
                print('nop')
        return re.sub('[_]+(?P<first>[a-z])', lambda m: m.group('first').upper(), string)

    @staticmethod
    def capitalize_first_char(string):
        if False:
            return 10
        return string[0].upper() + string[1:]

    def __del__(self):
        if False:
            print('Hello World!')
        self.close()

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '{0}:{1}'.format(self.__class__.__name__, self.serial)

class RazerDeviceBrightnessSuspend(RazerDevice):
    """
    Class for devices that have get_brightness and set_brightness
    """

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        if 'additional_methods' in kwargs:
            kwargs['additional_methods'].extend(['get_brightness', 'set_brightness'])
        else:
            kwargs['additional_methods'] = ['get_brightness', 'set_brightness']
        super().__init__(*args, **kwargs)