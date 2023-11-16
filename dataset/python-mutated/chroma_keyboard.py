"""
BlackWidow Chroma Effects
"""
import os
from openrazer_daemon.dbus_services import endpoint

@endpoint('razer.device.lighting.brightness', 'getBrightness', out_sig='d')
def get_brightness(self):
    if False:
        while True:
            i = 10
    "\n    Get the device's brightness\n\n    :return: Brightness\n    :rtype: float\n    "
    self.logger.debug('DBus call get_brightness')
    return self.zone['backlight']['brightness']

@endpoint('razer.device.lighting.brightness', 'setBrightness', in_sig='d')
def set_brightness(self, brightness):
    if False:
        i = 10
        return i + 15
    "\n    Set the device's brightness\n\n    :param brightness: Brightness\n    :type brightness: int\n    "
    self.logger.debug('DBus call set_brightness')
    driver_path = self.get_driver_path('matrix_brightness')
    self.method_args['brightness'] = brightness
    if brightness > 100:
        brightness = 100
    elif brightness < 0:
        brightness = 0
    self.set_persistence('backlight', 'brightness', int(brightness))
    brightness = int(round(brightness * (255.0 / 100.0)))
    with open(driver_path, 'w') as driver_file:
        driver_file.write(str(brightness))
    self.send_effect_event('setBrightness', brightness)

@endpoint('razer.device.led.gamemode', 'getGameMode', out_sig='b')
def get_game_mode(self):
    if False:
        for i in range(10):
            print('nop')
    '\n    Get game mode LED state\n\n    :return: Game mode LED state\n    :rtype: bool\n    '
    self.logger.debug('DBus call get_game_mode')
    driver_path = self.get_driver_path('game_led_state')
    with open(driver_path, 'r') as driver_file:
        return driver_file.read().strip() == '1'

@endpoint('razer.device.led.gamemode', 'setGameMode', in_sig='b')
def set_game_mode(self, enable):
    if False:
        for i in range(10):
            print('nop')
    '\n    Set game mode LED state\n\n    :param enable: Status of game mode\n    :type enable: bool\n    '
    self.logger.debug('DBus call set_game_mode')
    driver_path = self.get_driver_path('game_led_state')
    super_file = self.get_driver_path('key_super')
    alt_tab = self.get_driver_path('key_alt_tab')
    alt_f4 = self.get_driver_path('key_alt_f4')
    if os.path.exists(super_file):
        if enable:
            open(super_file, 'wb').write(b'\x01')
            open(alt_tab, 'wb').write(b'\x01')
            open(alt_f4, 'wb').write(b'\x01')
        else:
            open(super_file, 'wb').write(b'\x00')
            open(alt_tab, 'wb').write(b'\x00')
            open(alt_f4, 'wb').write(b'\x00')
    else:
        for kb_int in self.additional_interfaces:
            super_file = os.path.join(kb_int, 'key_super')
            alt_tab = os.path.join(kb_int, 'key_alt_tab')
            alt_f4 = os.path.join(kb_int, 'key_alt_f4')
            if os.path.exists(super_file):
                if enable:
                    open(super_file, 'wb').write(b'\x01')
                    open(alt_tab, 'wb').write(b'\x01')
                    open(alt_f4, 'wb').write(b'\x01')
                else:
                    open(super_file, 'wb').write(b'\x00')
                    open(alt_tab, 'wb').write(b'\x00')
                    open(alt_f4, 'wb').write(b'\x00')
    with open(driver_path, 'w') as driver_file:
        if enable:
            driver_file.write('1')
        else:
            driver_file.write('0')

@endpoint('razer.device.led.macromode', 'getMacroMode', out_sig='b')
def get_macro_mode(self):
    if False:
        return 10
    '\n    Get macro mode LED state\n\n    :return: Status of macro mode\n    :rtype: bool\n    '
    self.logger.debug('DBus call get_macro_mode')
    driver_path = self.get_driver_path('macro_led_state')
    with open(driver_path, 'r') as driver_file:
        return driver_file.read().strip() == '1'

@endpoint('razer.device.led.macromode', 'setMacroMode', in_sig='b')
def set_macro_mode(self, enable):
    if False:
        return 10
    '\n    Set macro mode LED state\n\n    :param enable: Status of macro mode\n    :type enable: bool\n    '
    self.logger.debug('DBus call set_macro_mode')
    driver_path = self.get_driver_path('macro_led_state')
    with open(driver_path, 'w') as driver_file:
        if enable:
            driver_file.write('1')
        else:
            driver_file.write('0')

@endpoint('razer.device.misc.keyswitchoptimization', 'getKeyswitchOptimization', out_sig='b')
def get_keyswitch_optimization(self):
    if False:
        i = 10
        return i + 15
    '\n    Get Keyswitch optimization state\n\n    :return: Status of keyswitch optimization\n    :rtype: bool\n    '
    self.logger.debug('DBus call get_keyswitch_optimization')
    driver_path = self.get_driver_path('keyswitch_optimization')
    with open(driver_path, 'r') as driver_file:
        return driver_file.read().strip() == '1'

@endpoint('razer.device.misc.keyswitchoptimization', 'setKeyswitchOptimization', in_sig='b')
def set_keyswitch_optimization(self, enable):
    if False:
        while True:
            i = 10
    '\n    Set Keyswitch optimization state\n\n    :param enable: Status of keyswitch optimization\n    :type enable: bool\n    '
    self.logger.debug('DBus call set_keyswitch_optimization')
    driver_path = self.get_driver_path('keyswitch_optimization')
    with open(driver_path, 'w') as driver_file:
        if enable:
            driver_file.write('1')
        else:
            driver_file.write('0')

@endpoint('razer.device.led.macromode', 'getMacroEffect', out_sig='i')
def get_macro_effect(self):
    if False:
        return 10
    '\n    Get the effect on the macro LED\n\n    :return: Macro LED effect ID\n    :rtype: int\n    '
    self.logger.debug('DBus call get_macro_effect')
    driver_path = self.get_driver_path('macro_led_effect')
    with open(driver_path, 'r') as driver_file:
        return int(driver_file.read().strip())

@endpoint('razer.device.led.macromode', 'setMacroEffect', in_sig='y')
def set_macro_effect(self, effect):
    if False:
        return 10
    '\n    Set the effect on the macro LED\n\n    :param effect: Macro LED effect ID\n    :type effect: int\n    '
    self.logger.debug('DBus call set_macro_effect')
    driver_path = self.get_driver_path('macro_led_effect')
    with open(driver_path, 'w') as driver_file:
        driver_file.write(str(int(effect)))

@endpoint('razer.device.lighting.chroma', 'setWave', in_sig='i')
def set_wave_effect(self, direction):
    if False:
        for i in range(10):
            print('nop')
    '\n    Set the wave effect on the device\n\n    :param direction: 1 - left to right, 2 right to left\n    :type direction: int\n    '
    self.logger.debug('DBus call set_wave_effect')
    self.send_effect_event('setWave', direction)
    self.set_persistence('backlight', 'effect', 'wave')
    self.set_persistence('backlight', 'wave_dir', int(direction))
    driver_path = self.get_driver_path('matrix_effect_wave')
    if direction not in self.WAVE_DIRS:
        direction = self.WAVE_DIRS[0]
    with open(driver_path, 'w') as driver_file:
        driver_file.write(str(direction))

@endpoint('razer.device.lighting.chroma', 'setWheel', in_sig='i')
def set_wheel_effect(self, direction):
    if False:
        for i in range(10):
            print('nop')
    '\n    Set the wheel effect on the device\n\n    :param direction: 1 - right, 2 - left\n    :type direction: int\n    '
    self.logger.debug('DBus call set_wheel_effect')
    self.send_effect_event('setWheel', direction)
    self.set_persistence('backlight', 'effect', 'wheel')
    self.set_persistence('backlight', 'wave_dir', int(direction))
    driver_path = self.get_driver_path('matrix_effect_wheel')
    if direction not in (1, 2):
        direction = 1
    with open(driver_path, 'w') as driver_file:
        driver_file.write(str(direction))

@endpoint('razer.device.lighting.chroma', 'setStatic', in_sig='yyy')
def set_static_effect(self, red, green, blue):
    if False:
        i = 10
        return i + 15
    '\n    Set the device to static colour\n\n    :param red: Red component\n    :type red: int\n\n    :param green: Green component\n    :type green: int\n\n    :param blue: Blue component\n    :type blue: int\n    '
    self.logger.debug('DBus call set_static_effect')
    self.send_effect_event('setStatic', red, green, blue)
    self.set_persistence('backlight', 'effect', 'static')
    self.zone['backlight']['colors'][0:3] = (int(red), int(green), int(blue))
    driver_path = self.get_driver_path('matrix_effect_static')
    payload = bytes([red, green, blue])
    with open(driver_path, 'wb') as driver_file:
        driver_file.write(payload)

@endpoint('razer.device.lighting.chroma', 'setBlinking', in_sig='yyy')
def set_blinking_effect(self, red, green, blue):
    if False:
        for i in range(10):
            print('nop')
    '\n    Set the device to static colour\n\n    :param red: Red component\n    :type red: int\n\n    :param green: Green component\n    :type green: int\n\n    :param blue: Blue component\n    :type blue: int\n    '
    self.logger.debug('DBus call set_blinking_effect')
    self.send_effect_event('setBlinking', red, green, blue)
    self.set_persistence('backlight', 'effect', 'blinking')
    self.zone['backlight']['colors'][0:3] = (int(red), int(green), int(blue))
    driver_path = self.get_driver_path('matrix_effect_blinking')
    payload = bytes([red, green, blue])
    with open(driver_path, 'wb') as driver_file:
        driver_file.write(payload)

@endpoint('razer.device.lighting.chroma', 'setSpectrum')
def set_spectrum_effect(self):
    if False:
        return 10
    '\n    Set the device to spectrum mode\n    '
    self.logger.debug('DBus call set_spectrum_effect')
    self.send_effect_event('setSpectrum')
    self.set_persistence('backlight', 'effect', 'spectrum')
    driver_path = self.get_driver_path('matrix_effect_spectrum')
    with open(driver_path, 'w') as driver_file:
        driver_file.write('1')

@endpoint('razer.device.lighting.chroma', 'setNone')
def set_none_effect(self):
    if False:
        for i in range(10):
            print('nop')
    '\n    Set the device to spectrum mode\n    '
    self.logger.debug('DBus call set_none_effect')
    self.send_effect_event('setNone')
    self.set_persistence('backlight', 'effect', 'none')
    driver_path = self.get_driver_path('matrix_effect_none')
    with open(driver_path, 'w') as driver_file:
        driver_file.write('1')

@endpoint('razer.device.misc', 'triggerReactive')
def trigger_reactive_effect(self):
    if False:
        while True:
            i = 10
    '\n    Trigger reactive on Firefly\n    '
    self.logger.debug('DBus call trigger_reactive_effect')
    self.send_effect_event('triggerReactive')
    driver_path = self.get_driver_path('matrix_reactive_trigger')
    with open(driver_path, 'w') as driver_file:
        driver_file.write('1')

@endpoint('razer.device.lighting.chroma', 'setReactive', in_sig='yyyy')
def set_reactive_effect(self, red, green, blue, speed):
    if False:
        i = 10
        return i + 15
    '\n    Set the device to reactive effect\n\n    :param red: Red component\n    :type red: int\n\n    :param green: Green component\n    :type green: int\n\n    :param blue: Blue component\n    :type blue: int\n\n    :param speed: Speed\n    :type speed: int\n    '
    self.logger.debug('DBus call set_reactive_effect')
    driver_path = self.get_driver_path('matrix_effect_reactive')
    self.send_effect_event('setReactive', red, green, blue, speed)
    self.set_persistence('backlight', 'effect', 'reactive')
    self.zone['backlight']['colors'][0:3] = (int(red), int(green), int(blue))
    if speed not in (1, 2, 3, 4):
        speed = 4
    self.set_persistence('backlight', 'speed', int(speed))
    payload = bytes([speed, red, green, blue])
    with open(driver_path, 'wb') as driver_file:
        driver_file.write(payload)

@endpoint('razer.device.lighting.chroma', 'setBreathRandom')
def set_breath_random_effect(self):
    if False:
        for i in range(10):
            print('nop')
    '\n    Set the device to random colour breathing effect\n    '
    self.logger.debug('DBus call set_breath_random_effect')
    self.send_effect_event('setBreathRandom')
    self.set_persistence('backlight', 'effect', 'breathRandom')
    driver_path = self.get_driver_path('matrix_effect_breath')
    payload = b'1'
    with open(driver_path, 'wb') as driver_file:
        driver_file.write(payload)

@endpoint('razer.device.lighting.chroma', 'setBreathSingle', in_sig='yyy')
def set_breath_single_effect(self, red, green, blue):
    if False:
        i = 10
        return i + 15
    '\n    Set the device to single colour breathing effect\n\n    :param red: Red component\n    :type red: int\n\n    :param green: Green component\n    :type green: int\n\n    :param blue: Blue component\n    :type blue: int\n    '
    self.logger.debug('DBus call set_breath_single_effect')
    self.send_effect_event('setBreathSingle', red, green, blue)
    self.set_persistence('backlight', 'effect', 'breathSingle')
    self.zone['backlight']['colors'][0:3] = (int(red), int(green), int(blue))
    driver_path = self.get_driver_path('matrix_effect_breath')
    payload = bytes([red, green, blue])
    with open(driver_path, 'wb') as driver_file:
        driver_file.write(payload)

@endpoint('razer.device.lighting.chroma', 'setBreathDual', in_sig='yyyyyy')
def set_breath_dual_effect(self, red1, green1, blue1, red2, green2, blue2):
    if False:
        for i in range(10):
            print('nop')
    '\n    Set the device to dual colour breathing effect\n\n    :param red1: Red component\n    :type red1: int\n\n    :param green1: Green component\n    :type green1: int\n\n    :param blue1: Blue component\n    :type blue1: int\n\n    :param red2: Red component\n    :type red2: int\n\n    :param green2: Green component\n    :type green2: int\n\n    :param blue2: Blue component\n    :type blue2: int\n    '
    self.logger.debug('DBus call set_breath_dual_effect')
    self.send_effect_event('setBreathDual', red1, green1, blue1, red2, green2, blue2)
    self.set_persistence('backlight', 'effect', 'breathDual')
    self.zone['backlight']['colors'][0:6] = (int(red1), int(green1), int(blue1), int(red2), int(green2), int(blue2))
    driver_path = self.get_driver_path('matrix_effect_breath')
    payload = bytes([red1, green1, blue1, red2, green2, blue2])
    with open(driver_path, 'wb') as driver_file:
        driver_file.write(payload)

@endpoint('razer.device.lighting.chroma', 'setBreathTriple', in_sig='yyyyyyyyy')
def set_breath_triple_effect(self, red1, green1, blue1, red2, green2, blue2, red3, green3, blue3):
    if False:
        return 10
    '\n    Set the device to triple colour breathing effect\n\n    :param red1: Red component\n    :type red1: int\n\n    :param green1: Green component\n    :type green1: int\n\n    :param blue1: Blue component\n    :type blue1: int\n\n    :param red2: Red component\n    :type red2: int\n\n    :param green2: Green component\n    :type green2: int\n\n    :param blue2: Blue component\n    :type blue2: int\n\n    :param red3: Red component\n    :type red3: int\n\n    :param green3: Green component\n    :type green3: int\n\n    :param blue3: Blue component\n    :type blue3: int\n    '
    self.logger.debug('DBus call set_breath_triple_effect')
    self.send_effect_event('setBreathTriple', red1, green1, blue1, red2, green2, blue2, red3, green3, blue3)
    self.set_persistence('backlight', 'effect', 'breathTriple')
    self.zone['backlight']['colors'][0:9] = (int(red1), int(green1), int(blue1), int(red2), int(green2), int(blue2), int(red3), int(green3), int(blue3))
    driver_path = self.get_driver_path('matrix_effect_breath')
    payload = bytes([red1, green1, blue1, red2, green2, blue2, red3, green3, blue3])
    with open(driver_path, 'wb') as driver_file:
        driver_file.write(payload)

@endpoint('razer.device.lighting.chroma', 'setCustom')
def set_custom_effect(self):
    if False:
        while True:
            i = 10
    '\n    Set the device to use custom LED matrix\n    '
    self.send_effect_event('setCustom')
    self._set_custom_effect()

@endpoint('razer.device.lighting.chroma', 'setKeyRow', in_sig='ay', byte_arrays=True)
def set_key_row(self, payload):
    if False:
        while True:
            i = 10
    '\n    Set the RGB matrix on the device\n\n    Byte array like\n    [1, 255, 255, 00, 255, 255, 00, 255, 255, 00, 255, 255, 00, 255, 255, 00, 255, 255, 00, 255, 255, 00, 255, 255, 00,\n        255, 255, 00, 255, 255, 00, 255, 255, 00, 255, 255, 00, 255, 255, 00, 255, 255, 00, 255, 00, 00]\n\n    First byte is row, on firefly its always 1, on keyboard its 0-5\n    Then its 3byte groups of RGB\n    :param payload: Binary payload\n    :type payload: bytes\n    '
    self.send_effect_event('setCustom')
    self._set_key_row(payload)

@endpoint('razer.device.lighting.custom', 'setRipple', in_sig='yyyd')
def set_ripple_effect(self, red, green, blue, refresh_rate):
    if False:
        while True:
            i = 10
    '\n    Set the daemon to serve a ripple effect of the specified colour\n\n    :param red: Red component\n    :type red: int\n\n    :param green: Green component\n    :type green: int\n\n    :param blue: Blue component\n    :type blue: int\n\n    :param refresh_rate: Refresh rate\n    :type refresh_rate: int\n    '
    self.logger.debug('DBus call set_ripple_effect')
    self.send_effect_event('setRipple', red, green, blue, refresh_rate)
    self.set_persistence('backlight', 'effect', 'ripple')
    self.zone['backlight']['colors'][0:3] = (int(red), int(green), int(blue))

@endpoint('razer.device.lighting.custom', 'setRippleRandomColour', in_sig='d')
def set_ripple_effect_random_colour(self, refresh_rate):
    if False:
        i = 10
        return i + 15
    '\n    Set the daemon to serve a ripple effect of random colours\n\n    :param refresh_rate: Refresh rate\n    :type refresh_rate: int\n    '
    self.logger.debug('DBus call set_ripple_effect')
    self.send_effect_event('setRipple', None, None, None, refresh_rate)
    self.set_persistence('backlight', 'effect', 'rippleRandomColour')

@endpoint('razer.device.lighting.chroma', 'setStarlightRandom', in_sig='y')
def set_starlight_random_effect(self, speed):
    if False:
        print('Hello World!')
    '\n    Set startlight random mode\n    '
    self.logger.debug('DBus call set_starlight_random')
    driver_path = self.get_driver_path('matrix_effect_starlight')
    with open(driver_path, 'wb') as driver_file:
        driver_file.write(bytes([speed]))
    self.send_effect_event('setStarlightRandom')
    self.set_persistence('backlight', 'effect', 'starlightRandom')
    self.set_persistence('backlight', 'speed', int(speed))

@endpoint('razer.device.lighting.chroma', 'setStarlightSingle', in_sig='yyyy')
def set_starlight_single_effect(self, red, green, blue, speed):
    if False:
        print('Hello World!')
    '\n    Set starlight mode\n    '
    self.logger.debug('DBus call set_starlight_single')
    driver_path = self.get_driver_path('matrix_effect_starlight')
    with open(driver_path, 'wb') as driver_file:
        driver_file.write(bytes([speed, red, green, blue]))
    self.send_effect_event('setStarlightSingle', red, green, blue, speed)
    self.set_persistence('backlight', 'effect', 'starlightSingle')
    self.set_persistence('backlight', 'speed', int(speed))
    self.zone['backlight']['colors'][0:3] = (int(red), int(green), int(blue))

@endpoint('razer.device.lighting.chroma', 'setStarlightDual', in_sig='yyyyyyy')
def set_starlight_dual_effect(self, red1, green1, blue1, red2, green2, blue2, speed):
    if False:
        i = 10
        return i + 15
    '\n    Set starlight dual mode\n    '
    self.logger.debug('DBus call set_starlight_dual')
    driver_path = self.get_driver_path('matrix_effect_starlight')
    with open(driver_path, 'wb') as driver_file:
        driver_file.write(bytes([speed, red1, green1, blue1, red2, green2, blue2]))
    self.send_effect_event('setStarlightDual', red1, green1, blue1, red2, green2, blue2, speed)
    self.set_persistence('backlight', 'effect', 'starlightDual')
    self.set_persistence('backlight', 'speed', int(speed))
    self.zone['backlight']['colors'][0:6] = (int(red1), int(green1), int(blue1), int(red2), int(green2), int(blue2))