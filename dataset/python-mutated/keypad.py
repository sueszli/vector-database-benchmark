"""
Keypad Effects
"""
from openrazer_daemon.dbus_services import endpoint

@endpoint('razer.device.lighting.profile_led', 'getRedLED', out_sig='b')
def keypad_get_profile_led_red(self):
    if False:
        print('Hello World!')
    '\n    Get red profile LED state\n\n    :return: Red profile LED state\n    :rtype: bool\n    '
    self.logger.debug('DBus call keypad_profile_led_red')
    driver_path = self.get_driver_path('profile_led_red')
    with open(driver_path, 'r') as driver_file:
        return driver_file.read().strip() == '1'

@endpoint('razer.device.lighting.profile_led', 'setRedLED', in_sig='b')
def keypad_set_profile_led_red(self, enable):
    if False:
        while True:
            i = 10
    '\n    Set red profile LED state\n\n    :param enable: Status of red profile LED\n    :type enable: bool\n    '
    self.logger.debug('DBus call keypad_set_profile_led_red')
    driver_path = self.get_driver_path('profile_led_red')
    with open(driver_path, 'w') as driver_file:
        if enable:
            driver_file.write('1')
        else:
            driver_file.write('0')

@endpoint('razer.device.lighting.profile_led', 'getGreenLED', out_sig='b')
def keypad_get_profile_led_green(self):
    if False:
        for i in range(10):
            print('nop')
    '\n    Get green profile LED state\n\n    :return: Green profile LED state\n    :rtype: bool\n    '
    self.logger.debug('DBus call keypad_get_profile_led_green')
    driver_path = self.get_driver_path('profile_led_green')
    with open(driver_path, 'r') as driver_file:
        return driver_file.read().strip() == '1'

@endpoint('razer.device.lighting.profile_led', 'setGreenLED', in_sig='b')
def keypad_set_profile_led_green(self, enable):
    if False:
        return 10
    '\n    Set green profile LED state\n\n    :param enable: Status of green profile LED\n    :type enable: bool\n    '
    self.logger.debug('DBus call keypad_set_profile_led_green')
    driver_path = self.get_driver_path('profile_led_green')
    with open(driver_path, 'w') as driver_file:
        if enable:
            driver_file.write('1')
        else:
            driver_file.write('0')

@endpoint('razer.device.lighting.profile_led', 'getBlueLED', out_sig='b')
def keypad_get_profile_led_blue(self):
    if False:
        while True:
            i = 10
    '\n    Get blue profile LED state\n\n    :return: Blue profile LED state\n    :rtype: bool\n    '
    self.logger.debug('DBus call keypad_get_profile_led_blue')
    driver_path = self.get_driver_path('profile_led_blue')
    with open(driver_path, 'r') as driver_file:
        return driver_file.read().strip() == '1'

@endpoint('razer.device.lighting.profile_led', 'setBlueLED', in_sig='b')
def keypad_set_profile_led_blue(self, enable):
    if False:
        i = 10
        return i + 15
    '\n    Set blue profile LED state\n\n    :param enable: Status of blue profile LED\n    :type enable: bool\n    '
    self.logger.debug('DBus call keypad_set_profile_led_blue')
    driver_path = self.get_driver_path('profile_led_blue')
    with open(driver_path, 'w') as driver_file:
        if enable:
            driver_file.write('1')
        else:
            driver_file.write('0')

@endpoint('razer.device.macro', 'getModeModifier', out_sig='b')
def keypad_get_mode_modifier(self):
    if False:
        i = 10
        return i + 15
    '\n    Get if the mode key is a modifier\n\n    :return: State\n    :rtype: bool\n    '
    self.logger.debug('DBus call keypad_get_mode_modifier')
    return self.key_manager.mode_modifier

@endpoint('razer.device.macro', 'setModeModifier', in_sig='b')
def keypad_set_mode_modifier(self, modifier):
    if False:
        return 10
    '\n    Set if the mode key is a modifier\n\n    :param modifier: State\n    :type modifier: bool\n    '
    self.logger.debug('DBus call keypad_set_mode_modifier')
    self.key_manager.mode_modifier = modifier