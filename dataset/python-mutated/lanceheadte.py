from openrazer_daemon.dbus_services import endpoint

@endpoint('razer.device.lighting.logo', 'setLogoWave', in_sig='i')
def set_logo_wave(self, direction):
    if False:
        i = 10
        return i + 15
    '\n    Set the wave effect on the device\n\n    :param direction: (0|1) - down to up, (1|2) up to down\n    :type direction: int\n    '
    self.logger.debug('DBus call set_logo_wave')
    self.send_effect_event('setWave', direction)
    self.set_persistence('logo', 'effect', 'wave')
    self.set_persistence('logo', 'wave_dir', int(direction))
    driver_path = self.get_driver_path('logo_matrix_effect_wave')
    if direction not in self.WAVE_DIRS:
        direction = self.WAVE_DIRS[0]
    with open(driver_path, 'w') as driver_file:
        driver_file.write(str(direction))

@endpoint('razer.device.lighting.scroll', 'setScrollWave', in_sig='i')
def set_scroll_wave(self, direction):
    if False:
        i = 10
        return i + 15
    '\n    Set the wave effect on the device\n\n    :param direction: (0|1) - down to up, (1|2) up to down\n    :type direction: int\n    '
    self.logger.debug('DBus call set_scroll_wave')
    self.send_effect_event('setWave', direction)
    self.set_persistence('scroll', 'effect', 'wave')
    self.set_persistence('scroll', 'wave_dir', int(direction))
    driver_path = self.get_driver_path('scroll_matrix_effect_wave')
    if direction not in self.WAVE_DIRS:
        direction = self.WAVE_DIRS[0]
    with open(driver_path, 'w') as driver_file:
        driver_file.write(str(direction))

@endpoint('razer.device.lighting.left', 'getLeftBrightness', out_sig='d')
def get_left_brightness(self):
    if False:
        i = 10
        return i + 15
    "\n    Get the device's brightness\n\n    :return: Brightness\n    :rtype: float\n    "
    self.logger.debug('DBus call get_left_brightness')
    return self.zone['left']['brightness']

@endpoint('razer.device.lighting.left', 'setLeftBrightness', in_sig='d')
def set_left_brightness(self, brightness):
    if False:
        return 10
    "\n    Set the device's brightness\n\n    :param brightness: Brightness\n    :type brightness: int\n    "
    self.logger.debug('DBus call set_left_brightness')
    driver_path = self.get_driver_path('left_led_brightness')
    self.method_args['brightness'] = brightness
    if brightness > 100:
        brightness = 100
    elif brightness < 0:
        brightness = 0
    self.set_persistence('left', 'brightness', int(brightness))
    brightness = int(round(brightness * (255.0 / 100.0)))
    with open(driver_path, 'w') as driver_file:
        driver_file.write(str(brightness))
    self.send_effect_event('setBrightness', brightness)

@endpoint('razer.device.lighting.left', 'setLeftWave', in_sig='i')
def set_left_wave(self, direction):
    if False:
        return 10
    '\n    Set the wave effect on the device\n\n    :param direction: (0|1) - down to up, (1|2) up to down\n    :type direction: int\n    '
    self.logger.debug('DBus call set_left_wave')
    self.send_effect_event('setWave', direction)
    self.set_persistence('left', 'effect', 'wave')
    self.set_persistence('left', 'wave_dir', int(direction))
    driver_path = self.get_driver_path('left_matrix_effect_wave')
    if direction not in self.WAVE_DIRS:
        direction = self.WAVE_DIRS[0]
    with open(driver_path, 'w') as driver_file:
        driver_file.write(str(direction))

@endpoint('razer.device.lighting.left', 'setLeftStatic', in_sig='yyy')
def set_left_static(self, red, green, blue):
    if False:
        i = 10
        return i + 15
    '\n    Set the device to static colour\n\n    :param red: Red component\n    :type red: int\n\n    :param green: Green component\n    :type green: int\n\n    :param blue: Blue component\n    :type blue: int\n    '
    self.logger.debug('DBus call set_left_static')
    self.send_effect_event('setStatic', red, green, blue)
    self.set_persistence('left', 'effect', 'static')
    self.zone['left']['colors'][0:3] = (int(red), int(green), int(blue))
    rgb_driver_path = self.get_driver_path('left_matrix_effect_static')
    payload = bytes([red, green, blue])
    with open(rgb_driver_path, 'wb') as rgb_driver_file:
        rgb_driver_file.write(payload)

@endpoint('razer.device.lighting.left', 'setLeftSpectrum')
def set_left_spectrum(self):
    if False:
        i = 10
        return i + 15
    '\n    Set the device to spectrum mode\n    '
    self.logger.debug('DBus call set_left_spectrum')
    self.send_effect_event('setSpectrum')
    self.set_persistence('left', 'effect', 'spectrum')
    effect_driver_path = self.get_driver_path('left_matrix_effect_spectrum')
    with open(effect_driver_path, 'w') as effect_driver_file:
        effect_driver_file.write('1')

@endpoint('razer.device.lighting.left', 'setLeftNone')
def set_left_none(self):
    if False:
        print('Hello World!')
    '\n    Set the device to effect none\n    '
    self.logger.debug('DBus call set_left_none')
    self.send_effect_event('setNone')
    self.set_persistence('left', 'effect', 'none')
    driver_path = self.get_driver_path('left_matrix_effect_none')
    with open(driver_path, 'w') as driver_file:
        driver_file.write('1')

@endpoint('razer.device.lighting.left', 'setLeftReactive', in_sig='yyyy')
def set_left_reactive(self, red, green, blue, speed):
    if False:
        return 10
    '\n    Set the device to reactive effect\n\n    :param red: Red component\n    :type red: int\n\n    :param green: Green component\n    :type green: int\n\n    :param blue: Blue component\n    :type blue: int\n\n    :param speed: Speed\n    :type speed: int\n    '
    self.logger.debug('DBus call set_left_reactive')
    driver_path = self.get_driver_path('left_matrix_effect_reactive')
    self.send_effect_event('setReactive', red, green, blue, speed)
    self.set_persistence('left', 'effect', 'reactive')
    self.zone['left']['colors'][0:3] = (int(red), int(green), int(blue))
    if speed not in (1, 2, 3, 4):
        speed = 4
    self.set_persistence('left', 'speed', int(speed))
    payload = bytes([speed, red, green, blue])
    with open(driver_path, 'wb') as driver_file:
        driver_file.write(payload)

@endpoint('razer.device.lighting.left', 'setLeftBreathRandom')
def set_left_breath_random(self):
    if False:
        return 10
    '\n    Set the device to random colour breathing effect\n    '
    self.logger.debug('DBus call set_left_breath_random')
    self.send_effect_event('setBreathRandom')
    self.set_persistence('left', 'effect', 'breathRandom')
    driver_path = self.get_driver_path('left_matrix_effect_breath')
    payload = b'1'
    with open(driver_path, 'wb') as driver_file:
        driver_file.write(payload)

@endpoint('razer.device.lighting.left', 'setLeftBreathSingle', in_sig='yyy')
def set_left_breath_single(self, red, green, blue):
    if False:
        print('Hello World!')
    '\n    Set the device to single colour breathing effect\n\n    :param red: Red component\n    :type red: int\n\n    :param green: Green component\n    :type green: int\n\n    :param blue: Blue component\n    :type blue: int\n    '
    self.logger.debug('DBus call set_left_breath_single')
    self.send_effect_event('setBreathSingle', red, green, blue)
    self.set_persistence('left', 'effect', 'breathSingle')
    self.zone['left']['colors'][0:3] = (int(red), int(green), int(blue))
    driver_path = self.get_driver_path('left_matrix_effect_breath')
    payload = bytes([red, green, blue])
    with open(driver_path, 'wb') as driver_file:
        driver_file.write(payload)

@endpoint('razer.device.lighting.left', 'setLeftBreathDual', in_sig='yyyyyy')
def set_left_breath_dual(self, red1, green1, blue1, red2, green2, blue2):
    if False:
        while True:
            i = 10
    '\n    Set the device to dual colour breathing effect\n\n    :param red1: Red component\n    :type red1: int\n\n    :param green1: Green component\n    :type green1: int\n\n    :param blue1: Blue component\n    :type blue1: int\n\n    :param red2: Red component\n    :type red2: int\n\n    :param green2: Green component\n    :type green2: int\n\n    :param blue2: Blue component\n    :type blue2: int\n    '
    self.logger.debug('DBus call set_left_breath_dual')
    self.send_effect_event('setBreathDual', red1, green1, blue1, red2, green2, blue2)
    self.set_persistence('left', 'effect', 'breathDual')
    self.zone['left']['colors'][0:6] = (int(red1), int(green1), int(blue1), int(red2), int(green2), int(blue2))
    driver_path = self.get_driver_path('left_matrix_effect_breath')
    payload = bytes([red1, green1, blue1, red2, green2, blue2])
    with open(driver_path, 'wb') as driver_file:
        driver_file.write(payload)

@endpoint('razer.device.lighting.right', 'getRightBrightness', out_sig='d')
def get_right_brightness(self):
    if False:
        for i in range(10):
            print('nop')
    "\n    Get the device's brightness\n    :return: Brightness\n    :rtype: float\n    "
    self.logger.debug('DBus call get_right_brightness')
    return self.zone['right']['brightness']

@endpoint('razer.device.lighting.right', 'setRightBrightness', in_sig='d')
def set_right_brightness(self, brightness):
    if False:
        i = 10
        return i + 15
    "\n    Set the device's brightness\n\n    :param brightness: Brightness\n    :type brightness: int\n    "
    self.logger.debug('DBus call set_right_brightness')
    driver_path = self.get_driver_path('right_led_brightness')
    self.method_args['brightness'] = brightness
    if brightness > 100:
        brightness = 100
    elif brightness < 0:
        brightness = 0
    self.set_persistence('right', 'brightness', int(brightness))
    brightness = int(round(brightness * (255.0 / 100.0)))
    with open(driver_path, 'w') as driver_file:
        driver_file.write(str(brightness))
    self.send_effect_event('setBrightness', brightness)

@endpoint('razer.device.lighting.right', 'setRightWave', in_sig='i')
def set_right_wave(self, direction):
    if False:
        for i in range(10):
            print('nop')
    '\n    Set the wave effect on the device\n\n    :param direction: (0|1) - down to up, (1|2) up to down\n    :type direction: int\n    '
    self.logger.debug('DBus call set_right_wave')
    self.send_effect_event('setWave', direction)
    self.set_persistence('right', 'effect', 'wave')
    self.set_persistence('right', 'wave_dir', int(direction))
    driver_path = self.get_driver_path('right_matrix_effect_wave')
    if direction not in self.WAVE_DIRS:
        direction = self.WAVE_DIRS[0]
    with open(driver_path, 'w') as driver_file:
        driver_file.write(str(direction))

@endpoint('razer.device.lighting.right', 'setRightStatic', in_sig='yyy')
def set_right_static(self, red, green, blue):
    if False:
        while True:
            i = 10
    '\n    Set the device to static colour\n\n    :param red: Red component\n    :type red: int\n\n    :param green: Green component\n    :type green: int\n\n    :param blue: Blue component\n    :type blue: int\n    '
    self.logger.debug('DBus call set_right_static')
    self.send_effect_event('setStatic', red, green, blue)
    self.set_persistence('right', 'effect', 'static')
    self.zone['right']['colors'][0:3] = (int(red), int(green), int(blue))
    rgb_driver_path = self.get_driver_path('right_matrix_effect_static')
    payload = bytes([red, green, blue])
    with open(rgb_driver_path, 'wb') as rgb_driver_file:
        rgb_driver_file.write(payload)

@endpoint('razer.device.lighting.right', 'setRightSpectrum')
def set_right_spectrum(self):
    if False:
        for i in range(10):
            print('nop')
    '\n    Set the device to spectrum mode\n    '
    self.logger.debug('DBus call set_right_spectrum')
    self.send_effect_event('setSpectrum')
    self.set_persistence('right', 'effect', 'spectrum')
    effect_driver_path = self.get_driver_path('right_matrix_effect_spectrum')
    with open(effect_driver_path, 'w') as effect_driver_file:
        effect_driver_file.write('1')

@endpoint('razer.device.lighting.right', 'setRightNone')
def set_right_none(self):
    if False:
        return 10
    '\n    Set the device to effect none\n    '
    self.logger.debug('DBus call set_right_none')
    self.send_effect_event('setNone')
    self.set_persistence('right', 'effect', 'none')
    driver_path = self.get_driver_path('right_matrix_effect_none')
    with open(driver_path, 'w') as driver_file:
        driver_file.write('1')

@endpoint('razer.device.lighting.right', 'setRightReactive', in_sig='yyyy')
def set_right_reactive(self, red, green, blue, speed):
    if False:
        print('Hello World!')
    '\n    Set the device to reactive effect\n\n    :param red: Red component\n    :type red: int\n\n    :param green: Green component\n    :type green: int\n\n    :param blue: Blue component\n    :type blue: int\n\n    :param speed: Speed\n    :type speed: int\n    '
    self.logger.debug('DBus call set_right_reactive')
    driver_path = self.get_driver_path('right_matrix_effect_reactive')
    self.send_effect_event('setReactive', red, green, blue, speed)
    self.set_persistence('right', 'effect', 'reactive')
    self.zone['right']['colors'][0:3] = (int(red), int(green), int(blue))
    if speed not in (1, 2, 3, 4):
        speed = 4
    self.set_persistence('right', 'speed', int(speed))
    payload = bytes([speed, red, green, blue])
    with open(driver_path, 'wb') as driver_file:
        driver_file.write(payload)

@endpoint('razer.device.lighting.right', 'setRightBreathRandom')
def set_right_breath_random(self):
    if False:
        i = 10
        return i + 15
    '\n    Set the device to random colour breathing effect\n    '
    self.logger.debug('DBus call set_right_breath_random')
    self.send_effect_event('setBreathRandom')
    self.set_persistence('right', 'effect', 'breathRandom')
    driver_path = self.get_driver_path('right_matrix_effect_breath')
    payload = b'1'
    with open(driver_path, 'wb') as driver_file:
        driver_file.write(payload)

@endpoint('razer.device.lighting.right', 'setRightBreathSingle', in_sig='yyy')
def set_right_breath_single(self, red, green, blue):
    if False:
        for i in range(10):
            print('nop')
    '\n    Set the device to single colour breathing effect\n\n    :param red: Red component\n    :type red: int\n\n    :param green: Green component\n    :type green: int\n\n    :param blue: Blue component\n    :type blue: int\n    '
    self.logger.debug('DBus call set_right_breath_single')
    self.send_effect_event('setBreathSingle', red, green, blue)
    self.set_persistence('right', 'effect', 'breathSingle')
    self.zone['right']['colors'][0:3] = (int(red), int(green), int(blue))
    driver_path = self.get_driver_path('right_matrix_effect_breath')
    payload = bytes([red, green, blue])
    with open(driver_path, 'wb') as driver_file:
        driver_file.write(payload)

@endpoint('razer.device.lighting.right', 'setRightBreathDual', in_sig='yyyyyy')
def set_right_breath_dual(self, red1, green1, blue1, red2, green2, blue2):
    if False:
        i = 10
        return i + 15
    '\n    Set the device to dual colour breathing effect\n\n    :param red1: Red component\n    :type red1: int\n\n    :param green1: Green component\n    :type green1: int\n\n    :param blue1: Blue component\n    :type blue1: int\n\n    :param red2: Red component\n    :type red2: int\n\n    :param green2: Green component\n    :type green2: int\n\n    :param blue2: Blue component\n    :type blue2: int\n    '
    self.logger.debug('DBus call set_right_breath_dual')
    self.send_effect_event('setBreathDual', red1, green1, blue1, red2, green2, blue2)
    self.set_persistence('right', 'effect', 'breathDual')
    self.zone['right']['colors'][0:6] = (int(red1), int(green1), int(blue1), int(red2), int(green2), int(blue2))
    driver_path = self.get_driver_path('right_matrix_effect_breath')
    payload = bytes([red1, green1, blue1, red2, green2, blue2])
    with open(driver_path, 'wb') as driver_file:
        driver_file.write(payload)

@endpoint('razer.device.lighting.backlight', 'setBacklightWave', in_sig='i')
def set_backlight_wave(self, direction):
    if False:
        print('Hello World!')
    '\n    Set the wave effect on the device\n\n    :param direction: (0|1) - down to up, (1|2) up to down\n    :type direction: int\n    '
    self.logger.debug('DBus call set_backlight_wave')
    self.send_effect_event('setWave', direction)
    self.set_persistence('backlight', 'effect', 'wave')
    self.set_persistence('backlight', 'wave_dir', int(direction))
    driver_path = self.get_driver_path('backlight_matrix_effect_wave')
    if direction not in self.WAVE_DIRS:
        direction = self.WAVE_DIRS[0]
    with open(driver_path, 'w') as driver_file:
        driver_file.write(str(direction))

@endpoint('razer.device.lighting.backlight', 'setBacklightStatic', in_sig='yyy')
def set_backlight_static(self, red, green, blue):
    if False:
        for i in range(10):
            print('nop')
    '\n    Set the device to static colour\n\n    :param red: Red component\n    :type red: int\n\n    :param green: Green component\n    :type green: int\n\n    :param blue: Blue component\n    :type blue: int\n    '
    self.logger.debug('DBus call set_backlight_static')
    self.send_effect_event('setStatic', red, green, blue)
    self.set_persistence('backlight', 'effect', 'static')
    self.zone['backlight']['colors'][0:3] = (int(red), int(green), int(blue))
    rgb_driver_path = self.get_driver_path('backlight_matrix_effect_static')
    payload = bytes([red, green, blue])
    with open(rgb_driver_path, 'wb') as rgb_driver_file:
        rgb_driver_file.write(payload)

@endpoint('razer.device.lighting.backlight', 'setBacklightSpectrum')
def set_backlight_spectrum(self):
    if False:
        return 10
    '\n    Set the device to spectrum mode\n    '
    self.logger.debug('DBus call set_backlight_spectrum')
    self.send_effect_event('setSpectrum')
    self.set_persistence('backlight', 'effect', 'spectrum')
    effect_driver_path = self.get_driver_path('backlight_matrix_effect_spectrum')
    with open(effect_driver_path, 'w') as effect_driver_file:
        effect_driver_file.write('1')

@endpoint('razer.device.lighting.backlight', 'setBacklightNone')
def set_backlight_none(self):
    if False:
        return 10
    '\n    Set the device to effect none\n    '
    self.logger.debug('DBus call set_backlight_none')
    self.send_effect_event('setNone')
    self.set_persistence('backlight', 'effect', 'none')
    driver_path = self.get_driver_path('backlight_matrix_effect_none')
    with open(driver_path, 'w') as driver_file:
        driver_file.write('1')

@endpoint('razer.device.lighting.backlight', 'setBacklightOn')
def set_backlight_on(self):
    if False:
        return 10
    '\n    Set the device to effect on\n    '
    self.logger.debug('DBus call set_backlight_on')
    self.send_effect_event('setOn')
    self.set_persistence('backlight', 'effect', 'on')
    driver_path = self.get_driver_path('backlight_matrix_effect_on')
    with open(driver_path, 'w') as driver_file:
        driver_file.write('1')

@endpoint('razer.device.lighting.backlight', 'setBacklightReactive', in_sig='yyyy')
def set_backlight_reactive(self, red, green, blue, speed):
    if False:
        for i in range(10):
            print('nop')
    '\n    Set the device to reactive effect\n\n    :param red: Red component\n    :type red: int\n\n    :param green: Green component\n    :type green: int\n\n    :param blue: Blue component\n    :type blue: int\n\n    :param speed: Speed\n    :type speed: int\n    '
    self.logger.debug('DBus call set_backlight_reactive')
    driver_path = self.get_driver_path('backlight_matrix_effect_reactive')
    self.send_effect_event('setReactive', red, green, blue, speed)
    self.set_persistence('backlight', 'effect', 'reactive')
    self.zone['backlight']['colors'][0:3] = (int(red), int(green), int(blue))
    if speed not in (1, 2, 3, 4):
        speed = 4
    self.set_persistence('backlight', 'speed', int(speed))
    payload = bytes([speed, red, green, blue])
    with open(driver_path, 'wb') as driver_file:
        driver_file.write(payload)

@endpoint('razer.device.lighting.backlight', 'setBacklightBreathRandom')
def set_backlight_breath_random(self):
    if False:
        for i in range(10):
            print('nop')
    '\n    Set the device to random colour breathing effect\n    '
    self.logger.debug('DBus call set_backlight_breath_random')
    self.send_effect_event('setBreathRandom')
    self.set_persistence('backlight', 'effect', 'breathRandom')
    driver_path = self.get_driver_path('backlight_matrix_effect_breath')
    payload = b'1'
    with open(driver_path, 'wb') as driver_file:
        driver_file.write(payload)

@endpoint('razer.device.lighting.backlight', 'setBacklightBreathSingle', in_sig='yyy')
def set_backlight_breath_single(self, red, green, blue):
    if False:
        return 10
    '\n    Set the device to single colour breathing effect\n\n    :param red: Red component\n    :type red: int\n\n    :param green: Green component\n    :type green: int\n\n    :param blue: Blue component\n    :type blue: int\n    '
    self.logger.debug('DBus call set_backlight_breath_single')
    self.send_effect_event('setBreathSingle', red, green, blue)
    self.set_persistence('backlight', 'effect', 'breathSingle')
    self.zone['backlight']['colors'][0:3] = (int(red), int(green), int(blue))
    driver_path = self.get_driver_path('backlight_matrix_effect_breath')
    payload = bytes([red, green, blue])
    with open(driver_path, 'wb') as driver_file:
        driver_file.write(payload)

@endpoint('razer.device.lighting.backlight', 'setBacklightBreathDual', in_sig='yyyyyy')
def set_backlight_breath_dual(self, red1, green1, blue1, red2, green2, blue2):
    if False:
        i = 10
        return i + 15
    '\n    Set the device to dual colour breathing effect\n\n    :param red1: Red component\n    :type red1: int\n\n    :param green1: Green component\n    :type green1: int\n\n    :param blue1: Blue component\n    :type blue1: int\n\n    :param red2: Red component\n    :type red2: int\n\n    :param green2: Green component\n    :type green2: int\n\n    :param blue2: Blue component\n    :type blue2: int\n    '
    self.logger.debug('DBus call set_backlight_breath_dual')
    self.send_effect_event('setBreathDual', red1, green1, blue1, red2, green2, blue2)
    self.set_persistence('backlight', 'effect', 'breathDual')
    self.zone['backlight']['colors'][0:6] = (int(red1), int(green1), int(blue1), int(red2), int(green2), int(blue2))
    driver_path = self.get_driver_path('backlight_matrix_effect_breath')
    payload = bytes([red1, green1, blue1, red2, green2, blue2])
    with open(driver_path, 'wb') as driver_file:
        driver_file.write(payload)