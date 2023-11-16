from openrazer_daemon.dbus_services import endpoint

@endpoint('razer.device.lighting.charging', 'getChargingBrightness', out_sig='d')
def get_charging_brightness(self):
    if False:
        for i in range(10):
            print('nop')
    "\n    Get the device's brightness\n    :return: Brightness\n    :rtype: float\n    "
    self.logger.debug('DBus call get_charging_brightness')
    return self.zone['charging']['brightness']

@endpoint('razer.device.lighting.charging', 'setChargingBrightness', in_sig='d')
def set_charging_brightness(self, brightness):
    if False:
        for i in range(10):
            print('nop')
    "\n    Set the device's brightness\n    :param brightness: Brightness\n    :type brightness: int\n    "
    self.logger.debug('DBus call set_charging_brightness')
    driver_path = self.get_driver_path('charging_led_brightness')
    self.method_args['brightness'] = brightness
    if brightness > 100:
        brightness = 100
    elif brightness < 0:
        brightness = 0
    self.set_persistence('charging', 'brightness', int(brightness))
    brightness = int(round(brightness * (255.0 / 100.0)))
    with open(driver_path, 'w') as driver_file:
        driver_file.write(str(brightness))
    self.send_effect_event('setBrightness', brightness)

@endpoint('razer.device.lighting.charging', 'setChargingWave', in_sig='i')
def set_charging_wave(self, direction):
    if False:
        return 10
    '\n    Set the wave effect on the device\n    :param direction: (0|1) - down to up, (1|2) up to down\n    :type direction: int\n    '
    self.logger.debug('DBus call set_charging_wave')
    self.send_effect_event('setWave', direction)
    self.set_persistence('charging', 'effect', 'wave')
    self.set_persistence('charging', 'wave_dir', int(direction))
    driver_path = self.get_driver_path('charging_matrix_effect_wave')
    if direction not in self.WAVE_DIRS:
        direction = self.WAVE_DIRS[0]
    with open(driver_path, 'w') as driver_file:
        driver_file.write(str(direction))

@endpoint('razer.device.lighting.charging', 'setChargingStatic', in_sig='yyy')
def set_charging_static(self, red, green, blue):
    if False:
        i = 10
        return i + 15
    '\n    Set the device to static colour\n\n    :param red: Red component\n    :type red: int\n\n    :param green: Green component\n    :type green: int\n\n    :param blue: Blue component\n    :type blue: int\n    '
    self.logger.debug('DBus call set_charging_static')
    self.send_effect_event('setStatic', red, green, blue)
    self.set_persistence('charging', 'effect', 'static')
    self.zone['charging']['colors'][0:3] = (int(red), int(green), int(blue))
    rgb_driver_path = self.get_driver_path('charging_matrix_effect_static')
    payload = bytes([red, green, blue])
    with open(rgb_driver_path, 'wb') as rgb_driver_file:
        rgb_driver_file.write(payload)

@endpoint('razer.device.lighting.charging', 'setChargingSpectrum')
def set_charging_spectrum(self):
    if False:
        i = 10
        return i + 15
    '\n    Set the device to spectrum mode\n    '
    self.logger.debug('DBus call set_charging_spectrum')
    self.send_effect_event('setSpectrum')
    self.set_persistence('charging', 'effect', 'spectrum')
    effect_driver_path = self.get_driver_path('charging_matrix_effect_spectrum')
    with open(effect_driver_path, 'w') as effect_driver_file:
        effect_driver_file.write('1')

@endpoint('razer.device.lighting.charging', 'setChargingNone')
def set_charging_none(self):
    if False:
        i = 10
        return i + 15
    '\n    Set the device to effect none\n    '
    self.logger.debug('DBus call set_charging_none')
    self.send_effect_event('setNone')
    self.set_persistence('charging', 'effect', 'none')
    driver_path = self.get_driver_path('charging_matrix_effect_none')
    with open(driver_path, 'w') as driver_file:
        driver_file.write('1')

@endpoint('razer.device.lighting.charging', 'setChargingBreathRandom')
def set_charging_breath_random(self):
    if False:
        while True:
            i = 10
    '\n    Set the device to random colour breathing effect\n    '
    self.logger.debug('DBus call set_charging_breath_random')
    self.send_effect_event('setBreathRandom')
    self.set_persistence('charging', 'effect', 'breathRandom')
    driver_path = self.get_driver_path('charging_matrix_effect_breath')
    payload = b'1'
    with open(driver_path, 'wb') as driver_file:
        driver_file.write(payload)

@endpoint('razer.device.lighting.charging', 'setChargingBreathSingle', in_sig='yyy')
def set_charging_breath_single(self, red, green, blue):
    if False:
        for i in range(10):
            print('nop')
    '\n    Set the device to single colour breathing effect\n\n    :param red: Red component\n    :type red: int\n\n    :param green: Green component\n    :type green: int\n\n    :param blue: Blue component\n    :type blue: int\n    '
    self.logger.debug('DBus call set_charging_breath_single')
    self.send_effect_event('setBreathSingle', red, green, blue)
    self.set_persistence('charging', 'effect', 'breathSingle')
    self.zone['charging']['colors'][0:3] = (int(red), int(green), int(blue))
    driver_path = self.get_driver_path('charging_matrix_effect_breath')
    payload = bytes([red, green, blue])
    with open(driver_path, 'wb') as driver_file:
        driver_file.write(payload)

@endpoint('razer.device.lighting.charging', 'setChargingBreathDual', in_sig='yyyyyy')
def set_charging_breath_dual(self, red1, green1, blue1, red2, green2, blue2):
    if False:
        print('Hello World!')
    '\n    Set the device to dual colour breathing effect\n\n    :param red1: Red component\n    :type red1: int\n\n    :param green1: Green component\n    :type green1: int\n\n    :param blue1: Blue component\n    :type blue1: int\n\n    :param red2: Red component\n    :type red2: int\n\n    :param green2: Green component\n    :type green2: int\n\n    :param blue2: Blue component\n    :type blue2: int\n    '
    self.logger.debug('DBus call set_charging_breath_dual')
    self.send_effect_event('setBreathDual', red1, green1, blue1, red2, green2, blue2)
    self.set_persistence('charging', 'effect', 'breathDual')
    self.zone['charging']['colors'][0:6] = (int(red1), int(green1), int(blue1), int(red2), int(green2), int(blue2))
    driver_path = self.get_driver_path('charging_matrix_effect_breath')
    payload = bytes([red1, green1, blue1, red2, green2, blue2])
    with open(driver_path, 'wb') as driver_file:
        driver_file.write(payload)

@endpoint('razer.device.lighting.fast_charging', 'getFastChargingBrightness', out_sig='d')
def get_fast_charging_brightness(self):
    if False:
        print('Hello World!')
    "\n    Get the device's brightness\n    :return: Brightness\n    :rtype: float\n    "
    self.logger.debug('DBus call get_fast_charging_brightness')
    return self.zone['fast_charging']['brightness']

@endpoint('razer.device.lighting.fast_charging', 'setFastChargingBrightness', in_sig='d')
def set_fast_charging_brightness(self, brightness):
    if False:
        i = 10
        return i + 15
    "\n    Set the device's brightness\n    :param brightness: Brightness\n    :type brightness: int\n    "
    self.logger.debug('DBus call set_fast_charging_brightness')
    driver_path = self.get_driver_path('fast_charging_led_brightness')
    self.method_args['brightness'] = brightness
    if brightness > 100:
        brightness = 100
    elif brightness < 0:
        brightness = 0
    self.set_persistence('fast_charging', 'brightness', int(brightness))
    brightness = int(round(brightness * (255.0 / 100.0)))
    with open(driver_path, 'w') as driver_file:
        driver_file.write(str(brightness))
    self.send_effect_event('setBrightness', brightness)

@endpoint('razer.device.lighting.fast_charging', 'setFastChargingWave', in_sig='i')
def set_fast_charging_wave(self, direction):
    if False:
        for i in range(10):
            print('nop')
    '\n    Set the wave effect on the device\n    :param direction: (0|1) - down to up, (1|2) up to down\n    :type direction: int\n    '
    self.logger.debug('DBus call set_fast_charging_wave')
    self.send_effect_event('setWave', direction)
    self.set_persistence('fast_charging', 'effect', 'wave')
    self.set_persistence('fast_charging', 'wave_dir', int(direction))
    driver_path = self.get_driver_path('fast_charging_matrix_effect_wave')
    if direction not in self.WAVE_DIRS:
        direction = self.WAVE_DIRS[0]
    with open(driver_path, 'w') as driver_file:
        driver_file.write(str(direction))

@endpoint('razer.device.lighting.fast_charging', 'setFastChargingStatic', in_sig='yyy')
def set_fast_charging_static(self, red, green, blue):
    if False:
        return 10
    '\n    Set the device to static colour\n\n    :param red: Red component\n    :type red: int\n\n    :param green: Green component\n    :type green: int\n\n    :param blue: Blue component\n    :type blue: int\n    '
    self.logger.debug('DBus call set_fast_charging_static')
    self.send_effect_event('setStatic', red, green, blue)
    self.set_persistence('fast_charging', 'effect', 'static')
    self.zone['fast_charging']['colors'][0:3] = (int(red), int(green), int(blue))
    rgb_driver_path = self.get_driver_path('fast_charging_matrix_effect_static')
    payload = bytes([red, green, blue])
    with open(rgb_driver_path, 'wb') as rgb_driver_file:
        rgb_driver_file.write(payload)

@endpoint('razer.device.lighting.fast_charging', 'setFastChargingSpectrum')
def set_fast_charging_spectrum(self):
    if False:
        i = 10
        return i + 15
    '\n    Set the device to spectrum mode\n    '
    self.logger.debug('DBus call set_fast_charging_spectrum')
    self.send_effect_event('setSpectrum')
    self.set_persistence('fast_charging', 'effect', 'spectrum')
    effect_driver_path = self.get_driver_path('fast_charging_matrix_effect_spectrum')
    with open(effect_driver_path, 'w') as effect_driver_file:
        effect_driver_file.write('1')

@endpoint('razer.device.lighting.fast_charging', 'setFastChargingNone')
def set_fast_charging_none(self):
    if False:
        print('Hello World!')
    '\n    Set the device to effect none\n    '
    self.logger.debug('DBus call set_fast_charging_none')
    self.send_effect_event('setNone')
    self.set_persistence('fast_charging', 'effect', 'none')
    driver_path = self.get_driver_path('fast_charging_matrix_effect_none')
    with open(driver_path, 'w') as driver_file:
        driver_file.write('1')

@endpoint('razer.device.lighting.fast_charging', 'setFastChargingBreathRandom')
def set_fast_charging_breath_random(self):
    if False:
        i = 10
        return i + 15
    '\n    Set the device to random colour breathing effect\n    '
    self.logger.debug('DBus call set_fast_charging_breath_random')
    self.send_effect_event('setBreathRandom')
    self.set_persistence('fast_charging', 'effect', 'breathRandom')
    driver_path = self.get_driver_path('fast_charging_matrix_effect_breath')
    payload = b'1'
    with open(driver_path, 'wb') as driver_file:
        driver_file.write(payload)

@endpoint('razer.device.lighting.fast_charging', 'setFastChargingBreathSingle', in_sig='yyy')
def set_fast_charging_breath_single(self, red, green, blue):
    if False:
        for i in range(10):
            print('nop')
    '\n    Set the device to single colour breathing effect\n\n    :param red: Red component\n    :type red: int\n\n    :param green: Green component\n    :type green: int\n\n    :param blue: Blue component\n    :type blue: int\n    '
    self.logger.debug('DBus call set_fast_charging_breath_single')
    self.send_effect_event('setBreathSingle', red, green, blue)
    self.set_persistence('fast_charging', 'effect', 'breathSingle')
    self.zone['fast_charging']['colors'][0:3] = (int(red), int(green), int(blue))
    driver_path = self.get_driver_path('fast_charging_matrix_effect_breath')
    payload = bytes([red, green, blue])
    with open(driver_path, 'wb') as driver_file:
        driver_file.write(payload)

@endpoint('razer.device.lighting.fast_charging', 'setFastChargingBreathDual', in_sig='yyyyyy')
def set_fast_charging_breath_dual(self, red1, green1, blue1, red2, green2, blue2):
    if False:
        while True:
            i = 10
    '\n    Set the device to dual colour breathing effect\n\n    :param red1: Red component\n    :type red1: int\n\n    :param green1: Green component\n    :type green1: int\n\n    :param blue1: Blue component\n    :type blue1: int\n\n    :param red2: Red component\n    :type red2: int\n\n    :param green2: Green component\n    :type green2: int\n\n    :param blue2: Blue component\n    :type blue2: int\n    '
    self.logger.debug('DBus call set_fast_charging_breath_dual')
    self.send_effect_event('setBreathDual', red1, green1, blue1, red2, green2, blue2)
    self.set_persistence('fast_charging', 'effect', 'breathDual')
    self.zone['fast_charging']['colors'][0:6] = (int(red1), int(green1), int(blue1), int(red2), int(green2), int(blue2))
    driver_path = self.get_driver_path('fast_charging_matrix_effect_breath')
    payload = bytes([red1, green1, blue1, red2, green2, blue2])
    with open(driver_path, 'wb') as driver_file:
        driver_file.write(payload)

@endpoint('razer.device.lighting.fully_charged', 'getFullyChargedBrightness', out_sig='d')
def get_fully_charged_brightness(self):
    if False:
        i = 10
        return i + 15
    "\n    Get the device's brightness\n    :return: Brightness\n    :rtype: float\n    "
    self.logger.debug('DBus call get_fully_charged_brightness')
    return self.zone['fully_charged']['brightness']

@endpoint('razer.device.lighting.fully_charged', 'setFullyChargedBrightness', in_sig='d')
def set_fully_charged_brightness(self, brightness):
    if False:
        for i in range(10):
            print('nop')
    "\n    Set the device's brightness\n    :param brightness: Brightness\n    :type brightness: int\n    "
    self.logger.debug('DBus call set_fully_charged_brightness')
    driver_path = self.get_driver_path('fully_charged_led_brightness')
    self.method_args['brightness'] = brightness
    if brightness > 100:
        brightness = 100
    elif brightness < 0:
        brightness = 0
    self.set_persistence('fully_charged', 'brightness', int(brightness))
    brightness = int(round(brightness * (255.0 / 100.0)))
    with open(driver_path, 'w') as driver_file:
        driver_file.write(str(brightness))
    self.send_effect_event('setBrightness', brightness)

@endpoint('razer.device.lighting.fully_charged', 'setFullyChargedWave', in_sig='i')
def set_fully_charged_wave(self, direction):
    if False:
        for i in range(10):
            print('nop')
    '\n    Set the wave effect on the device\n    :param direction: (0|1) - down to up, (1|2) up to down\n    :type direction: int\n    '
    self.logger.debug('DBus call set_fully_charged_wave')
    self.send_effect_event('setWave', direction)
    self.set_persistence('fully_charged', 'effect', 'wave')
    self.set_persistence('fully_charged', 'wave_dir', int(direction))
    driver_path = self.get_driver_path('fully_charged_matrix_effect_wave')
    if direction not in self.WAVE_DIRS:
        direction = self.WAVE_DIRS[0]
    with open(driver_path, 'w') as driver_file:
        driver_file.write(str(direction))

@endpoint('razer.device.lighting.fully_charged', 'setFullyChargedStatic', in_sig='yyy')
def set_fully_charged_static(self, red, green, blue):
    if False:
        while True:
            i = 10
    '\n    Set the device to static colour\n\n    :param red: Red component\n    :type red: int\n\n    :param green: Green component\n    :type green: int\n\n    :param blue: Blue component\n    :type blue: int\n    '
    self.logger.debug('DBus call set_fully_charged_static')
    self.send_effect_event('setStatic', red, green, blue)
    self.set_persistence('fully_charged', 'effect', 'static')
    self.zone['fully_charged']['colors'][0:3] = (int(red), int(green), int(blue))
    rgb_driver_path = self.get_driver_path('fully_charged_matrix_effect_static')
    payload = bytes([red, green, blue])
    with open(rgb_driver_path, 'wb') as rgb_driver_file:
        rgb_driver_file.write(payload)

@endpoint('razer.device.lighting.fully_charged', 'setFullyChargedSpectrum')
def set_fully_charged_spectrum(self):
    if False:
        i = 10
        return i + 15
    '\n    Set the device to spectrum mode\n    '
    self.logger.debug('DBus call set_fully_charged_spectrum')
    self.send_effect_event('setSpectrum')
    self.set_persistence('fully_charged', 'effect', 'spectrum')
    effect_driver_path = self.get_driver_path('fully_charged_matrix_effect_spectrum')
    with open(effect_driver_path, 'w') as effect_driver_file:
        effect_driver_file.write('1')

@endpoint('razer.device.lighting.fully_charged', 'setFullyChargedNone')
def set_fully_charged_none(self):
    if False:
        i = 10
        return i + 15
    '\n    Set the device to effect none\n    '
    self.logger.debug('DBus call set_fully_charged_none')
    self.send_effect_event('setNone')
    self.set_persistence('fully_charged', 'effect', 'none')
    driver_path = self.get_driver_path('fully_charged_matrix_effect_none')
    with open(driver_path, 'w') as driver_file:
        driver_file.write('1')

@endpoint('razer.device.lighting.fully_charged', 'setFullyChargedBreathRandom')
def set_fully_charged_breath_random(self):
    if False:
        for i in range(10):
            print('nop')
    '\n    Set the device to random colour breathing effect\n    '
    self.logger.debug('DBus call set_fully_charged_breath_random')
    self.send_effect_event('setBreathRandom')
    self.set_persistence('fully_charged', 'effect', 'breathRandom')
    driver_path = self.get_driver_path('fully_charged_matrix_effect_breath')
    payload = b'1'
    with open(driver_path, 'wb') as driver_file:
        driver_file.write(payload)

@endpoint('razer.device.lighting.fully_charged', 'setFullyChargedBreathSingle', in_sig='yyy')
def set_fully_charged_breath_single(self, red, green, blue):
    if False:
        for i in range(10):
            print('nop')
    '\n    Set the device to single colour breathing effect\n\n    :param red: Red component\n    :type red: int\n\n    :param green: Green component\n    :type green: int\n\n    :param blue: Blue component\n    :type blue: int\n    '
    self.logger.debug('DBus call set_fully_charged_breath_single')
    self.send_effect_event('setBreathSingle', red, green, blue)
    self.set_persistence('fully_charged', 'effect', 'breathSingle')
    self.zone['fully_charged']['colors'][0:3] = (int(red), int(green), int(blue))
    driver_path = self.get_driver_path('fully_charged_matrix_effect_breath')
    payload = bytes([red, green, blue])
    with open(driver_path, 'wb') as driver_file:
        driver_file.write(payload)

@endpoint('razer.device.lighting.fully_charged', 'setFullyChargedBreathDual', in_sig='yyyyyy')
def set_fully_charged_breath_dual(self, red1, green1, blue1, red2, green2, blue2):
    if False:
        return 10
    '\n    Set the device to dual colour breathing effect\n\n    :param red1: Red component\n    :type red1: int\n\n    :param green1: Green component\n    :type green1: int\n\n    :param blue1: Blue component\n    :type blue1: int\n\n    :param red2: Red component\n    :type red2: int\n\n    :param green2: Green component\n    :type green2: int\n\n    :param blue2: Blue component\n    :type blue2: int\n    '
    self.logger.debug('DBus call set_fully_charged_breath_dual')
    self.send_effect_event('setBreathDual', red1, green1, blue1, red2, green2, blue2)
    self.set_persistence('fully_charged', 'effect', 'breathDual')
    self.zone['fully_charged']['colors'][0:6] = (int(red1), int(green1), int(blue1), int(red2), int(green2), int(blue2))
    driver_path = self.get_driver_path('fully_charged_matrix_effect_breath')
    payload = bytes([red1, green1, blue1, red2, green2, blue2])
    with open(driver_path, 'wb') as driver_file:
        driver_file.write(payload)