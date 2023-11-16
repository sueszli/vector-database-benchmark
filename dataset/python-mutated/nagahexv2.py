from openrazer_daemon.dbus_services import endpoint

@endpoint('razer.device.lighting.logo', 'setLogoStatic', in_sig='yyy')
def set_logo_static(self, red, green, blue):
    if False:
        i = 10
        return i + 15
    '\n    Set the device to static colour\n\n    :param red: Red component\n    :type red: int\n\n    :param green: Green component\n    :type green: int\n\n    :param blue: Blue component\n    :type blue: int\n    '
    self.logger.debug('DBus call set_static_effect')
    self.send_effect_event('setStatic', red, green, blue)
    self.set_persistence('logo', 'effect', 'static')
    self.zone['logo']['colors'][0:3] = (int(red), int(green), int(blue))
    rgb_driver_path = self.get_driver_path('logo_matrix_effect_static')
    payload = bytes([red, green, blue])
    with open(rgb_driver_path, 'wb') as rgb_driver_file:
        rgb_driver_file.write(payload)

@endpoint('razer.device.lighting.logo', 'setLogoSpectrum')
def set_logo_spectrum(self):
    if False:
        for i in range(10):
            print('nop')
    '\n    Set the device to spectrum mode\n    '
    self.logger.debug('DBus call set_logo_spectrum')
    self.send_effect_event('setSpectrum')
    self.set_persistence('logo', 'effect', 'spectrum')
    effect_driver_path = self.get_driver_path('logo_matrix_effect_spectrum')
    with open(effect_driver_path, 'w') as effect_driver_file:
        effect_driver_file.write('1')

@endpoint('razer.device.lighting.logo', 'setLogoNone')
def set_logo_none(self):
    if False:
        print('Hello World!')
    '\n    Set the device to effect none\n    '
    self.logger.debug('DBus call set_none_effect')
    self.send_effect_event('setNone')
    self.set_persistence('logo', 'effect', 'none')
    driver_path = self.get_driver_path('logo_matrix_effect_none')
    with open(driver_path, 'w') as driver_file:
        driver_file.write('1')

@endpoint('razer.device.lighting.logo', 'setLogoOn')
def set_logo_on(self):
    if False:
        for i in range(10):
            print('nop')
    '\n    Set the device to effect on\n    '
    self.logger.debug('DBus call set_logo_on')
    self.send_effect_event('setOn')
    self.set_persistence('logo', 'effect', 'on')
    driver_path = self.get_driver_path('logo_matrix_effect_on')
    with open(driver_path, 'w') as driver_file:
        driver_file.write('1')

@endpoint('razer.device.lighting.logo', 'setLogoReactive', in_sig='yyyy')
def set_logo_reactive(self, red, green, blue, speed):
    if False:
        for i in range(10):
            print('nop')
    '\n    Set the device to reactive effect\n\n    :param red: Red component\n    :type red: int\n\n    :param green: Green component\n    :type green: int\n\n    :param blue: Blue component\n    :type blue: int\n\n    :param speed: Speed\n    :type speed: int\n    '
    self.logger.debug('DBus call set_reactive_effect')
    driver_path = self.get_driver_path('logo_matrix_effect_reactive')
    self.send_effect_event('setReactive', red, green, blue, speed)
    self.set_persistence('logo', 'effect', 'reactive')
    self.zone['logo']['colors'][0:3] = (int(red), int(green), int(blue))
    self.set_persistence('logo', 'speed', int(speed))
    if speed not in (1, 2, 3, 4):
        speed = 4
    payload = bytes([speed, red, green, blue])
    with open(driver_path, 'wb') as driver_file:
        driver_file.write(payload)

@endpoint('razer.device.lighting.logo', 'setLogoBreathMono')
def set_logo_breath_mono(self):
    if False:
        return 10
    '\n    Set the device to mono colour breathing effect\n    '
    self.logger.debug('DBus call set_logo_breath_mono')
    self.send_effect_event('setBreathMono')
    self.set_persistence('logo', 'effect', 'breathMono')
    driver_path = self.get_driver_path('logo_matrix_effect_breath')
    with open(driver_path, 'wb') as driver_file:
        driver_file.write(b'1')

@endpoint('razer.device.lighting.logo', 'setLogoBreathRandom')
def set_logo_breath_random(self):
    if False:
        return 10
    '\n    Set the device to random colour breathing effect\n    '
    self.logger.debug('DBus call set_breath_random_effect')
    self.send_effect_event('setBreathRandom')
    self.set_persistence('logo', 'effect', 'breathRandom')
    driver_path = self.get_driver_path('logo_matrix_effect_breath')
    payload = b'1'
    with open(driver_path, 'wb') as driver_file:
        driver_file.write(payload)

@endpoint('razer.device.lighting.logo', 'setLogoBreathSingle', in_sig='yyy')
def set_logo_breath_single(self, red, green, blue):
    if False:
        while True:
            i = 10
    '\n    Set the device to single colour breathing effect\n\n    :param red: Red component\n    :type red: int\n\n    :param green: Green component\n    :type green: int\n\n    :param blue: Blue component\n    :type blue: int\n    '
    self.logger.debug('DBus call set_breath_single_effect')
    self.send_effect_event('setBreathSingle', red, green, blue)
    self.set_persistence('logo', 'effect', 'breathSingle')
    self.zone['logo']['colors'][0:3] = (int(red), int(green), int(blue))
    driver_path = self.get_driver_path('logo_matrix_effect_breath')
    payload = bytes([red, green, blue])
    with open(driver_path, 'wb') as driver_file:
        driver_file.write(payload)

@endpoint('razer.device.lighting.logo', 'setLogoBreathDual', in_sig='yyyyyy')
def set_logo_breath_dual(self, red1, green1, blue1, red2, green2, blue2):
    if False:
        while True:
            i = 10
    '\n    Set the device to dual colour breathing effect\n\n    :param red1: Red component\n    :type red1: int\n\n    :param green1: Green component\n    :type green1: int\n\n    :param blue1: Blue component\n    :type blue1: int\n\n    :param red2: Red component\n    :type red2: int\n\n    :param green2: Green component\n    :type green2: int\n\n    :param blue2: Blue component\n    :type blue2: int\n    '
    self.logger.debug('DBus call set_breath_dual_effect')
    self.send_effect_event('setBreathDual', red1, green1, blue1, red2, green2, blue2)
    self.set_persistence('logo', 'effect', 'breathDual')
    self.zone['logo']['colors'][0:6] = (int(red1), int(green1), int(blue1), int(red2), int(green2), int(blue2))
    driver_path = self.get_driver_path('logo_matrix_effect_breath')
    payload = bytes([red1, green1, blue1, red2, green2, blue2])
    with open(driver_path, 'wb') as driver_file:
        driver_file.write(payload)

@endpoint('razer.device.lighting.logo', 'setLogoBlinking', in_sig='yyy')
def set_logo_blinking(self, red, green, blue):
    if False:
        while True:
            i = 10
    '\n    Set the device to blinking mode\n\n    :param red: Red component\n    :type red: int\n\n    :param green: Green component\n    :type green: int\n\n    :param blue: Blue component\n    :type blue: int\n    '
    self.logger.debug('DBus call set_logo_blinking')
    self.send_effect_event('setBlinking', red, green, blue)
    self.set_persistence('logo', 'effect', 'blinking')
    self.zone['logo']['colors'][0:3] = (int(red), int(green), int(blue))
    rgb_driver_path = self.get_driver_path('logo_matrix_effect_blinking')
    payload = bytes([red, green, blue])
    with open(rgb_driver_path, 'wb') as rgb_driver_file:
        rgb_driver_file.write(payload)

@endpoint('razer.device.lighting.scroll', 'setScrollStatic', in_sig='yyy')
def set_scroll_static(self, red, green, blue):
    if False:
        while True:
            i = 10
    '\n    Set the device to static colour\n\n    :param red: Red component\n    :type red: int\n\n    :param green: Green component\n    :type green: int\n\n    :param blue: Blue component\n    :type blue: int\n    '
    self.logger.debug('DBus call set_static_effect')
    self.send_effect_event('setStatic', red, green, blue)
    self.set_persistence('scroll', 'effect', 'static')
    self.zone['scroll']['colors'][0:3] = (int(red), int(green), int(blue))
    rgb_driver_path = self.get_driver_path('scroll_matrix_effect_static')
    payload = bytes([red, green, blue])
    with open(rgb_driver_path, 'wb') as rgb_driver_file:
        rgb_driver_file.write(payload)

@endpoint('razer.device.lighting.scroll', 'setScrollSpectrum')
def set_scroll_spectrum(self):
    if False:
        for i in range(10):
            print('nop')
    '\n    Set the device to spectrum mode\n    '
    self.logger.debug('DBus call set_scroll_spectrum')
    self.send_effect_event('setSpectrum')
    self.set_persistence('scroll', 'effect', 'spectrum')
    effect_driver_path = self.get_driver_path('scroll_matrix_effect_spectrum')
    with open(effect_driver_path, 'w') as effect_driver_file:
        effect_driver_file.write('1')

@endpoint('razer.device.lighting.scroll', 'setScrollNone')
def set_scroll_none(self):
    if False:
        print('Hello World!')
    '\n    Set the device to effect none\n    '
    self.logger.debug('DBus call set_none_effect')
    self.send_effect_event('setNone')
    self.set_persistence('scroll', 'effect', 'none')
    driver_path = self.get_driver_path('scroll_matrix_effect_none')
    with open(driver_path, 'w') as driver_file:
        driver_file.write('1')

@endpoint('razer.device.lighting.scroll', 'setScrollOn')
def set_scroll_on(self):
    if False:
        print('Hello World!')
    '\n    Set the device to effect on\n    '
    self.logger.debug('DBus call set_scroll_on')
    self.send_effect_event('setOn')
    self.set_persistence('scroll', 'effect', 'on')
    driver_path = self.get_driver_path('scroll_matrix_effect_on')
    with open(driver_path, 'w') as driver_file:
        driver_file.write('1')

@endpoint('razer.device.lighting.scroll', 'setScrollReactive', in_sig='yyyy')
def set_scroll_reactive(self, red, green, blue, speed):
    if False:
        for i in range(10):
            print('nop')
    '\n    Set the device to reactive effect\n\n    :param red: Red component\n    :type red: int\n\n    :param green: Green component\n    :type green: int\n\n    :param blue: Blue component\n    :type blue: int\n\n    :param speed: Speed\n    :type speed: int\n    '
    self.logger.debug('DBus call set_reactive_effect')
    driver_path = self.get_driver_path('scroll_matrix_effect_reactive')
    self.send_effect_event('setReactive', red, green, blue, speed)
    self.set_persistence('scroll', 'effect', 'reactive')
    self.zone['scroll']['colors'][0:3] = (int(red), int(green), int(blue))
    self.set_persistence('scroll', 'speed', int(speed))
    if speed not in (1, 2, 3, 4):
        speed = 4
    payload = bytes([speed, red, green, blue])
    with open(driver_path, 'wb') as driver_file:
        driver_file.write(payload)

@endpoint('razer.device.lighting.scroll', 'setScrollBreathMono')
def set_scroll_breath_mono(self):
    if False:
        while True:
            i = 10
    '\n    Set the device to mono colour breathing effect\n    '
    self.logger.debug('DBus call set_scroll_breath_mono')
    self.send_effect_event('setBreathMono')
    self.set_persistence('scroll', 'effect', 'breathMono')
    driver_path = self.get_driver_path('scroll_matrix_effect_breath')
    with open(driver_path, 'wb') as driver_file:
        driver_file.write(b'1')

@endpoint('razer.device.lighting.scroll', 'setScrollBreathRandom')
def set_scroll_breath_random(self):
    if False:
        print('Hello World!')
    '\n    Set the device to random colour breathing effect\n    '
    self.logger.debug('DBus call set_breath_random_effect')
    self.send_effect_event('setBreathRandom')
    self.set_persistence('scroll', 'effect', 'breathRandom')
    driver_path = self.get_driver_path('scroll_matrix_effect_breath')
    payload = b'1'
    with open(driver_path, 'wb') as driver_file:
        driver_file.write(payload)

@endpoint('razer.device.lighting.scroll', 'setScrollBreathSingle', in_sig='yyy')
def set_scroll_breath_single(self, red, green, blue):
    if False:
        print('Hello World!')
    '\n    Set the device to single colour breathing effect\n\n    :param red: Red component\n    :type red: int\n\n    :param green: Green component\n    :type green: int\n\n    :param blue: Blue component\n    :type blue: int\n    '
    self.logger.debug('DBus call set_breath_single_effect')
    self.send_effect_event('setBreathSingle', red, green, blue)
    self.set_persistence('scroll', 'effect', 'breathSingle')
    self.zone['scroll']['colors'][0:3] = (int(red), int(green), int(blue))
    driver_path = self.get_driver_path('scroll_matrix_effect_breath')
    payload = bytes([red, green, blue])
    with open(driver_path, 'wb') as driver_file:
        driver_file.write(payload)

@endpoint('razer.device.lighting.scroll', 'setScrollBreathDual', in_sig='yyyyyy')
def set_scroll_breath_dual(self, red1, green1, blue1, red2, green2, blue2):
    if False:
        return 10
    '\n    Set the device to dual colour breathing effect\n\n    :param red1: Red component\n    :type red1: int\n\n    :param green1: Green component\n    :type green1: int\n\n    :param blue1: Blue component\n    :type blue1: int\n\n    :param red2: Red component\n    :type red2: int\n\n    :param green2: Green component\n    :type green2: int\n\n    :param blue2: Blue component\n    :type blue2: int\n    '
    self.logger.debug('DBus call set_breath_dual_effect')
    self.send_effect_event('setBreathDual', red1, green1, blue1, red2, green2, blue2)
    self.set_persistence('scroll', 'effect', 'breathDual')
    self.zone['scroll']['colors'][0:6] = (int(red1), int(green1), int(blue1), int(red2), int(green2), int(blue2))
    driver_path = self.get_driver_path('scroll_matrix_effect_breath')
    payload = bytes([red1, green1, blue1, red2, green2, blue2])
    with open(driver_path, 'wb') as driver_file:
        driver_file.write(payload)

@endpoint('razer.device.lighting.scroll', 'setScrollBlinking', in_sig='yyy')
def set_scroll_blinking(self, red, green, blue):
    if False:
        print('Hello World!')
    '\n    Set the device to blinking mode\n\n    :param red: Red component\n    :type red: int\n\n    :param green: Green component\n    :type green: int\n\n    :param blue: Blue component\n    :type blue: int\n    '
    self.logger.debug('DBus call set_scroll_blinking')
    self.send_effect_event('setBlinking', red, green, blue)
    self.set_persistence('scroll', 'effect', 'blinking')
    self.zone['scroll']['colors'][0:3] = (int(red), int(green), int(blue))
    rgb_driver_path = self.get_driver_path('scroll_matrix_effect_blinking')
    payload = bytes([red, green, blue])
    with open(rgb_driver_path, 'wb') as rgb_driver_file:
        rgb_driver_file.write(payload)