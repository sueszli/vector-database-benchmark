from openrazer_daemon.dbus_services import endpoint

@endpoint('razer.device.lighting.channel', 'getNumChannels', out_sig='q')
def get_num_channels(self):
    if False:
        print('Hello World!')
    return self.NUM_CHANNELS

def _get_channel_brightness(self, channel):
    if False:
        for i in range(10):
            print('nop')
    driver_path = self.get_driver_path(channel + '_led_brightness')
    with open(driver_path, 'r') as driver_file:
        return float(driver_file.read().strip()) / (255.0 / 100.0)

@endpoint('razer.device.lighting.channel', 'getChannelBrightness', in_sig='q', out_sig='d')
def get_channel_brightness(self, channel):
    if False:
        for i in range(10):
            print('nop')
    '\n    Get the channel brightness\n\n    :param channel: Channel number to get the brightness of\n    :type channel: int\n\n    :return: Brightness\n    :rtype: float\n    '
    channel_name = 'channel{}'.format(channel)
    self.logger.debug('DBus call get_{}_brightness'.format(channel_name))
    return _get_channel_brightness(self, channel_name)

def _set_channel_brightness(self, channel, brightness):
    if False:
        i = 10
        return i + 15
    driver_path = self.get_driver_path(channel + '_led_brightness')
    self.method_args['brightness'] = brightness
    if brightness > 100:
        brightness = 100
    elif brightness < 0:
        brightness = 0
    self.set_persistence(channel, 'brightness', int(brightness))
    brightness = int(round(brightness * (255.0 / 100.0)))
    with open(driver_path, 'w') as driver_file:
        driver_file.write(str(brightness))
    self.send_effect_event('setBrightness', brightness)

@endpoint('razer.device.lighting.channel', 'setChannelBrightness', in_sig='qd')
def set_channel_brightness(self, channel, brightness):
    if False:
        return 10
    "\n    Set the device's brightness\n\n    :param channel: Channel\n    :type channel: int\n\n    :param brightness: Brightness\n    :type brightness: int\n    "
    channel_name = 'channel{}'.format(channel)
    self.logger.debug('DBus call set_{}_brightness'.format(channel_name))
    _set_channel_brightness(self, channel_name, brightness)

def _get_channel_size(self, channel):
    if False:
        i = 10
        return i + 15
    driver_path = self.get_driver_path(channel + '_size')
    with open(driver_path, 'r') as driver_file:
        return int(driver_file.read().strip())

@endpoint('razer.device.lighting.channel', 'getChannelSize', in_sig='q', out_sig='i')
def get_channel_size(self, channel):
    if False:
        return 10
    "\n    Get the device's size\n\n    :param channel: Channel\n    :type channel: int\n\n    :return: Size\n    :rtype: float\n    "
    channel_name = 'channel{}'.format(channel)
    self.logger.debug('DBus call get_{}_size'.format(channel_name))
    return _get_channel_size(self, channel_name)

def _set_channel_size(self, channel, size):
    if False:
        return 10
    driver_path = self.get_driver_path(channel + '_size')
    self.method_args['size'] = size
    if size > 255:
        size = 255
    elif size < 0:
        size = 0
    self.set_persistence(channel, 'size', int(size))
    with open(driver_path, 'w') as driver_file:
        driver_file.write(str(size))
    self.send_effect_event('setSize', size)

@endpoint('razer.device.lighting.channel', 'setChannelSize', in_sig='qi')
def set_channel_size(self, channel, size):
    if False:
        return 10
    "\n    Set the device's size\n    :param channel: Channel\n    :type channel: int\n\n    :param size: Size\n    :type size: int\n    "
    channel_name = 'channel{}'.format(channel)
    self.logger.debug('DBus call set_{}_size'.format(channel_name))
    _set_channel_size(self, channel_name, size)