"""
BlackWidow Ultimate 2013 effects
"""
from openrazer_daemon.dbus_services import endpoint

@endpoint('razer.device.lighting.bw2013', 'setPulsate')
def bw_set_pulsate(self):
    if False:
        for i in range(10):
            print('nop')
    '\n    Set pulsate mode\n    '
    self.logger.debug('DBus call bw_set_pulsate')
    driver_path = self.get_driver_path('matrix_effect_pulsate')
    self.set_persistence('backlight', 'effect', 'pulsate')
    with open(driver_path, 'w') as driver_file:
        driver_file.write('1')
    self.send_effect_event('setPulsate')

@endpoint('razer.device.lighting.bw2013', 'setStatic')
def bw_set_static(self):
    if False:
        print('Hello World!')
    '\n    Set static mode\n    '
    self.logger.debug('DBus call bw_set_static')
    driver_path = self.get_driver_path('matrix_effect_static')
    self.set_persistence('backlight', 'effect', 'static')
    with open(driver_path, 'w') as driver_file:
        driver_file.write('1')
    self.send_effect_event('setStatic')