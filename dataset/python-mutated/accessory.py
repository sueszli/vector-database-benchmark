"""
Module for accessory methods
"""
from openrazer_daemon.dbus_services import endpoint

@endpoint('razer.device.misc.mug', 'isMugPresent', out_sig='b')
def is_mug_present(self):
    if False:
        for i in range(10):
            print('nop')
    "\n    Get if the mug is present\n\n    :return: True if there's a mug\n    :rtype: bool\n    "
    self.logger.debug('DBus call is_mug_present')
    driver_path = self.get_driver_path('is_mug_present')
    with open(driver_path, 'r') as driver_file:
        return int(driver_file.read().strip()) == 1