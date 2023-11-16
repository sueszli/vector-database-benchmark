import colorsys
import random
from openrazer.client import DeviceManager
from openrazer.client import constants as razer_constants
device_manager = DeviceManager()
print('Found {} Razer devices'.format(len(device_manager.devices)))
devices = device_manager.devices
for device in list(devices):
    if not device.fx.advanced:
        print('Skipping device ' + device.name + ' (' + device.serial + ')')
        devices.remove(device)
print()
device_manager.sync_effects = False

def random_color():
    if False:
        for i in range(10):
            print('nop')
    rgb = colorsys.hsv_to_rgb(random.uniform(0, 1), random.uniform(0.5, 1), 1)
    return tuple(map(lambda x: int(256 * x), rgb))
for device in devices:
    print('Drawing to device ' + device.name + ' (' + device.serial + ')')
    (rows, cols) = (device.fx.advanced.rows, device.fx.advanced.cols)
    for row in range(rows):
        for col in range(cols):
            device.fx.advanced.matrix[row, col] = random_color()
    device.fx.advanced.draw()