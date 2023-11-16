from collections import defaultdict
import colorsys
import random
import sys
import time
import threading
from openrazer.client import DeviceManager
from openrazer.client import constants as razer_constants
quit = False
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
        while True:
            i = 10
    rgb = colorsys.hsv_to_rgb(random.uniform(0, 1), random.uniform(0.5, 1), 1)
    return tuple(map(lambda x: int(256 * x), rgb))

def starlight_key(device, row, col, active):
    if False:
        print('Hello World!')
    color = random_color()
    hue = random.uniform(0, 1)
    start_time = time.time()
    fade_time = 2
    elapsed = 0
    while elapsed < fade_time:
        value = 1 - elapsed / fade_time
        rgb = colorsys.hsv_to_rgb(hue, 1, value)
        color = tuple(map(lambda x: int(256 * x), rgb))
        device.fx.advanced.matrix[row, col] = color
        time.sleep(1 / 60)
        elapsed = time.time() - start_time
    device.fx.advanced.matrix[row, col] = (0, 0, 0)
    active[row, col] = False

def starlight_effect(device):
    if False:
        i = 10
        return i + 15
    (rows, cols) = (device.fx.advanced.rows, device.fx.advanced.cols)
    active = defaultdict(bool)
    device.fx.advanced.matrix.reset()
    device.fx.advanced.draw()
    while True:
        (row, col) = (random.randrange(rows), random.randrange(cols))
        if not active[row, col]:
            active[row, col] = True
            threading.Thread(target=starlight_key, args=(device, row, col, active)).start()
        time.sleep(0.1)
        if quit:
            break
    device.fx.advanced.restore()
threads = []
for device in devices:
    print('Starting starlight for device ' + device.name + ' (' + device.serial + ')')
    t = threading.Thread(target=starlight_effect, args=(device,), daemon=True)
    t.start()
    threads.append(t)
try:
    while any((t.is_alive() for t in threads)):
        for device in devices:
            device.fx.advanced.draw()
        time.sleep(1 / 60)
except KeyboardInterrupt:
    quit = True
    for t in threads:
        t.join()
    sys.exit(0)