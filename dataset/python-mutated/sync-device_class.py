import re
from homeassistant.components.binary_sensor import BinarySensorDeviceClass
from homeassistant.components.button import ButtonDeviceClass
from homeassistant.components.cover import CoverDeviceClass
from homeassistant.components.number import NumberDeviceClass
from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.components.switch import SwitchDeviceClass
BLOCKLIST = ('enum',)
DOMAINS = {'binary_sensor': BinarySensorDeviceClass, 'button': ButtonDeviceClass, 'cover': CoverDeviceClass, 'number': NumberDeviceClass, 'sensor': SensorDeviceClass, 'switch': SwitchDeviceClass}

def sub(path, pattern, repl):
    if False:
        i = 10
        return i + 15
    with open(path) as handle:
        content = handle.read()
    content = re.sub(pattern, repl, content, flags=re.MULTILINE)
    with open(path, 'w') as handle:
        handle.write(content)

def main():
    if False:
        for i in range(10):
            print('nop')
    classes = {'EMPTY': ''}
    allowed = {}
    for (domain, enum) in DOMAINS.items():
        available = {cls.value.upper(): cls.value for cls in enum if cls.value not in BLOCKLIST}
        classes.update(available)
        allowed[domain] = list(available.keys()) + ['EMPTY']
    out = ''
    for cls in sorted(classes):
        out += f'DEVICE_CLASS_{cls.upper()} = "{classes[cls]}"\n'
    sub('esphome/const.py', '(DEVICE_CLASS_\\w+ = "\\w*"\r?\n)+', out)
    for domain in sorted(allowed):
        out = ''
        for item in sorted(allowed[domain]):
            out += f'    DEVICE_CLASS_{item.upper()},\n'
        sub(f'esphome/components/{domain}/__init__.py', '(    DEVICE_CLASS_\\w+,\r?\n)+', out)
if __name__ == '__main__':
    main()