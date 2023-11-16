"""Mockup for the numato component interface."""
from numato_gpio import NumatoGpioError

class NumatoModuleMock:
    """Mockup for the numato_gpio module."""
    NumatoGpioError = NumatoGpioError

    def __init__(self):
        if False:
            return 10
        'Initialize the numato_gpio module mockup class.'
        self.devices = {}

    class NumatoDeviceMock:
        """Mockup for the numato_gpio.NumatoUsbGpio class."""

        def __init__(self, device):
            if False:
                while True:
                    i = 10
            'Initialize numato device mockup.'
            self.device = device
            self.callbacks = {}
            self.ports = set()
            self.values = {}

        def setup(self, port, direction):
            if False:
                return 10
            'Mockup for setup.'
            self.ports.add(port)
            self.values[port] = None

        def write(self, port, value):
            if False:
                return 10
            'Mockup for write.'
            self.values[port] = value

        def read(self, port):
            if False:
                return 10
            'Mockup for read.'
            return 1

        def adc_read(self, port):
            if False:
                print('Hello World!')
            'Mockup for adc_read.'
            return 1023

        def add_event_detect(self, port, callback, direction):
            if False:
                i = 10
                return i + 15
            'Mockup for add_event_detect.'
            self.callbacks[port] = callback

        def notify(self, enable):
            if False:
                print('Hello World!')
            'Mockup for notify.'

        def mockup_inject_notification(self, port, value):
            if False:
                i = 10
                return i + 15
            'Make the mockup execute a notification callback.'
            self.callbacks[port](port, value)
    OUT = 0
    IN = 1
    RISING = 1
    FALLING = 2
    BOTH = 3

    def discover(self, _=None):
        if False:
            return 10
        'Mockup for the numato device discovery.\n\n        Ignore the device list argument, mock discovers /dev/ttyACM0.\n        '
        self.devices[0] = NumatoModuleMock.NumatoDeviceMock('/dev/ttyACM0')

    def cleanup(self):
        if False:
            while True:
                i = 10
        'Mockup for the numato device cleanup.'
        self.devices.clear()