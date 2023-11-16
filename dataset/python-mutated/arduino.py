class EnclosureArduino:
    """
    Listens to enclosure commands for Mycroft's Arduino.

    Performs the associated command on Arduino by writing on the Serial port.
    """

    def __init__(self, bus, writer):
        if False:
            while True:
                i = 10
        self.bus = bus
        self.writer = writer
        self.__init_events()

    def __init_events(self):
        if False:
            return 10
        self.bus.on('enclosure.system.reset', self.reset)
        self.bus.on('enclosure.system.mute', self.mute)
        self.bus.on('enclosure.system.unmute', self.unmute)
        self.bus.on('enclosure.system.blink', self.blink)

    def reset(self, event=None):
        if False:
            return 10
        self.writer.write('system.reset')

    def mute(self, event=None):
        if False:
            while True:
                i = 10
        self.writer.write('system.mute')

    def unmute(self, event=None):
        if False:
            for i in range(10):
                print('nop')
        self.writer.write('system.unmute')

    def blink(self, event=None):
        if False:
            for i in range(10):
                print('nop')
        times = 1
        if event and event.data:
            times = event.data.get('times', times)
        self.writer.write('system.blink=' + str(times))