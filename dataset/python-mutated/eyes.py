class EnclosureEyes:
    """
    Listens to enclosure commands for Mycroft's Eyes.

    Performs the associated command on Arduino by writing on the Serial port.
    """

    def __init__(self, bus, writer):
        if False:
            i = 10
            return i + 15
        self.bus = bus
        self.writer = writer
        self._num_pixels = 12 * 2
        self._current_rgb = [(255, 255, 255) for i in range(self._num_pixels)]
        self.__init_events()

    def __init_events(self):
        if False:
            return 10
        self.bus.on('enclosure.eyes.on', self.on)
        self.bus.on('enclosure.eyes.off', self.off)
        self.bus.on('enclosure.eyes.blink', self.blink)
        self.bus.on('enclosure.eyes.narrow', self.narrow)
        self.bus.on('enclosure.eyes.look', self.look)
        self.bus.on('enclosure.eyes.color', self.color)
        self.bus.on('enclosure.eyes.level', self.brightness)
        self.bus.on('enclosure.eyes.volume', self.volume)
        self.bus.on('enclosure.eyes.spin', self.spin)
        self.bus.on('enclosure.eyes.timedspin', self.timed_spin)
        self.bus.on('enclosure.eyes.reset', self.reset)
        self.bus.on('enclosure.eyes.setpixel', self.set_pixel)
        self.bus.on('enclosure.eyes.fill', self.fill)
        self.bus.on('enclosure.eyes.rgb.get', self.handle_get_color)

    def handle_get_color(self, message):
        if False:
            for i in range(10):
                print('nop')
        'Get the eye RGB color for all pixels\n        Returns:\n           (list) list of (r,g,b) tuples for each eye pixel\n        '
        self.bus.emit(message.reply('enclosure.eyes.rgb', {'pixels': self._current_rgb}))

    def on(self, event=None):
        if False:
            i = 10
            return i + 15
        self.writer.write('eyes.on')

    def off(self, event=None):
        if False:
            while True:
                i = 10
        self.writer.write('eyes.off')

    def blink(self, event=None):
        if False:
            print('Hello World!')
        side = 'b'
        if event and event.data:
            side = event.data.get('side', side)
        self.writer.write('eyes.blink=' + side)

    def narrow(self, event=None):
        if False:
            i = 10
            return i + 15
        self.writer.write('eyes.narrow')

    def look(self, event=None):
        if False:
            print('Hello World!')
        if event and event.data:
            side = event.data.get('side', '')
            self.writer.write('eyes.look=' + side)

    def color(self, event=None):
        if False:
            return 10
        (r, g, b) = (255, 255, 255)
        if event and event.data:
            r = int(event.data.get('r', r))
            g = int(event.data.get('g', g))
            b = int(event.data.get('b', b))
        color = r * 65536 + g * 256 + b
        self._current_rgb = [(r, g, b) for i in range(self._num_pixels)]
        self.writer.write('eyes.color=' + str(color))

    def set_pixel(self, event=None):
        if False:
            while True:
                i = 10
        idx = 0
        (r, g, b) = (255, 255, 255)
        if event and event.data:
            idx = int(event.data.get('idx', idx))
            r = int(event.data.get('r', r))
            g = int(event.data.get('g', g))
            b = int(event.data.get('b', b))
        self._current_rgb[idx] = (r, g, b)
        color = r * 65536 + g * 256 + b
        self.writer.write('eyes.set=' + str(idx) + ',' + str(color))

    def fill(self, event=None):
        if False:
            while True:
                i = 10
        amount = 0
        if event and event.data:
            percent = int(event.data.get('percentage', 0))
            amount = int(round(23.0 * percent / 100.0))
        self.writer.write('eyes.fill=' + str(amount))

    def brightness(self, event=None):
        if False:
            while True:
                i = 10
        level = 30
        if event and event.data:
            level = event.data.get('level', level)
        self.writer.write('eyes.level=' + str(level))

    def volume(self, event=None):
        if False:
            return 10
        volume = 4
        if event and event.data:
            volume = event.data.get('volume', volume)
        self.writer.write('eyes.volume=' + str(volume))

    def reset(self, event=None):
        if False:
            print('Hello World!')
        self.writer.write('eyes.reset')

    def spin(self, event=None):
        if False:
            for i in range(10):
                print('nop')
        self.writer.write('eyes.spin')

    def timed_spin(self, event=None):
        if False:
            print('Hello World!')
        length = 5000
        if event and event.data:
            length = event.data.get('length', length)
        self.writer.write('eyes.spin=' + str(length))