import time
from PIL import Image

class EnclosureMouth:
    """
    Listens to enclosure commands for Mycroft's Mouth.

    Performs the associated command on Arduino by writing on the Serial port.
    """

    def __init__(self, bus, writer):
        if False:
            print('Hello World!')
        self.bus = bus
        self.writer = writer
        self.is_timer_on = False
        self.__init_events()
        self.showing_visemes = False

    def __init_events(self):
        if False:
            return 10
        self.bus.on('enclosure.mouth.reset', self.reset)
        self.bus.on('enclosure.mouth.talk', self.talk)
        self.bus.on('enclosure.mouth.think', self.think)
        self.bus.on('enclosure.mouth.listen', self.listen)
        self.bus.on('enclosure.mouth.smile', self.smile)
        self.bus.on('enclosure.mouth.viseme_list', self.viseme_list)
        self.bus.on('enclosure.mouth.text', self.text)
        self.bus.on('enclosure.mouth.display', self.display)
        self.bus.on('enclosure.mouth.display_image', self.display_image)
        self.bus.on('enclosure.weather.display', self.display_weather)
        self.bus.on('mycroft.stop', self.clear_visemes)
        self.bus.on('enclosure.mouth.events.activate', self._activate_visemes)
        self.bus.on('enclosure.mouth.events.deactivate', self._deactivate_visemes)

    def _activate_visemes(self, event=None):
        if False:
            return 10
        self.bus.on('enclosure.mouth.viseme_list', self.viseme_list)

    def _deactivate_visemes(self, event=None):
        if False:
            i = 10
            return i + 15
        self.bus.remove('enclosure.mouth.viseme_list', self.viseme_list)

    def reset(self, event=None):
        if False:
            i = 10
            return i + 15
        self.writer.write('mouth.reset')

    def talk(self, event=None):
        if False:
            for i in range(10):
                print('nop')
        self.writer.write('mouth.talk')

    def think(self, event=None):
        if False:
            return 10
        self.writer.write('mouth.think')

    def listen(self, event=None):
        if False:
            print('Hello World!')
        self.writer.write('mouth.listen')

    def smile(self, event=None):
        if False:
            for i in range(10):
                print('nop')
        self.writer.write('mouth.smile')

    def viseme_list(self, event=None):
        if False:
            return 10
        if event and event.data:
            start = event.data['start']
            visemes = event.data['visemes']
            self.showing_visemes = True
            for (code, end) in visemes:
                if not self.showing_visemes:
                    break
                if time.time() < start + end:
                    self.writer.write('mouth.viseme=' + code)
                    time.sleep(start + end - time.time())
            self.reset()

    def clear_visemes(self, event=None):
        if False:
            i = 10
            return i + 15
        self.showing_visemes = False

    def text(self, event=None):
        if False:
            while True:
                i = 10
        text = ''
        if event and event.data:
            text = event.data.get('text', text)
        self.writer.write('mouth.text=' + text)

    def __display(self, code, clear_previous, x_offset, y_offset):
        if False:
            return 10
        ' Write the encoded image to enclosure screen.\n\n        Args:\n            code (str):           encoded image to display\n            clean_previous (str): if "True" will clear the screen before\n                                  drawing.\n            x_offset (int):       x direction offset\n            y_offset (int):       y direction offset\n        '
        clear_previous = int(str(clear_previous) == 'True')
        clear_previous = 'cP=' + str(clear_previous) + ','
        x_offset = 'x=' + str(x_offset) + ','
        y_offset = 'y=' + str(y_offset) + ','
        message = 'mouth.icon=' + x_offset + y_offset + clear_previous + code
        if len(message) > 60:
            message1 = message[:31] + '$'
            message2 = 'mouth.icon=$' + message[31:]
            self.writer.write(message1)
            time.sleep(0.25)
            self.writer.write(message2)
        else:
            time.sleep(0.1)
            self.writer.write(message)

    def display(self, event=None):
        if False:
            for i in range(10):
                print('nop')
        ' Display a Mark-1 specific code.\n        Args:\n            event (Message): messagebus message with data to display\n        '
        code = ''
        x_offset = ''
        y_offset = ''
        clear_previous = ''
        if event and event.data:
            code = event.data.get('img_code', code)
            x_offset = int(event.data.get('xOffset', x_offset))
            y_offset = int(event.data.get('yOffset', y_offset))
            clear_previous = event.data.get('clearPrev', clear_previous)
            self.__display(code, clear_previous, x_offset, y_offset)

    def display_image(self, event=None):
        if False:
            print('Hello World!')
        ' Display an image on the enclosure.\n\n        The method uses PIL to convert the image supplied into a code\n        suitable for the Mark-1 display.\n\n        Args:\n            event (Message): messagebus message with data to display\n        '
        if not event:
            return
        image_absolute_path = event.data['img_path']
        refresh = event.data['clearPrev']
        invert = event.data['invert']
        x_offset = event.data['xOffset']
        y_offset = event.data['yOffset']
        threshold = event.data.get('threshold', 70)
        img = Image.open(image_absolute_path).convert('RGBA')
        img2 = Image.new('RGBA', img.size, (255, 255, 255))
        width = img.size[0]
        height = img.size[1]
        img = Image.alpha_composite(img2, img)
        img = img.convert('L')
        if width > 32:
            img = img.crop((0, 0, 32, height))
            width = img.size[0]
            height = img.size[1]
        if height > 8:
            img = img.crop((0, 0, width, 8))
            width = img.size[0]
            height = img.size[1]
        encode = ''
        width_codes = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a']
        height_codes = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
        encode += width_codes[width - 1]
        encode += height_codes[height - 1]
        binary_values = []
        for i in range(width):
            for j in range(height):
                if img.getpixel((i, j)) < threshold:
                    if invert is False:
                        binary_values.append('1')
                    else:
                        binary_values.append('0')
                elif invert is False:
                    binary_values.append('0')
                else:
                    binary_values.append('1')
        number_of_top_pixel = 0
        number_of_bottom_pixel = 0
        if height > 4:
            number_of_top_pixel = 4
            number_of_bottom_pixel = height - 4
        else:
            number_of_top_pixel = height
        binary_list = []
        binary_code = ''
        increment = 0
        alternate = False
        for val in binary_values:
            binary_code += val
            increment += 1
            if increment == number_of_top_pixel and alternate is False:
                binary_list.append(binary_code[::-1])
                increment = 0
                binary_code = ''
                alternate = True
            elif increment == number_of_bottom_pixel and alternate is True:
                binary_list.append(binary_code[::-1])
                increment = 0
                binary_code = ''
                alternate = False
        pixel_codes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
        for binary_values in binary_list:
            number = int(binary_values, 2)
            pixel_code = pixel_codes[number]
            encode += pixel_code
        self.__display(encode, refresh, x_offset, y_offset)

    def display_weather(self, event=None):
        if False:
            for i in range(10):
                print('nop')
        if event and event.data:
            img_code = event.data.get('img_code', None)
            icon = None
            if img_code == 0:
                icon = 'IICEIBMDNLMDIBCEAA'
            elif img_code == 1:
                icon = 'IIEEGBGDHLHDHBGEEA'
            elif img_code == 2:
                icon = 'IIIBMDMDODODODMDIB'
            elif img_code == 3:
                icon = 'IIMAOJOFPBPJPFOBMA'
            elif img_code == 4:
                icon = 'IIMIOFOBPFPDPJOFMA'
            elif img_code == 5:
                icon = 'IIAAIIMEODLBJAAAAA'
            elif img_code == 6:
                icon = 'IIJEKCMBPHMBKCJEAA'
            elif img_code == 7:
                icon = 'IIABIBIBIJIJJGJAGA'
            temp = event.data.get('temp', None)
            if icon is not None and temp is not None:
                icon = 'x=2,' + icon
                msg = 'weather.display=' + str(temp) + ',' + str(icon)
                self.writer.write(msg)