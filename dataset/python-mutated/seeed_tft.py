"""
MicroPython Seeedstudio TFT Shield V2 driver, SPI interfaces, Analog GPIO
Contains SD-card reader, LCD and Touch sensor

The pca10040 pin layout is used as reference.

Example usage of LCD:

    from seeedstudio_tft_shield_v2 import ILI9341

    lcd = ILI9341(240, 320)
    lcd.text("Hello World!, 32, 32)
    lcd.show()

Example usage of SD card reader:

    import os
    from seeedstudio_tft_shield_v2 import mount_tf

    tf = mount_tf()
    os.listdir()
"""
import os
import time
import framebuf
from machine import SPI, Pin
from sdcard import SDCard

def mount_tf(self, mount_point='/'):
    if False:
        print('Hello World!')
    sd = SDCard(SPI(0), Pin('P15', mode=Pin.OUT))
    os.mount(sd, mount_point)

class ILI9341:

    def __init__(self, width, height):
        if False:
            return 10
        self.width = width
        self.height = height
        self.pages = self.height // 8
        self.buffer = bytearray(self.pages * self.width)
        self.framebuf = framebuf.FrameBuffer(self.buffer, self.width, self.height, framebuf.MONO_VLSB)
        self.spi = SPI(0)
        self.cs = Pin('P16', mode=Pin.OUT, pull=Pin.PULL_UP)
        self.dc = Pin('P17', mode=Pin.OUT, pull=Pin.PULL_UP)
        self.cs.high()
        self.dc.high()
        self.spi.init(baudrate=8000000, phase=0, polarity=0)
        self.init_display()

    def init_display(self):
        if False:
            return 10
        time.sleep_ms(500)
        self.write_cmd(1)
        time.sleep_ms(200)
        self.write_cmd(207)
        self.write_data(bytearray([0, 139, 48]))
        self.write_cmd(237)
        self.write_data(bytearray([103, 3, 18, 129]))
        self.write_cmd(232)
        self.write_data(bytearray([133, 16, 122]))
        self.write_cmd(203)
        self.write_data(bytearray([57, 44, 0, 52, 2]))
        self.write_cmd(247)
        self.write_data(bytearray([32]))
        self.write_cmd(234)
        self.write_data(bytearray([0, 0]))
        self.write_cmd(192)
        self.write_data(bytearray([27]))
        self.write_cmd(193)
        self.write_data(bytearray([16]))
        self.write_cmd(197)
        self.write_data(bytearray([63, 60]))
        self.write_cmd(199)
        self.write_data(bytearray([183]))
        self.write_cmd(54)
        self.write_data(bytearray([8]))
        self.write_cmd(58)
        self.write_data(bytearray([85]))
        self.write_cmd(177)
        self.write_data(bytearray([0, 27]))
        self.write_cmd(182)
        self.write_data(bytearray([10, 162]))
        self.write_cmd(242)
        self.write_data(bytearray([0]))
        self.write_cmd(38)
        self.write_data(bytearray([1]))
        self.write_cmd(224)
        self.write_data(bytearray([15, 42, 40, 8, 14, 8, 84, 169, 67, 10, 15, 0, 0, 0, 0]))
        self.write_cmd(225)
        self.write_data(bytearray([0, 21, 23, 7, 17, 6, 43, 86, 60, 5, 16, 15, 63, 63, 15]))
        self.write_cmd(17)
        time.sleep_ms(120)
        self.write_cmd(41)
        time.sleep_ms(500)
        self.fill(0)

    def show(self):
        if False:
            return 10
        self.write_cmd(42)
        self.write_data(bytearray([0, 0]))
        self.write_data(bytearray([0, 239]))
        self.write_cmd(43)
        self.write_data(bytearray([0, 0]))
        self.write_data(bytearray([1, 63]))
        self.write_cmd(44)
        for row in range(0, self.pages):
            for pixel_pos in range(0, 8):
                for col in range(0, self.width):
                    compressed_pixel = self.buffer[row * 240 + col]
                    if compressed_pixel >> pixel_pos & 1 == 0:
                        self.write_data(bytearray([0, 0]))
                    else:
                        self.write_data(bytearray([255, 255]))

    def fill(self, col):
        if False:
            return 10
        self.framebuf.fill(col)

    def pixel(self, x, y, col):
        if False:
            print('Hello World!')
        self.framebuf.pixel(x, y, col)

    def scroll(self, dx, dy):
        if False:
            for i in range(10):
                print('nop')
        self.framebuf.scroll(dx, dy)

    def text(self, string, x, y, col=1):
        if False:
            return 10
        self.framebuf.text(string, x, y, col)

    def write_cmd(self, cmd):
        if False:
            for i in range(10):
                print('nop')
        self.dc.low()
        self.cs.low()
        self.spi.write(bytearray([cmd]))
        self.cs.high()

    def write_data(self, buf):
        if False:
            print('Hello World!')
        self.dc.high()
        self.cs.low()
        self.spi.write(buf)
        self.cs.high()