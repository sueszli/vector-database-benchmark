"""LILYGO TTGO LoRa32 MicroPython Helper Library."""
from machine import Pin, SoftI2C, Signal
from lilygo_oled import OLED
from micropython import const

class Lora32Base:
    """Base class defining common pins."""

    def __init__(self, define_helpers=True):
        if False:
            print('Hello World!')
        self.LORA_MOSI = const(27)
        self.LORA_MISO = const(19)
        self.LORA_SCLK = const(5)
        self.LORA_CS = const(18)
        self.LORA_DIO = const(26)
        self.LORA_RST = const(23)
        self.DAC1 = const(26)
        self.LED = const(25)
        self.OLED_SDA = const(21)
        self.OLED_SCL = const(22)
        if define_helpers:
            self.create_helpers()

    def create_helpers(self):
        if False:
            print('Hello World!')
        self.led = Pin(self.LED, Pin.OUT)
        self.i2c = SoftI2C(scl=Pin(self.OLED_SCL), sda=Pin(self.OLED_SDA))
        self.oled = OLED(self.i2c)

class Lora32v1_0(Lora32Base):
    """Device Support for LILYGO TTGO LoRa32 v1.0."""

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__(define_helpers=False)
        self.LORA_RST = const(14)
        self.OLED_SDA = const(4)
        self.OLED_SCL = const(15)
        self.OLED_RST = const(16)
        super().create_helpers()

class Lora32v1_2(Lora32Base):
    """Device Support for LILYGO TTGO LoRa32 v1.2 (T-Fox)."""

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.DS3231_SDA = const(21)
        self.DS3231_SCL = const(22)

class Lora32(Lora32Base):
    """Device Support for LILYGO TTGO LoRa32 v1.6 and v2.0."""

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.SD_CS = const(13)
        self.SD_MOSI = const(15)
        self.SD_MISO = const(2)
        self.SD_SCLK = const(14)