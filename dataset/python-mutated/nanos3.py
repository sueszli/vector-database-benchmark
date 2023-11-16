from micropython import const
from machine import Pin, ADC
import time
RGB_DATA = const(41)
RGB_PWR = const(42)
SPI_MOSI = const(35)
SPI_MISO = const(37)
SPI_CLK = const(36)
I2C_SDA = const(8)
I2C_SCL = const(9)

def set_pixel_power(state):
    if False:
        return 10
    'Enable or Disable power to the onboard NeoPixel to either show colour, or to reduce power for deep sleep.'
    Pin(RGB_PWR, Pin.OUT).value(state)

def rgb_color_wheel(wheel_pos):
    if False:
        i = 10
        return i + 15
    'Color wheel to allow for cycling through the rainbow of RGB colors.'
    wheel_pos = wheel_pos % 255
    if wheel_pos < 85:
        return (255 - wheel_pos * 3, 0, wheel_pos * 3)
    elif wheel_pos < 170:
        wheel_pos -= 85
        return (0, wheel_pos * 3, 255 - wheel_pos * 3)
    else:
        wheel_pos -= 170
        return (wheel_pos * 3, 255 - wheel_pos * 3, 0)