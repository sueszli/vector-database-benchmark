from micropython import const
from machine import Pin, ADC
import time
VBUS_SENSE = const(33)
VBAT_SENSE = const(10)
RGB_DATA = const(18)
RGB_PWR = const(17)
SPI_MOSI = const(35)
SPI_MISO = const(37)
SPI_CLK = const(36)
I2C_SDA = const(8)
I2C_SCL = const(9)

def set_pixel_power(state):
    if False:
        for i in range(10):
            print('nop')
    'Enable or Disable power to the onboard NeoPixel to either show colour, or to reduce power for deep sleep.'
    Pin(RGB_PWR, Pin.OUT).value(state)

def get_battery_voltage():
    if False:
        i = 10
        return i + 15
    '\n    Returns the current battery voltage. If no battery is connected, returns 4.2V which is the charge voltage\n    This is an approximation only, but useful to detect if the charge state of the battery is getting low.\n    '
    adc = ADC(Pin(VBAT_SENSE))
    adc.atten(ADC.ATTN_2_5DB)
    measuredvbat = adc.read()
    measuredvbat /= 3657
    measuredvbat *= 4.2
    return round(measuredvbat, 2)

def get_vbus_present():
    if False:
        return 10
    'Detect if VBUS (5V) power source is present'
    return Pin(VBUS_SENSE, Pin.IN).value() == 1

def rgb_color_wheel(wheel_pos):
    if False:
        while True:
            i = 10
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