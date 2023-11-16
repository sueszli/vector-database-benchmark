from micropython import const
from machine import Pin, ADC
import machine, time
VBUS_SENSE = const(34)
VBAT_SENSE = const(2)
RGB_DATA = const(40)
RGB_PWR = const(39)
RGB_MATRIX_DATA = const(21)
RGB_MATRIX_PWR = const(4)
SPI_MOSI = const(35)
SPI_MISO = const(37)
SPI_CLK = const(36)
I2C_SDA = const(8)
I2C_SCL = const(9)
DAC1 = const(17)
DAC2 = const(18)

def set_pixel_power(state):
    if False:
        return 10
    'Enable or Disable power to the onboard NeoPixel to either show colour, or to reduce power for deep sleep.'
    Pin(RGB_PWR, Pin.OUT).value(state)

def set_pixel_matrix_power(state):
    if False:
        print('Hello World!')
    'Enable or Disable power to the onboard NeoPixel RGB Matrix to either show colours, or to reduce power for deep sleep.'
    Pin(RGB_MATRIX_PWR, Pin.OUT).value(state)

def get_battery_voltage():
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns the current battery voltage. If no battery is connected, returns 4.2V which is the charge voltage\n    This is an approximation only, but useful to detect if the charge state of the battery is getting low.\n    '
    adc = ADC(Pin(VBAT_SENSE))
    measuredvbat = adc.read()
    measuredvbat /= 8192
    measuredvbat *= 4.2
    return round(measuredvbat, 2)

def get_vbus_present():
    if False:
        return 10
    'Detect if VBUS (5V) power source is present'
    return Pin(VBUS_SENSE, Pin.IN).value() == 1

def rgb_color_wheel(wheel_pos):
    if False:
        for i in range(10):
            print('nop')
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

def go_deepsleep(t):
    if False:
        while True:
            i = 10
    'Deep sleep helper that also powers down the on-board NeoPixel.'
    set_pixel_power(False)
    set_pixel_matrix_power(False)
    machine.deepsleep(t)