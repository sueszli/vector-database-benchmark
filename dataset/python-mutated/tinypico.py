from micropython import const
from machine import Pin, SPI, ADC
import machine, time, esp32
BAT_VOLTAGE = const(35)
BAT_CHARGE = const(34)
DOTSTAR_CLK = const(12)
DOTSTAR_DATA = const(2)
DOTSTAR_PWR = const(13)
SPI_MOSI = const(23)
SPI_CLK = const(18)
SPI_MISO = const(19)
I2C_SDA = const(21)
I2C_SCL = const(22)
DAC1 = const(25)
DAC2 = const(26)

def get_battery_voltage():
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns the current battery voltage. If no battery is connected, returns 3.7V\n    This is an approximation only, but useful to detect of the charge state of the battery is getting low.\n    '
    adc = ADC(Pin(BAT_VOLTAGE))
    measuredvbat = adc.read()
    measuredvbat /= 4095
    measuredvbat *= 3.7
    return measuredvbat

def get_battery_charging():
    if False:
        return 10
    "\n    Returns the current battery charging state.\n    This can trigger false positives as the charge IC can't tell the difference between a full battery or no battery connected.\n    "
    measuredVal = 0
    io = Pin(BAT_CHARGE, Pin.IN)
    for y in range(0, 10):
        measuredVal += io.value()
    return measuredVal == 0

def set_dotstar_power(state):
    if False:
        while True:
            i = 10
    'Set the power for the on-board Dotstar to allow no current draw when not needed.'
    if state:
        Pin(DOTSTAR_PWR, Pin.OUT, None, value=0)
    else:
        Pin(DOTSTAR_PWR, Pin.IN, None)
    Pin(DOTSTAR_CLK, Pin.OUT if state else Pin.IN)
    Pin(DOTSTAR_DATA, Pin.OUT if state else Pin.IN)
    time.sleep(0.035)

def dotstar_color_wheel(wheel_pos):
    if False:
        return 10
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
        i = 10
        return i + 15
    'Deep sleep helper that also powers down the on-board Dotstar.'
    set_dotstar_power(False)
    machine.deepsleep(t)