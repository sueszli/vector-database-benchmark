print('pins test')
'\n`adafruit_boardtest.boardtest_gpio`\n====================================================\nToggles all available GPIO on a board. Verify their operation with an LED,\nmultimeter, another microcontroller, etc.\n\nRun this script as its own main.py to individually run the test, or compile\nwith mpy-cross and call from separate test script.\n\n* Author(s): Shawn Hymel for Adafruit Industries\n\nImplementation Notes\n--------------------\n\n**Software and Dependencies:**\n\n* Adafruit CircuitPython firmware for the supported boards:\n  https://github.com/adafruit/circuitpython/releases\n\n'
import time
import board
import digitalio
import supervisor
__version__ = '0.0.0-auto.0'
__repo__ = 'https://github.com/adafruit/Adafruit_CircuitPython_BoardTest.git'
LED_ON_DELAY_TIME = 0.2
LED_OFF_DELAY_TIME = 0.2
LED_PIN_NAMES = ['L', 'LED', 'RED_LED', 'GREEN_LED', 'BLUE_LED']
PASS = 'PASS'
FAIL = 'FAIL'
NA = 'N/A'

def _is_number(val):
    if False:
        print('Hello World!')
    try:
        float(val)
        return True
    except ValueError:
        return False

def _deinit_pins(gpios):
    if False:
        for i in range(10):
            print('nop')
    for g in gpios:
        g.deinit()

def _toggle_wait(pin_gpios):
    if False:
        while True:
            i = 10
    timestamp = time.monotonic()
    led_state = False
    failed = []
    for pg in pin_gpios:
        (pin, gpio) = pg
        print('Is pin %s toggling? [y/n]' % pin)
        done = False
        while not done:
            if led_state:
                if time.monotonic() > timestamp + LED_ON_DELAY_TIME:
                    led_state = False
                    timestamp = time.monotonic()
            elif time.monotonic() > timestamp + LED_OFF_DELAY_TIME:
                led_state = True
                timestamp = time.monotonic()
            gpio.value = led_state
            if supervisor.runtime.serial_bytes_available:
                answer = input()
                if bool(answer == 'y'):
                    done = True
                elif bool(answer == 'n'):
                    failed += pin
                    done = True
    return failed

def buildPin(pin):
    if False:
        print('Hello World!')
    gpio = digitalio.DigitalInOut(pin)
    return gpio

def run_test(pins):
    if False:
        print('Hello World!')
    '\n    Toggles all available GPIO on and off repeatedly.\n\n    :param list[str] pins: list of pins to run the test on\n    :return: tuple(str, list[str]): test result followed by list of pins tested\n    '
    analog_pins = [p for p in pins if p[0] == 'A' and _is_number(p[1])]
    digital_pins = [p for p in pins if p[0] == 'D' and _is_number(p[1])]
    gpio_pins = analog_pins + digital_pins
    if gpio_pins:
        print('GPIO pins found:', end=' ')
        for pin in gpio_pins:
            print(pin, end=' ')
        print('\n')
        gpios = [buildPin(getattr(board, p)) for p in gpio_pins]
        print('built GPIOs')
        for gpio in gpios:
            gpio.direction = digitalio.Direction.OUTPUT
        result = _toggle_wait(zip(gpio_pins, gpios))
        _deinit_pins(gpios)
        if result:
            return (FAIL, gpio_pins)
        return (PASS, gpio_pins)
    print('No GPIO pins found')
    return (NA, [])
run_test([p for p in dir(board)])