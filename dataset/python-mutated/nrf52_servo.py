import time
from machine import PWM, Pin

class Servo:

    def __init__(self, pin_name=''):
        if False:
            return 10
        if pin_name:
            self.pin = Pin(pin_name, mode=Pin.OUT, pull=Pin.PULL_DOWN)
        else:
            self.pin = Pin('P22', mode=Pin.OUT, pull=Pin.PULL_DOWN)

    def left(self):
        if False:
            return 10
        p = PWM(0, self.pin, freq=PWM.FREQ_125KHZ, pulse_width=105, period=2500, mode=PWM.MODE_HIGH_LOW)
        p.init()
        time.sleep_ms(200)
        p.deinit()

    def center(self):
        if False:
            i = 10
            return i + 15
        p = PWM(0, self.pin, freq=PWM.FREQ_125KHZ, pulse_width=188, period=2500, mode=PWM.MODE_HIGH_LOW)
        p.init()
        time.sleep_ms(200)
        p.deinit()

    def right(self):
        if False:
            return 10
        p = PWM(0, self.pin, freq=PWM.FREQ_125KHZ, pulse_width=275, period=2500, mode=PWM.MODE_HIGH_LOW)
        p.init()
        time.sleep_ms(200)
        p.deinit()