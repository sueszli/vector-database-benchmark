import time
import pwmio

def fade(pin):
    if False:
        i = 10
        return i + 15
    led = pwmio.PWMOut(pin, frequency=5000, duty_cycle=0)
    while True:
        for i in range(100):
            if i < 50:
                led.duty_cycle = int(i * 2 * 65535 / 100)
            else:
                led.duty_cycle = 65535 - int((i - 50) * 2 * 65535 / 100)
            time.sleep(0.01)