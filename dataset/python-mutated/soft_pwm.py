import time
from hwconfig import LED

def pwm_cycle(led, duty, cycles):
    if False:
        for i in range(10):
            print('nop')
    duty_off = 20 - duty
    for i in range(cycles):
        if duty:
            led.on()
            time.sleep_ms(duty)
        if duty_off:
            led.off()
            time.sleep_ms(duty_off)
while True:
    for i in range(1, 21):
        pwm_cycle(LED, i, 2)
    for i in range(20, 0, -1):
        pwm_cycle(LED, i, 2)