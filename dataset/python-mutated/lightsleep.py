import alarm
import board
import time
import digitalio
import neopixel
wake_pin = board.X1
led_pin = board.LED
led = digitalio.DigitalInOut(led_pin)
led.direction = digitalio.Direction.OUTPUT

def blink(num_blinks):
    if False:
        for i in range(10):
            print('nop')
    for i in range(num_blinks):
        led.value = True
        time.sleep(0.2)
        led.value = False
        time.sleep(0.2)

def show_timealarm():
    if False:
        print('Hello World!')
    blink(2)

def show_pinalarm():
    if False:
        print('Hello World!')
    blink(1)

def show_noalarm():
    if False:
        i = 10
        return i + 15
    blink(3)
pin_alarm = alarm.pin.PinAlarm(pin=wake_pin, value=False, edge=True, pull=True)
while True:
    time_alarm = alarm.time.TimeAlarm(monotonic_time=time.monotonic() + 10)
    ret_alarm = alarm.light_sleep_until_alarms(pin_alarm, time_alarm)
    print('Returned alarm vs global alarm:')
    print(ret_alarm)
    print(alarm.wake_alarm)
    if isinstance(ret_alarm, alarm.time.TimeAlarm):
        show_timealarm()
    elif isinstance(ret_alarm, alarm.pin.PinAlarm):
        show_pinalarm()
    else:
        show_noalarm()