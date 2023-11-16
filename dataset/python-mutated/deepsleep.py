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
        i = 10
        return i + 15
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
        for i in range(10):
            print('nop')
    blink(1)

def show_noalarm():
    if False:
        i = 10
        return i + 15
    blink(3)
print('Wake alarm:')
print(alarm.wake_alarm)
if isinstance(alarm.wake_alarm, alarm.time.TimeAlarm):
    show_timealarm()
elif isinstance(alarm.wake_alarm, alarm.pin.PinAlarm):
    show_pinalarm()
else:
    show_noalarm()
time_alarm = alarm.time.TimeAlarm(monotonic_time=time.monotonic() + 10)
pin_alarm = alarm.pin.PinAlarm(pin=wake_pin, value=True, edge=True, pull=True)
alarm.exit_and_deep_sleep_until_alarms(time_alarm, pin_alarm)