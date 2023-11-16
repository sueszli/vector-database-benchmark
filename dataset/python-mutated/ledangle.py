import pyb

def led_angle(seconds_to_run_for):
    if False:
        i = 10
        return i + 15
    l1 = pyb.LED(1)
    l2 = pyb.LED(2)
    accel = pyb.Accel()
    for i in range(20 * seconds_to_run_for):
        x = accel.x()
        if x < -10:
            l1.on()
            l2.off()
        elif x > 10:
            l1.off()
            l2.on()
        else:
            l1.off()
            l2.off()
        pyb.delay(50)