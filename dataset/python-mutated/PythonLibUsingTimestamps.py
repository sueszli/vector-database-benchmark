import time

def timezone_correction():
    if False:
        for i in range(10):
            print('nop')
    dst = 3600 if time.daylight == 0 else 0
    tz = 7200 + time.timezone
    return (tz + dst) * 1000

def timestamp_as_integer():
    if False:
        return 10
    t = 1308419034931 + timezone_correction()
    print('*INFO:%d* Known timestamp' % t)
    print('*HTML:%d* <b>Current</b>' % int(time.time() * 1000))
    time.sleep(0.1)

def timestamp_as_float():
    if False:
        for i in range(10):
            print('nop')
    t = 1308419034930.5024 + timezone_correction()
    print('*INFO:%f* Known timestamp' % t)
    print('*HTML:%f* <b>Current</b>' % float(time.time() * 1000))
    time.sleep(0.1)