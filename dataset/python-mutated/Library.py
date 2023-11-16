import time

def busy_sleep(seconds):
    if False:
        i = 10
        return i + 15
    max_time = time.time() + int(seconds)
    while time.time() < max_time:
        time.sleep(0)

def swallow_exception(timeout=3):
    if False:
        i = 10
        return i + 15
    try:
        busy_sleep(timeout)
    except:
        pass
    else:
        raise AssertionError('Expected exception did not occur!')