import time
import timer
import win32event
import win32gui

class glork:

    def __init__(self, delay=1000, max=10):
        if False:
            for i in range(10):
                print('nop')
        self.x = 0
        self.max = max
        self.id = timer.set_timer(delay, self.increment)
        self.event = win32event.CreateEvent(None, 0, 0, None)

    def increment(self, id, time):
        if False:
            i = 10
            return i + 15
        print('x = %d' % self.x)
        self.x = self.x + 1
        if self.x > self.max:
            timer.kill_timer(id)
            win32event.SetEvent(self.event)

def demo(delay=1000, stop=10):
    if False:
        return 10
    g = glork(delay, stop)
    start_time = time.time()
    while 1:
        rc = win32event.MsgWaitForMultipleObjects((g.event,), 0, 500, win32event.QS_ALLEVENTS)
        if rc == win32event.WAIT_OBJECT_0:
            break
        elif rc == win32event.WAIT_OBJECT_0 + 1:
            if win32gui.PumpWaitingMessages():
                raise RuntimeError('We got an unexpected WM_QUIT message!')
        elif time.time() - start_time > 30:
            raise RuntimeError('We timed out waiting for the timers to expire!')
if __name__ == '__main__':
    demo()