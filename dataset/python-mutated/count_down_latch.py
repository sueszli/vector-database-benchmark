import threading

class CountDownLatch:

    def __init__(self, num: int):
        if False:
            print('Hello World!')
        self._num: int = num
        self.lock = threading.Condition()

    def count_down(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.lock.acquire()
        self._num -= 1
        if self._num <= 0:
            self.lock.notify_all()
        self.lock.release()

    def wait(self) -> None:
        if False:
            while True:
                i = 10
        self.lock.acquire()
        while self._num > 0:
            self.lock.wait()
        self.lock.release()