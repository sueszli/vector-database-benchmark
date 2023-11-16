import time
from datetime import datetime

class GameFrameLimiter:

    def __init__(self, fps=30):
        if False:
            return 10
        self.frame_time = 1 / fps
        self.started_at = None

    def start(self):
        if False:
            while True:
                i = 10
        self.started_at = datetime.utcnow()

    def stop_and_delay(self):
        if False:
            for i in range(10):
                print('nop')
        duration = (datetime.utcnow() - self.started_at).microseconds / 1000000
        remaining_frame_time = self.frame_time - duration
        if remaining_frame_time > 0:
            time.sleep(remaining_frame_time)

    def benchmark(self):
        if False:
            print('Hello World!')
        pass