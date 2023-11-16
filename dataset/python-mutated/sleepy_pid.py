from ray import serve

@serve.deployment
class SleepyPid:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        import time
        time.sleep(10)

    def __call__(self) -> int:
        if False:
            return 10
        import os
        return os.getpid()
app = SleepyPid.bind()