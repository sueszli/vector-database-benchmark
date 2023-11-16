import sys
import time
from st2common.runners.base_action import Action

class PrintToStdoutAndStderrAction(Action):

    def run(self, count=100, sleep_delay=0.5):
        if False:
            i = 10
            return i + 15
        for i in range(0, count):
            if i % 2 == 0:
                text = 'stderr'
                stream = sys.stderr
            else:
                text = 'stdout'
                stream = sys.stdout
            stream.write('%s -> Line: %s\n' % (text, i + 1))
            stream.flush()
            time.sleep(sleep_delay)