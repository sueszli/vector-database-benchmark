"""
This example shows how to apply a filter function to the captured output
of a run. This is often useful when using progress bars or similar in the text
UI and you don't want to store formatting characters like backspaces and
linefeeds in the database.
"""
import sys
import time
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
ex = Experiment('progress')
ex.captured_out_filter = apply_backspaces_and_linefeeds

def write_and_flush(*args):
    if False:
        while True:
            i = 10
    for arg in args:
        sys.stdout.write(arg)
    sys.stdout.flush()

class ProgressMonitor:

    def __init__(self, count):
        if False:
            print('Hello World!')
        (self.count, self.progress) = (count, 0)

    def show(self, n=1):
        if False:
            return 10
        self.progress += n
        text = 'Completed {}/{} tasks'.format(self.progress, self.count)
        write_and_flush('\x08' * 80, '\r', text)

    def done(self):
        if False:
            for i in range(10):
                print('nop')
        write_and_flush('\n')

def progress(items):
    if False:
        while True:
            i = 10
    p = ProgressMonitor(len(items))
    for item in items:
        yield item
        p.show()
    p.done()

@ex.main
def main():
    if False:
        while True:
            i = 10
    for item in progress(range(100)):
        time.sleep(0.05)
if __name__ == '__main__':
    run = ex.run_commandline()
    print('=' * 80)
    print('Captured output: ', repr(run.captured_out))