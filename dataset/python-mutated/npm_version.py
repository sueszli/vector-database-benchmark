import subprocess
from ..utils import ThreadedSegment

class Segment(ThreadedSegment):

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            p1 = subprocess.Popen(['npm', '--version'], stdout=subprocess.PIPE)
            self.version = p1.communicate()[0].decode('utf-8').rstrip()
        except OSError:
            self.version = None

    def add_to_powerline(self):
        if False:
            for i in range(10):
                print('nop')
        self.join()
        if self.version:
            self.powerline.append('npm ' + self.version, 15, 18)