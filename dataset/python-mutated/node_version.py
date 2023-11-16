import subprocess
from ..utils import ThreadedSegment

class Segment(ThreadedSegment):

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            p1 = subprocess.Popen(['node', '--version'], stdout=subprocess.PIPE)
            self.version = p1.communicate()[0].decode('utf-8').rstrip()
        except OSError:
            self.version = None

    def add_to_powerline(self):
        if False:
            print('Hello World!')
        self.join()
        if not self.version:
            return
        self.powerline.append('node ' + self.version, 15, 18)