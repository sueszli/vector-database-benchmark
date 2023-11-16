import os
import socket
from ..utils import BasicSegment

class Segment(BasicSegment):

    def add_to_powerline(self):
        if False:
            while True:
                i = 10
        powerline = self.powerline
        term = os.getenv('TERM')
        if not ('xterm' in term or 'rxvt' in term):
            return
        if powerline.args.shell == 'bash':
            set_title = '\\[\\e]0;\\u@\\h: \\w\\a\\]'
        elif powerline.args.shell == 'zsh':
            set_title = '%{\x1b]0;%n@%m: %~\x07%}'
        else:
            set_title = '\x1b]0;%s@%s: %s\x07' % (os.getenv('USER'), socket.gethostname().split('.')[0], powerline.cwd)
        powerline.append(set_title, None, None, '')