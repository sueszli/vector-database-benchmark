from .cmd import Cmd
from .base import AbstractHandler

class SetHandler(AbstractHandler):
    cmds = ('set',)

    def handle(self, cmd: Cmd):
        if False:
            i = 10
            return i + 15
        assert self.session is not None
        options = cmd.options
        for (key, value) in options.items():
            if value is not None:
                setattr(self.session.options, key, value)