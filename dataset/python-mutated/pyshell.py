from pupylib.PupyModule import config, PupyModule, PupyArgumentParser, REQUIRE_REPL
from pupylib.utils.rpyc_utils import redirected_stdo
import readline
__class_name__ = 'InteractivePythonShell'

def enqueue_output(out, queue):
    if False:
        return 10
    for c in iter(lambda : out.read(1), b''):
        queue.put(c)

@config(cat='admin')
class InteractivePythonShell(PupyModule):
    """ open an interactive python shell on the remote client """
    io = REQUIRE_REPL
    dependencies = ['pyshell']

    @classmethod
    def init_argparse(cls):
        if False:
            i = 10
            return i + 15
        cls.arg_parser = PupyArgumentParser(prog='pyshell', description=cls.__doc__)

    def run(self, args):
        if False:
            return 10
        PyShellController = self.client.remote('pyshell.controller', 'PyShellController', False)
        try:
            with redirected_stdo(self):
                old_completer = readline.get_completer()
                try:
                    psc = PyShellController()
                    readline.set_completer(psc.get_completer())
                    readline.parse_and_bind('tab: complete')
                    while True:
                        cmd = raw_input('>>> ')
                        psc.write(cmd)
                finally:
                    readline.set_completer(old_completer)
                    readline.parse_and_bind('tab: complete')
        except (EOFError, KeyboardInterrupt):
            self.log('pyshell closed')