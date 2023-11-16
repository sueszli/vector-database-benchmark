from pupylib.PupyModule import config, PupyModule, PupyArgumentParser
__class_name__ = 'PsModule'

@config(cat='admin')
class PsModule(PupyModule):
    """ list process information """
    is_module = False

    @classmethod
    def init_argparse(cls):
        if False:
            while True:
                i = 10
        cls.arg_parser = PupyArgumentParser(prog='getpid', description=cls.__doc__)

    def run(self, args):
        if False:
            return 10
        getpid = self.client.remote('os', 'getpid')
        pid = getpid()
        self.success('PID: {}'.format(pid))