from pupylib.PupyModule import config, PupyModule, PupyArgumentParser
__class_name__ = 'getuid'

@config(cat='admin')
class getuid(PupyModule):
    """ get username """
    is_module = False
    dependencies = ['pupyutils.basic_cmds']

    @classmethod
    def init_argparse(cls):
        if False:
            while True:
                i = 10
        cls.arg_parser = PupyArgumentParser(prog='getuid', description=cls.__doc__)

    def run(self, args):
        if False:
            for i in range(10):
                print('nop')
        getuid = self.client.remote('pupyutils.basic_cmds', 'getuid')
        self.success(getuid())