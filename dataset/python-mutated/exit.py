from pupylib.PupyModule import PupyModule, PupyArgumentParser
from pupylib.PupyErrors import PupyModuleError
__class_name__ = 'ExitModule'

class ExitModule(PupyModule):
    """ exit the client on the other side """
    is_module = False

    @classmethod
    def init_argparse(cls):
        if False:
            while True:
                i = 10
        cls.arg_parser = PupyArgumentParser(prog='exit', description=cls.__doc__)
        cls.arg_parser.add_argument('--yes', action='store_true', help='exit confirmation')

    def run(self, args):
        if False:
            print('Hello World!')
        if args.yes:
            try:
                self.client.conn.exit()
            except Exception:
                pass
        else:
            raise PupyModuleError('Please conform with --yes to perform this action.')