from pupylib.PupyModule import config, PupyModule, PupyArgumentParser
__class_name__ = 'PsModule'

@config(cat='admin')
class PsModule(PupyModule):
    """ list parent process information """
    is_module = False
    dependencies = {'windows': ['pupwinutils.processes']}

    @classmethod
    def init_argparse(cls):
        if False:
            print('Hello World!')
        cls.arg_parser = PupyArgumentParser(prog='getppid', description=cls.__doc__)

    def run(self, args):
        if False:
            while True:
                i = 10
        if self.client.is_windows():
            get_current_ppid = self.client.remote('pupwinutils.processes', 'get_current_ppid')
            outputlist = get_current_ppid()
            for out in outputlist:
                self.success('%s: %s' % (out, outputlist[out]))
            return
        else:
            getppid = self.client.remote('os', 'getppid')
            self.success('PPID: {}'.format(getppid()))