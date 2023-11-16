"""
Test interacting with the wheel system. This script is useful when testing
wheel modules
"""
import optparse
import pprint
import salt.auth
import salt.config
import salt.wheel

def parse():
    if False:
        i = 10
        return i + 15
    '\n    Parse the command line options\n    '
    parser = optparse.OptionParser()
    parser.add_option('-f', '--fun', '--function', dest='fun', help='The wheel function to execute')
    parser.add_option('-a', '--auth', dest='eauth', help='The external authentication mechanism to use')
    (options, args) = parser.parse_args()
    cli = options.__dict__
    for arg in args:
        if '=' in arg:
            comps = arg.split('=')
            cli[comps[0]] = comps[1]
    return cli

class Wheeler:
    """
    Set up communication with the wheel interface
    """

    def __init__(self, cli):
        if False:
            while True:
                i = 10
        self.opts = salt.config.master_config('/etc/salt')
        self.opts.update(cli)
        self.__eauth()
        self.wheel = salt.wheel.Wheel(self.opts)

    def __eauth(self):
        if False:
            while True:
                i = 10
        '\n        Fill in the blanks for the eauth system\n        '
        if self.opts['eauth']:
            resolver = salt.auth.Resolver(self.opts)
            res = resolver.cli(self.opts['eauth'])
        self.opts.update(res)

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Execute the wheel call\n        '
        return self.wheel.master_call(**self.opts)
if __name__ == '__main__':
    wheeler = Wheeler(parse())
    pprint.pprint(wheeler.run())