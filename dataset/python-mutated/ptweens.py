import argparse
import sys
import textwrap
from pyramid.interfaces import ITweens
from pyramid.paster import bootstrap, setup_logging
from pyramid.scripts.common import parse_vars
from pyramid.tweens import INGRESS, MAIN

def main(argv=sys.argv, quiet=False):
    if False:
        i = 10
        return i + 15
    command = PTweensCommand(argv, quiet)
    return command.run()

class PTweensCommand:
    description = '    Print all implicit and explicit tween objects used by a Pyramid\n    application.  The handler output includes whether the system is using an\n    explicit tweens ordering (will be true when the "pyramid.tweens"\n    deployment setting is used) or an implicit tweens ordering (will be true\n    when the "pyramid.tweens" deployment setting is *not* used).\n\n    This command accepts one positional argument named "config_uri" which\n    specifies the PasteDeploy config file to use for the interactive\n    shell. The format is "inifile#name". If the name is left off, "main"\n    will be assumed.  Example: "ptweens myapp.ini#main".\n\n    '
    parser = argparse.ArgumentParser(description=textwrap.dedent(description), formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config_uri', nargs='?', default=None, help='The URI to the configuration file.')
    parser.add_argument('config_vars', nargs='*', default=(), help='Variables required by the config file. For example, `http_port=%%(http_port)s` would expect `http_port=8080` to be passed here.')
    stdout = sys.stdout
    bootstrap = staticmethod(bootstrap)
    setup_logging = staticmethod(setup_logging)

    def __init__(self, argv, quiet=False):
        if False:
            i = 10
            return i + 15
        self.quiet = quiet
        self.args = self.parser.parse_args(argv[1:])

    def _get_tweens(self, registry):
        if False:
            for i in range(10):
                print('nop')
        from pyramid.config import Configurator
        config = Configurator(registry=registry)
        return config.registry.queryUtility(ITweens)

    def out(self, msg):
        if False:
            return 10
        if not self.quiet:
            print(msg)

    def show_chain(self, chain):
        if False:
            return 10
        fmt = '%-10s  %-65s'
        self.out(fmt % ('Position', 'Name'))
        self.out(fmt % ('-' * len('Position'), '-' * len('Name')))
        self.out(fmt % ('-', INGRESS))
        for (pos, (name, _)) in enumerate(chain):
            self.out(fmt % (pos, name))
        self.out(fmt % ('-', MAIN))

    def run(self):
        if False:
            print('Hello World!')
        if not self.args.config_uri:
            self.out('Requires a config file argument')
            return 2
        config_uri = self.args.config_uri
        config_vars = parse_vars(self.args.config_vars)
        self.setup_logging(config_uri, global_conf=config_vars)
        env = self.bootstrap(config_uri, options=config_vars)
        registry = env['registry']
        tweens = self._get_tweens(registry)
        if tweens is not None:
            explicit = tweens.explicit
            if explicit:
                self.out('"pyramid.tweens" config value set (explicitly ordered tweens used)')
                self.out('')
                self.out('Explicit Tween Chain (used)')
                self.out('')
                self.show_chain(tweens.explicit)
                self.out('')
                self.out('Implicit Tween Chain (not used)')
                self.out('')
                self.show_chain(tweens.implicit())
            else:
                self.out('"pyramid.tweens" config value NOT set (implicitly ordered tweens used)')
                self.out('')
                self.out('Implicit Tween Chain')
                self.out('')
                self.show_chain(tweens.implicit())
        return 0
if __name__ == '__main__':
    sys.exit(main() or 0)