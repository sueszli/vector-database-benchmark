from __future__ import absolute_import
from errbot import BotPlugin, Command, arg_botcmd, botcmd, botmatch

def say_foo(plugin, msg, args):
    if False:
        return 10
    return 'foo %s' % type(plugin)

class Dyna(BotPlugin):
    """Just a test plugin to see if dynamic plugin API works."""

    @botcmd
    def add_simple(self, _, _1):
        if False:
            while True:
                i = 10
        simple1 = Command(lambda plugin, msg, args: 'yep %s' % type(plugin), name='say_yep')
        simple2 = Command(say_foo)
        self.create_dynamic_plugin('simple with special#', (simple1, simple2), doc='documented')
        return 'added'

    @botcmd
    def remove_simple(self, msg, args):
        if False:
            i = 10
            return i + 15
        self.destroy_dynamic_plugin('simple with special#')
        return 'removed'

    @botcmd
    def add_arg(self, _, _1):
        if False:
            i = 10
            return i + 15
        cmd1_name = 'echo_to_me'
        cmd1 = Command(lambda plugin, msg, args: 'string to echo is %s' % args.positional_arg, cmd_type=arg_botcmd, cmd_args=('positional_arg',), cmd_kwargs={'unpack_args': False, 'name': cmd1_name}, name=cmd1_name)
        self.create_dynamic_plugin('arg', (cmd1,), doc='documented')
        return 'added'

    @botcmd
    def remove_arg(self, msg, args):
        if False:
            return 10
        self.destroy_dynamic_plugin('arg')
        return 'removed'

    @botcmd
    def add_re(self, _, _1):
        if False:
            return 10
        re1 = Command(lambda plugin, msg, match: 'fffound', name='ffound', cmd_type=botmatch, cmd_args=('^.*cheese.*$',))
        self.create_dynamic_plugin('re', (re1,))
        return 'added'

    @botcmd
    def remove_re(self, msg, args):
        if False:
            for i in range(10):
                print('nop')
        self.destroy_dynamic_plugin('re')
        return 'removed'

    @botcmd
    def add_saw(self, _, _1):
        if False:
            return 10
        re1 = Command(lambda plugin, msg, args: '+'.join(args), name='splitme', cmd_type=botcmd, cmd_kwargs={'split_args_with': ','})
        self.create_dynamic_plugin('saw', (re1,))
        return 'added'

    @botcmd
    def remove_saw(self, msg, args):
        if False:
            print('Hello World!')
        self.destroy_dynamic_plugin('saw')
        return 'removed'

    @botcmd
    def clash(self, msg, args):
        if False:
            return 10
        return 'original'

    @botcmd
    def add_clashing(self, _, _1):
        if False:
            for i in range(10):
                print('nop')
        simple1 = Command(lambda plugin, msg, args: 'dynamic', name='clash')
        self.create_dynamic_plugin('clashing', (simple1,))
        return 'added'

    @botcmd
    def remove_clashing(self, _, _1):
        if False:
            return 10
        self.destroy_dynamic_plugin('clashing')
        return 'removed'