from errbot import BotPlugin, cmdfilter

class TestCommandNotFoundFilter(BotPlugin):

    @cmdfilter(catch_unprocessed=True)
    def command_not_found(self, msg, cmd, args, dry_run, emptycmd=False):
        if False:
            for i in range(10):
                print('nop')
        if not emptycmd:
            return (msg, cmd, args)
        return f'Command fell through: {msg}'