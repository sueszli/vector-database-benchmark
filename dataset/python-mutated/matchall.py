from errbot import BotPlugin, botmatch

class MatchAll(BotPlugin):

    @botmatch('.*')
    def all(self, msg, match):
        if False:
            for i in range(10):
                print('nop')
        return 'Works!'