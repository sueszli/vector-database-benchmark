from errbot import BotPlugin, botcmd

class Child2(BotPlugin):

    def shared_function(self):
        if False:
            return 10
        return 'Hello from Child2'