from errbot import BotPlugin, botcmd

class Child1(BotPlugin):

    def shared_function(self):
        if False:
            for i in range(10):
                print('nop')
        return 'Hello from Child1'