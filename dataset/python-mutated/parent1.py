from errbot import BotPlugin

class Parent1(BotPlugin):

    def shared_function(self):
        if False:
            for i in range(10):
                print('nop')
        return 'youpi'