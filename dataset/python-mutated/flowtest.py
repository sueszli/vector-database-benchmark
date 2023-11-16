from __future__ import absolute_import
from errbot import BotPlugin, botcmd

class FlowTest(BotPlugin):
    """A plugin to test the flows
    see flowtest.png for the structure.
    """

    @botcmd
    def a(self, msg, args):
        if False:
            i = 10
            return i + 15
        return 'a'

    @botcmd
    def b(self, msg, args):
        if False:
            print('Hello World!')
        return 'b'

    @botcmd
    def c(self, msg, args):
        if False:
            for i in range(10):
                print('nop')
        return 'c'

    @botcmd(flow_only=True)
    def d(self, msg, args):
        if False:
            print('Hello World!')
        return 'd'

    @botcmd
    def e(self, msg, args):
        if False:
            return 10
        return 'e'