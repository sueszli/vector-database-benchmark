from __future__ import absolute_import
from errbot import BotFlow, FlowRoot, botflow

class FlowDefinitions(BotFlow):
    """A plugin to test the flows
    see flowtest.png for the structure.
    """

    @botflow
    def w1(self, flow: FlowRoot):
        if False:
            return 10
        'documentation of W1'
        a_node = flow.connect('a')
        b_node = a_node.connect('b')
        c_node = a_node.connect('c')
        d_node = c_node.connect('d')
        assert a_node.hints
        assert b_node.hints
        assert c_node.hints
        assert d_node.hints

    @botflow
    def w2(self, flow: FlowRoot):
        if False:
            while True:
                i = 10
        'documentation of W2'
        c_node = flow.connect('c', auto_trigger=True)
        b_node = c_node.connect('b')
        e_node = flow.connect('e', auto_trigger=True)
        d_node = e_node.connect('d')

    @botflow
    def w3(self, flow: FlowRoot):
        if False:
            i = 10
            return i + 15
        'documentation of W3'
        c_node = flow.connect('a', room_flow=True)
        b_node = c_node.connect('b')

    @botflow
    def w4(self, flow: FlowRoot):
        if False:
            for i in range(10):
                print('nop')
        'documentation of W4'
        a_node = flow.connect('a')
        b_node = a_node.connect('b')
        c_node = b_node.connect('c')
        c_node.connect('d')
        flow.hints = False
        a_node.hints = False
        b_node.hints = True
        c_node.hint = False