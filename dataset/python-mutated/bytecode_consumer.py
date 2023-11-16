"""
Created on Apr 28, 2012

@author: sean
"""
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from .disassembler_ import disassembler

class ByteCodeConsumer(object):
    """
    ByteCodeVisitor
    """

    def __init__(self, code):
        if False:
            print('Hello World!')
        self.code = code
        self.byte_code = code.co_code

    def consume(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Consume byte-code\n        '
        generic_consume = getattr(self, 'generic_consume', None)
        for instr in disassembler(self.code):
            method_name = 'consume_%s' % instr.opname
            method = getattr(self, method_name, generic_consume)
            if not method:
                raise AttributeError('class %r has no method %r' % (type(self).__name__, method_name))
            self.instruction_pre(instr)
            method(instr)
            self.instruction_post(instr)

    def instruction_pre(self, instr):
        if False:
            while True:
                i = 10
        '\n        consumer calls this instruction before every instruction.\n        '

    def instruction_post(self, instr):
        if False:
            i = 10
            return i + 15
        '\n        consumer calls this instruction after every instruction.\n        '

class StackedByteCodeConsumer(ByteCodeConsumer):
    """
    A consumer with the concept of a stack.
    """

    def __init__(self, code):
        if False:
            print('Hello World!')
        ByteCodeConsumer.__init__(self, code)
        self._stack = []

    def pop_top(self):
        if False:
            return 10
        return self._stack.pop()

    def push(self, value):
        if False:
            for i in range(10):
                print('nop')
        self._stack.append(value)