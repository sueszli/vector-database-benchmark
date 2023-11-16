"""
Created on May 10, 2012

@author: sean
"""
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import opcode
import sys
py3 = sys.version_info.major >= 3
co_ord = (lambda c: c) if py3 else ord

class Instruction(object):
    """
    A Python byte-code instruction.
    """

    def __init__(self, i=-1, op=None, lineno=None):
        if False:
            return 10
        self.i = i
        self.op = op
        self.lineno = lineno
        self.oparg = None
        self.arg = None
        self.extended_arg = 0
        self.linestart = False

    @property
    def opname(self):
        if False:
            i = 10
            return i + 15
        return opcode.opname[self.op]

    @property
    def is_jump(self):
        if False:
            for i in range(10):
                print('nop')
        return self.op in opcode.hasjrel or self.op in opcode.hasjabs

    @property
    def to(self):
        if False:
            print('Hello World!')
        if self.op in opcode.hasjrel:
            return self.arg
        elif self.op in opcode.hasjabs:
            return self.oparg
        else:
            raise Exception('this is not a jump op (%s)' % (self.opname,))

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        res = '<%s(%i)' % (opcode.opname[self.op], self.i)
        if self.arg is not None:
            res += ' arg=%r' % (self.arg,)
        elif self.oparg is not None:
            res += ' oparg=%r' % (self.oparg,)
        return res + '>'

    def __str__(self):
        if False:
            while True:
                i = 10
        result = []
        if self.linestart:
            result.append('%3d' % self.lineno)
        else:
            result.append('   ')
        if self.lasti:
            result.append('-->')
        else:
            result.append('   ')
        if self.label:
            result.append('>>')
        else:
            result.append('  ')
        result.append(repr(self.i).rjust(4))
        result.append(opcode.opname[self.op].ljust(20))
        if self.op >= opcode.HAVE_ARGUMENT:
            result.append(repr(self.oparg).rjust(5))
            if self.op in opcode.hasconst:
                result.append('(' + repr(self.arg) + ')')
            elif self.op in opcode.hasname:
                result.append('(' + repr(self.arg) + ')')
            elif self.op in opcode.hasjrel:
                result.append('(to ' + repr(self.arg) + ')')
            elif self.op in opcode.haslocal:
                result.append('(' + repr(self.arg) + ')')
            elif self.op in opcode.hascompare:
                result.append('(' + repr(self.arg) + ')')
            elif self.op in opcode.hasfree:
                result.append('(' + repr(self.arg) + ')')
        return ' '.join(result)