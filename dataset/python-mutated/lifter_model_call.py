from miasm.ir.analysis import LifterModelCall
from miasm.arch.msp430.sem import Lifter_MSP430
from miasm.ir.ir import AssignBlock
from miasm.expression.expression import *

class LifterModelCallMsp430Base(Lifter_MSP430, LifterModelCall):

    def __init__(self, loc_db):
        if False:
            i = 10
            return i + 15
        Lifter_MSP430.__init__(self, loc_db)
        self.ret_reg = self.arch.regs.R15

    def call_effects(self, addr, instr):
        if False:
            i = 10
            return i + 15
        call_assignblk = AssignBlock([ExprAssign(self.ret_reg, ExprOp('call_func_ret', addr, self.sp, self.arch.regs.R15)), ExprAssign(self.sp, ExprOp('call_func_stack', addr, self.sp))], instr)
        return ([call_assignblk], [])

class LifterModelCallMsp430(LifterModelCallMsp430Base):

    def __init__(self, loc_db):
        if False:
            print('Hello World!')
        LifterModelCallMsp430Base.__init__(self, loc_db)

    def get_out_regs(self, _):
        if False:
            print('Hello World!')
        return set([self.ret_reg, self.sp])