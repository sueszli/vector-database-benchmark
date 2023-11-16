"""Helpers for dealing with nonlocal control such as 'break' and 'return'.

Model how these behave differently in different contexts.
"""
from __future__ import annotations
from abc import abstractmethod
from typing import TYPE_CHECKING
from mypyc.ir.ops import NO_TRACEBACK_LINE_NO, BasicBlock, Branch, Goto, Integer, Register, Return, Unreachable, Value
from mypyc.irbuild.targets import AssignmentTarget
from mypyc.primitives.exc_ops import restore_exc_info_op, set_stop_iteration_value
if TYPE_CHECKING:
    from mypyc.irbuild.builder import IRBuilder

class NonlocalControl:
    """ABC representing a stack frame of constructs that modify nonlocal control flow.

    The nonlocal control flow constructs are break, continue, and
    return, and their behavior is modified by a number of other
    constructs.  The most obvious is loop, which override where break
    and continue jump to, but also `except` (which needs to clear
    exc_info when left) and (eventually) finally blocks (which need to
    ensure that the finally block is always executed when leaving the
    try/except blocks).
    """

    @abstractmethod
    def gen_break(self, builder: IRBuilder, line: int) -> None:
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def gen_continue(self, builder: IRBuilder, line: int) -> None:
        if False:
            return 10
        pass

    @abstractmethod
    def gen_return(self, builder: IRBuilder, value: Value, line: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

class BaseNonlocalControl(NonlocalControl):
    """Default nonlocal control outside any statements that affect it."""

    def gen_break(self, builder: IRBuilder, line: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        assert False, 'break outside of loop'

    def gen_continue(self, builder: IRBuilder, line: int) -> None:
        if False:
            i = 10
            return i + 15
        assert False, 'continue outside of loop'

    def gen_return(self, builder: IRBuilder, value: Value, line: int) -> None:
        if False:
            i = 10
            return i + 15
        builder.add(Return(value))

class LoopNonlocalControl(NonlocalControl):
    """Nonlocal control within a loop."""

    def __init__(self, outer: NonlocalControl, continue_block: BasicBlock, break_block: BasicBlock) -> None:
        if False:
            return 10
        self.outer = outer
        self.continue_block = continue_block
        self.break_block = break_block

    def gen_break(self, builder: IRBuilder, line: int) -> None:
        if False:
            return 10
        builder.add(Goto(self.break_block))

    def gen_continue(self, builder: IRBuilder, line: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        builder.add(Goto(self.continue_block))

    def gen_return(self, builder: IRBuilder, value: Value, line: int) -> None:
        if False:
            print('Hello World!')
        self.outer.gen_return(builder, value, line)

class GeneratorNonlocalControl(BaseNonlocalControl):
    """Default nonlocal control in a generator function outside statements."""

    def gen_return(self, builder: IRBuilder, value: Value, line: int) -> None:
        if False:
            while True:
                i = 10
        builder.assign(builder.fn_info.generator_class.next_label_target, Integer(-1), line)
        builder.builder.push_error_handler(None)
        builder.goto_and_activate(BasicBlock())
        builder.call_c(set_stop_iteration_value, [value], NO_TRACEBACK_LINE_NO)
        builder.add(Unreachable())
        builder.builder.pop_error_handler()

class CleanupNonlocalControl(NonlocalControl):
    """Abstract nonlocal control that runs some cleanup code."""

    def __init__(self, outer: NonlocalControl) -> None:
        if False:
            print('Hello World!')
        self.outer = outer

    @abstractmethod
    def gen_cleanup(self, builder: IRBuilder, line: int) -> None:
        if False:
            return 10
        ...

    def gen_break(self, builder: IRBuilder, line: int) -> None:
        if False:
            return 10
        self.gen_cleanup(builder, line)
        self.outer.gen_break(builder, line)

    def gen_continue(self, builder: IRBuilder, line: int) -> None:
        if False:
            i = 10
            return i + 15
        self.gen_cleanup(builder, line)
        self.outer.gen_continue(builder, line)

    def gen_return(self, builder: IRBuilder, value: Value, line: int) -> None:
        if False:
            return 10
        self.gen_cleanup(builder, line)
        self.outer.gen_return(builder, value, line)

class TryFinallyNonlocalControl(NonlocalControl):
    """Nonlocal control within try/finally."""

    def __init__(self, target: BasicBlock) -> None:
        if False:
            print('Hello World!')
        self.target = target
        self.ret_reg: None | Register | AssignmentTarget = None

    def gen_break(self, builder: IRBuilder, line: int) -> None:
        if False:
            return 10
        builder.error('break inside try/finally block is unimplemented', line)

    def gen_continue(self, builder: IRBuilder, line: int) -> None:
        if False:
            while True:
                i = 10
        builder.error('continue inside try/finally block is unimplemented', line)

    def gen_return(self, builder: IRBuilder, value: Value, line: int) -> None:
        if False:
            i = 10
            return i + 15
        if self.ret_reg is None:
            if builder.fn_info.is_generator:
                self.ret_reg = builder.make_spill_target(builder.ret_types[-1])
            else:
                self.ret_reg = Register(builder.ret_types[-1])
        assert isinstance(self.ret_reg, (Register, AssignmentTarget))
        builder.assign(self.ret_reg, value, line)
        builder.add(Goto(self.target))

class ExceptNonlocalControl(CleanupNonlocalControl):
    """Nonlocal control for except blocks.

    Just makes sure that sys.exc_info always gets restored when we leave.
    This is super annoying.
    """

    def __init__(self, outer: NonlocalControl, saved: Value | AssignmentTarget) -> None:
        if False:
            return 10
        super().__init__(outer)
        self.saved = saved

    def gen_cleanup(self, builder: IRBuilder, line: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        builder.call_c(restore_exc_info_op, [builder.read(self.saved)], line)

class FinallyNonlocalControl(CleanupNonlocalControl):
    """Nonlocal control for finally blocks.

    Just makes sure that sys.exc_info always gets restored when we
    leave and the return register is decrefed if it isn't null.
    """

    def __init__(self, outer: NonlocalControl, saved: Value) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(outer)
        self.saved = saved

    def gen_cleanup(self, builder: IRBuilder, line: int) -> None:
        if False:
            return 10
        (target, cleanup) = (BasicBlock(), BasicBlock())
        builder.add(Branch(self.saved, target, cleanup, Branch.IS_ERROR))
        builder.activate_block(cleanup)
        builder.call_c(restore_exc_info_op, [self.saved], line)
        builder.goto_and_activate(target)