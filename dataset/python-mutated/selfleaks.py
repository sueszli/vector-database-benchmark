from __future__ import annotations
from typing import Set, Tuple
from mypyc.analysis.dataflow import CFG, MAYBE_ANALYSIS, AnalysisResult, run_analysis
from mypyc.ir.ops import Assign, AssignMulti, BasicBlock, Box, Branch, Call, CallC, Cast, ComparisonOp, Extend, FloatComparisonOp, FloatNeg, FloatOp, GetAttr, GetElementPtr, Goto, InitStatic, IntOp, KeepAlive, LoadAddress, LoadErrorValue, LoadGlobal, LoadLiteral, LoadMem, LoadStatic, MethodCall, OpVisitor, RaiseStandardError, Register, RegisterOp, Return, SetAttr, SetMem, Truncate, TupleGet, TupleSet, Unborrow, Unbox, Unreachable
from mypyc.ir.rtypes import RInstance
GenAndKill = Tuple[Set[None], Set[None]]
CLEAN: GenAndKill = (set(), set())
DIRTY: GenAndKill = ({None}, {None})

class SelfLeakedVisitor(OpVisitor[GenAndKill]):
    """Analyze whether 'self' may be seen by arbitrary code in '__init__'.

    More formally, the set is not empty if along some path from IR entry point
    arbitrary code could have been executed that has access to 'self'.

    (We don't consider access via 'gc.get_objects()'.)
    """

    def __init__(self, self_reg: Register) -> None:
        if False:
            while True:
                i = 10
        self.self_reg = self_reg

    def visit_goto(self, op: Goto) -> GenAndKill:
        if False:
            while True:
                i = 10
        return CLEAN

    def visit_branch(self, op: Branch) -> GenAndKill:
        if False:
            return 10
        return CLEAN

    def visit_return(self, op: Return) -> GenAndKill:
        if False:
            return 10
        return DIRTY

    def visit_unreachable(self, op: Unreachable) -> GenAndKill:
        if False:
            i = 10
            return i + 15
        return CLEAN

    def visit_assign(self, op: Assign) -> GenAndKill:
        if False:
            return 10
        if op.src is self.self_reg or op.dest is self.self_reg:
            return DIRTY
        return CLEAN

    def visit_assign_multi(self, op: AssignMulti) -> GenAndKill:
        if False:
            i = 10
            return i + 15
        return CLEAN

    def visit_set_mem(self, op: SetMem) -> GenAndKill:
        if False:
            while True:
                i = 10
        return CLEAN

    def visit_call(self, op: Call) -> GenAndKill:
        if False:
            while True:
                i = 10
        fn = op.fn
        if fn.class_name and fn.name == '__init__':
            self_type = op.fn.sig.args[0].type
            assert isinstance(self_type, RInstance)
            cl = self_type.class_ir
            if not cl.init_self_leak:
                return CLEAN
        return self.check_register_op(op)

    def visit_method_call(self, op: MethodCall) -> GenAndKill:
        if False:
            for i in range(10):
                print('nop')
        return self.check_register_op(op)

    def visit_load_error_value(self, op: LoadErrorValue) -> GenAndKill:
        if False:
            i = 10
            return i + 15
        return CLEAN

    def visit_load_literal(self, op: LoadLiteral) -> GenAndKill:
        if False:
            i = 10
            return i + 15
        return CLEAN

    def visit_get_attr(self, op: GetAttr) -> GenAndKill:
        if False:
            i = 10
            return i + 15
        cl = op.class_type.class_ir
        if cl.get_method(op.attr):
            return self.check_register_op(op)
        return CLEAN

    def visit_set_attr(self, op: SetAttr) -> GenAndKill:
        if False:
            return 10
        cl = op.class_type.class_ir
        if cl.get_method(op.attr):
            return self.check_register_op(op)
        return CLEAN

    def visit_load_static(self, op: LoadStatic) -> GenAndKill:
        if False:
            print('Hello World!')
        return CLEAN

    def visit_init_static(self, op: InitStatic) -> GenAndKill:
        if False:
            while True:
                i = 10
        return self.check_register_op(op)

    def visit_tuple_get(self, op: TupleGet) -> GenAndKill:
        if False:
            for i in range(10):
                print('nop')
        return CLEAN

    def visit_tuple_set(self, op: TupleSet) -> GenAndKill:
        if False:
            for i in range(10):
                print('nop')
        return self.check_register_op(op)

    def visit_box(self, op: Box) -> GenAndKill:
        if False:
            while True:
                i = 10
        return self.check_register_op(op)

    def visit_unbox(self, op: Unbox) -> GenAndKill:
        if False:
            print('Hello World!')
        return self.check_register_op(op)

    def visit_cast(self, op: Cast) -> GenAndKill:
        if False:
            for i in range(10):
                print('nop')
        return self.check_register_op(op)

    def visit_raise_standard_error(self, op: RaiseStandardError) -> GenAndKill:
        if False:
            while True:
                i = 10
        return CLEAN

    def visit_call_c(self, op: CallC) -> GenAndKill:
        if False:
            while True:
                i = 10
        return self.check_register_op(op)

    def visit_truncate(self, op: Truncate) -> GenAndKill:
        if False:
            return 10
        return CLEAN

    def visit_extend(self, op: Extend) -> GenAndKill:
        if False:
            for i in range(10):
                print('nop')
        return CLEAN

    def visit_load_global(self, op: LoadGlobal) -> GenAndKill:
        if False:
            return 10
        return CLEAN

    def visit_int_op(self, op: IntOp) -> GenAndKill:
        if False:
            print('Hello World!')
        return CLEAN

    def visit_comparison_op(self, op: ComparisonOp) -> GenAndKill:
        if False:
            i = 10
            return i + 15
        return CLEAN

    def visit_float_op(self, op: FloatOp) -> GenAndKill:
        if False:
            return 10
        return CLEAN

    def visit_float_neg(self, op: FloatNeg) -> GenAndKill:
        if False:
            print('Hello World!')
        return CLEAN

    def visit_float_comparison_op(self, op: FloatComparisonOp) -> GenAndKill:
        if False:
            while True:
                i = 10
        return CLEAN

    def visit_load_mem(self, op: LoadMem) -> GenAndKill:
        if False:
            while True:
                i = 10
        return CLEAN

    def visit_get_element_ptr(self, op: GetElementPtr) -> GenAndKill:
        if False:
            i = 10
            return i + 15
        return CLEAN

    def visit_load_address(self, op: LoadAddress) -> GenAndKill:
        if False:
            return 10
        return CLEAN

    def visit_keep_alive(self, op: KeepAlive) -> GenAndKill:
        if False:
            return 10
        return CLEAN

    def visit_unborrow(self, op: Unborrow) -> GenAndKill:
        if False:
            i = 10
            return i + 15
        return CLEAN

    def check_register_op(self, op: RegisterOp) -> GenAndKill:
        if False:
            i = 10
            return i + 15
        if any((src is self.self_reg for src in op.sources())):
            return DIRTY
        return CLEAN

def analyze_self_leaks(blocks: list[BasicBlock], self_reg: Register, cfg: CFG) -> AnalysisResult[None]:
    if False:
        print('Hello World!')
    return run_analysis(blocks=blocks, cfg=cfg, gen_and_kill=SelfLeakedVisitor(self_reg), initial=set(), backward=False, kind=MAYBE_ANALYSIS)