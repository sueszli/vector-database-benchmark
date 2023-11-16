from miasm.core.utils import decode_hex, encode_hex
import miasm.expression.expression as m2_expr
from miasm.ir.symbexec import SymbolicExecutionEngine
from miasm.arch.x86.arch import is_op_segm

class EmulatedSymbExec(SymbolicExecutionEngine):
    """Symbolic exec instance linked with a jitter"""
    x86_cpuid = {0: {0: 10, 1: 1970169159, 2: 1818588270, 3: 1231384169}, 1: {0: 132690, 1: 2048, 2: 521, 3: 126614015}, 2: {0: 0, 1: 0, 2: 0, 3: 0}, 4: {0: 0, 1: 0, 2: 0, 3: 0}, 7: {0: 0, 1: 1 << 0 | 1 << 3, 2: 0, 3: 0}, 2147483648: {0: 2147483656, 1: 0, 2: 0, 3: 0}, 2147483649: {0: 0, 1: 0, 2: 1 << 0 | 1 << 8, 3: 1 << 11 | 1 << 29}}

    def __init__(self, cpu, vm, *args, **kwargs):
        if False:
            while True:
                i = 10
        'Instantiate an EmulatedSymbExec, associated to CPU @cpu and bind\n        memory accesses.\n        @cpu: JitCpu instance\n        '
        super(EmulatedSymbExec, self).__init__(*args, **kwargs)
        self.cpu = cpu
        self.vm = vm

    def reset_regs(self):
        if False:
            while True:
                i = 10
        'Set registers value to 0. Ignore register aliases'
        for reg in self.lifter.arch.regs.all_regs_ids_no_alias:
            self.symbols.symbols_id[reg] = m2_expr.ExprInt(0, size=reg.size)

    def mem_read(self, expr_mem):
        if False:
            print('Hello World!')
        'Memory read wrapper for symbolic execution\n        @expr_mem: ExprMem'
        addr = expr_mem.ptr
        if not addr.is_int():
            return super(EmulatedSymbExec, self).mem_read(expr_mem)
        addr = int(addr)
        size = expr_mem.size // 8
        value = self.vm.get_mem(addr, size)
        if self.vm.is_little_endian():
            value = value[::-1]
        self.vm.add_mem_read(addr, size)
        return m2_expr.ExprInt(int(encode_hex(value), 16), expr_mem.size)

    def mem_write(self, dest, data):
        if False:
            for i in range(10):
                print('nop')
        'Memory read wrapper for symbolic execution\n        @dest: ExprMem instance\n        @data: Expr instance'
        data = self.expr_simp(data)
        if not isinstance(data, m2_expr.ExprInt):
            raise RuntimeError('A simplification is missing: %s' % data)
        to_write = int(data)
        addr = int(dest.ptr)
        size = data.size // 8
        content = hex(to_write).replace('0x', '').replace('L', '')
        content = '0' * (size * 2 - len(content)) + content
        content = decode_hex(content)
        if self.vm.is_little_endian():
            content = content[::-1]
        self.vm.set_mem(addr, content)

    def update_cpu_from_engine(self):
        if False:
            print('Hello World!')
        'Updates @cpu instance according to new CPU values'
        for symbol in self.symbols:
            if isinstance(symbol, m2_expr.ExprId):
                if hasattr(self.cpu, symbol.name):
                    value = self.symbols.symbols_id[symbol]
                    if not isinstance(value, m2_expr.ExprInt):
                        raise ValueError('A simplification is missing: %s' % value)
                    setattr(self.cpu, symbol.name, int(value))
            else:
                raise NotImplementedError('Type not handled: %s' % symbol)

    def update_engine_from_cpu(self):
        if False:
            print('Hello World!')
        'Updates CPU values according to @cpu instance'
        for symbol in self.symbols:
            if isinstance(symbol, m2_expr.ExprId):
                if hasattr(self.cpu, symbol.name):
                    value = m2_expr.ExprInt(getattr(self.cpu, symbol.name), symbol.size)
                    self.symbols.symbols_id[symbol] = value
            else:
                raise NotImplementedError('Type not handled: %s' % symbol)

    def _simp_handle_segm(self, e_s, expr):
        if False:
            while True:
                i = 10
        "Handle 'segm' operation"
        if not is_op_segm(expr):
            return expr
        if not expr.args[0].is_int():
            return expr
        segm_nb = int(expr.args[0])
        segmaddr = self.cpu.get_segm_base(segm_nb)
        return e_s(m2_expr.ExprInt(segmaddr, expr.size) + expr.args[1])

    def _simp_handle_x86_cpuid(self, e_s, expr):
        if False:
            print('Hello World!')
        'From miasm/jitter/op_semantics.h: x86_cpuid'
        if expr.op != 'x86_cpuid':
            return expr
        if any((not arg.is_int() for arg in expr.args)):
            return expr
        (a, reg_num) = (int(arg) for arg in expr.args)
        return m2_expr.ExprInt(self.x86_cpuid[a][reg_num], expr.size)

    def enable_emulated_simplifications(self):
        if False:
            return 10
        'Enable simplifications needing a CPU instance on associated\n        ExpressionSimplifier\n        '
        self.expr_simp.enable_passes({m2_expr.ExprOp: [self._simp_handle_segm, self._simp_handle_x86_cpuid]})