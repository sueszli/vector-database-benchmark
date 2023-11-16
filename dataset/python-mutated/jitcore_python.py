from __future__ import print_function
from builtins import zip
import miasm.jitter.jitcore as jitcore
from miasm.expression.expression import ExprInt, ExprLoc
import miasm.jitter.csts as csts
from miasm.expression.simplifications import expr_simp_explicit
from miasm.jitter.emulatedsymbexec import EmulatedSymbExec

class JitCore_Python(jitcore.JitCore):
    """JiT management, using Miasm2 Symbol Execution engine as backend"""
    SymbExecClass = EmulatedSymbExec

    def __init__(self, lifter, bin_stream):
        if False:
            print('Hello World!')
        super(JitCore_Python, self).__init__(lifter, bin_stream)
        self.lifter = lifter
        self.ircfg = self.lifter.new_ircfg()
        self.symbexec = self.SymbExecClass(None, None, self.lifter, {}, sb_expr_simp=expr_simp_explicit)
        self.symbexec.enable_emulated_simplifications()

    def set_cpu_vm(self, cpu, vm):
        if False:
            while True:
                i = 10
        self.symbexec.cpu = cpu
        self.symbexec.vm = vm

    def load(self):
        if False:
            print('Hello World!')
        'Preload symbols according to current architecture'
        self.symbexec.reset_regs()

    def arch_specific(self):
        if False:
            print('Hello World!')
        'Return arch specific information for the current architecture'
        arch = self.lifter.arch
        has_delayslot = False
        if arch.name == 'mips32':
            from miasm.arch.mips32.jit import mipsCGen
            cgen_class = mipsCGen
            has_delayslot = True
        elif arch.name == 'arm':
            from miasm.arch.arm.jit import arm_CGen
            cgen_class = arm_CGen
        else:
            from miasm.jitter.codegen import CGen
            cgen_class = CGen
        return (cgen_class(self.lifter), has_delayslot)

    def add_block(self, asmblock):
        if False:
            while True:
                i = 10
        'Create a python function corresponding to an AsmBlock\n        @asmblock: AsmBlock\n        '
        (codegen, has_delayslot) = self.arch_specific()
        irblocks_list = codegen.block2assignblks(asmblock)
        instr_offsets = [line.offset for line in asmblock.lines]
        loc_db = self.lifter.loc_db
        local_loc_keys = []
        for irblocks in irblocks_list:
            for irblock in irblocks:
                local_loc_keys.append(irblock.loc_key)

        def myfunc(cpu):
            if False:
                print('Hello World!')
            'Execute the function according to cpu and vmmngr states\n            @cpu: JitCpu instance\n            '
            vmmngr = cpu.vmmngr
            exec_engine = self.symbexec
            exec_engine.update_engine_from_cpu()
            cur_loc_key = asmblock.loc_key
            update_pc = lambda value: setattr(cpu, self.lifter.pc.name, value)
            while True:
                for (instr, irblocks) in zip(asmblock.lines, irblocks_list):
                    for (index, irblock) in enumerate(irblocks):
                        if irblock.loc_key == cur_loc_key:
                            break
                    else:
                        continue
                    break
                else:
                    raise RuntimeError('Unable to find the block for %r' % cur_loc_key)
                (instr_attrib, irblocks_attributes) = codegen.get_attributes(instr, irblocks, self.log_mn, self.log_regs)
                irblock_attributes = irblocks_attributes[index]
                new_irblock = self.lifter.irbloc_fix_regs_for_mode(irblock, self.lifter.attrib)
                if index == 0:
                    if instr_attrib.log_mn:
                        print('%.8X %s' % (instr_attrib.instr.offset, instr_attrib.instr.to_string(loc_db)))
                instr = instr_attrib.instr
                for (index, assignblk) in enumerate(irblock):
                    attributes = irblock_attributes[index]
                    exec_engine.eval_updt_assignblk(assignblk)
                    if attributes.mem_read or attributes.mem_write:
                        flag = ~csts.EXCEPT_CODE_AUTOMOD & csts.EXCEPT_DO_NOT_UPDATE_PC
                        if vmmngr.get_exception() & flag != 0:
                            update_pc(instr.offset)
                            return instr.offset
                    exec_engine.update_cpu_from_engine()
                    if attributes.set_exception:
                        if cpu.get_exception() > csts.EXCEPT_NUM_UPDT_EIP:
                            update_pc(instr.offset)
                            return instr.offset
                dst = exec_engine.eval_expr(self.lifter.IRDst)
                if dst.is_int():
                    loc_key = loc_db.get_or_create_offset_location(int(dst))
                    dst = ExprLoc(loc_key, dst.size)
                assert dst.is_loc()
                loc_key = dst.loc_key
                offset = loc_db.get_location_offset(loc_key)
                if offset is None:
                    cur_loc_key = loc_key
                    continue
                if instr_attrib.log_regs:
                    update_pc(offset)
                    cpu.dump_gpregs_with_attrib(self.lifter.attrib)
                if instr_attrib.mem_read | instr_attrib.mem_write:
                    vmmngr.check_memory_breakpoint()
                    vmmngr.check_invalid_code_blocs()
                    if vmmngr.get_exception():
                        update_pc(offset)
                        return offset
                if instr_attrib.set_exception:
                    if cpu.get_exception():
                        update_pc(offset)
                        return offset
                if instr_attrib.mem_read | instr_attrib.mem_write:
                    vmmngr.reset_memory_access()
                if loc_key in local_loc_keys and offset > instr.offset:
                    cur_loc_key = loc_key
                    continue
                if has_delayslot:
                    delay_slot_set = exec_engine.eval_expr(codegen.delay_slot_set)
                    if delay_slot_set.is_int() and int(delay_slot_set) != 0:
                        return int(exec_engine.eval_expr(codegen.delay_slot_dst))
                assert offset is not None
                return offset
        offset = loc_db.get_location_offset(asmblock.loc_key)
        assert offset is not None
        self.offset_to_jitted_func[offset] = myfunc

    def exec_wrapper(self, loc_key, cpu, _offset_to_jitted_func, _stop_offsets, _max_exec_per_call):
        if False:
            print('Hello World!')
        "Call the function @loc_key with @cpu\n        @loc_key: function's loc_key\n        @cpu: JitCpu instance\n        "
        fc_ptr = self.offset_to_jitted_func[loc_key]
        return fc_ptr(cpu)