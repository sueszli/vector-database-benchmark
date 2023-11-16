from __future__ import print_function
import os
import tempfile
from builtins import int as int_types
from future.utils import viewitems, viewvalues
import idaapi
import ida_kernwin
import idc
import ida_funcs
import idautils
from miasm.core.bin_stream_ida import bin_stream_ida
from miasm.expression.simplifications import expr_simp
from miasm.ir.ir import IRBlock, AssignBlock
from miasm.analysis.data_flow import load_from_int
from utils import guess_machine, expr2colorstr
from miasm.expression.expression import ExprLoc, ExprInt, ExprOp, ExprAssign
from miasm.analysis.simplifier import IRCFGSimplifierCommon, IRCFGSimplifierSSA
from miasm.core.locationdb import LocationDB
TYPE_GRAPH_IR = 0
TYPE_GRAPH_IRSSA = 1
TYPE_GRAPH_IRSSAUNSSA = 2
OPTION_GRAPH_CODESIMPLIFY = 1
OPTION_GRAPH_USE_IDA_STACK = 2
OPTION_GRAPH_DONTMODSTACK = 4
OPTION_GRAPH_LOADMEMINT = 8

class GraphIRForm(ida_kernwin.Form):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        ida_kernwin.Form.__init__(self, 'BUTTON YES* Launch\nBUTTON CANCEL NONE\nGraph IR Settings\n\n{FormChangeCb}\nAnalysis:\n<Graph IR :{rGraphIR}>\n<Graph IR + SSA :{rGraphIRSSA}>\n<Graph IR + SSA + UnSSA :{rGraphIRSSAUNSSA}>{cScope}>\n\nOptions:\n<Simplify code:{rCodeSimplify}>\n<Use ida stack:{rUseIdaStack}>\n<Subcalls dont change stack:{rDontModStack}>\n<Load static memory:{rLoadMemInt}>{cOptions}>\n', {'FormChangeCb': ida_kernwin.Form.FormChangeCb(self.OnFormChange), 'cScope': ida_kernwin.Form.RadGroupControl(('rGraphIR', 'rGraphIRSSA', 'rGraphIRSSAUNSSA')), 'cOptions': ida_kernwin.Form.ChkGroupControl(('rCodeSimplify', 'rUseIdaStack', 'rDontModStack', 'rLoadMemInt'))})
        (form, _) = self.Compile()
        form.rCodeSimplify.checked = True
        form.rUseIdaStack.checked = True
        form.rDontModStack.checked = False
        form.rLoadMemInt.checked = False

    def OnFormChange(self, _):
        if False:
            return 10
        return 1

def label_init(self, name='', offset=None):
    if False:
        print('Hello World!')
    self.fixedblocs = False
    if isinstance(name, int_types):
        name = 'loc_%X' % (int(name) & 18446744073709551615)
    self.name = name
    self.attrib = None
    if offset is None:
        self.offset = None
    else:
        self.offset = int(offset)

def label_str(self):
    if False:
        while True:
            i = 10
    if isinstance(self.offset, int_types):
        return '%s:0x%x' % (self.name, self.offset)
    return '%s:%s' % (self.name, self.offset)

def color_irblock(irblock, lifter):
    if False:
        print('Hello World!')
    out = []
    lbl = idaapi.COLSTR('%s:' % lifter.loc_db.pretty_str(irblock.loc_key), idaapi.SCOLOR_INSN)
    out.append(lbl)
    for assignblk in irblock:
        for (dst, src) in sorted(viewitems(assignblk)):
            dst_f = expr2colorstr(dst, loc_db=lifter.loc_db)
            src_f = expr2colorstr(src, loc_db=lifter.loc_db)
            line = idaapi.COLSTR('%s = %s' % (dst_f, src_f), idaapi.SCOLOR_INSN)
            out.append('    %s' % line)
        out.append('')
    out.pop()
    return '\n'.join(out)

class GraphMiasmIR(idaapi.GraphViewer):

    def __init__(self, ircfg, title, result):
        if False:
            print('Hello World!')
        idaapi.GraphViewer.__init__(self, title)
        self.ircfg = ircfg
        self.result = result
        self.names = {}

    def OnRefresh(self):
        if False:
            for i in range(10):
                print('nop')
        self.Clear()
        addr_id = {}
        for (loc_key, irblock) in viewitems(self.ircfg.blocks):
            id_irblock = self.AddNode(color_irblock(irblock, self.ircfg))
            addr_id[loc_key] = id_irblock
        for (loc_key, irblock) in viewitems(self.ircfg.blocks):
            if not irblock:
                continue
            all_dst = self.ircfg.dst_trackback(irblock)
            for dst in all_dst:
                if not dst.is_loc():
                    continue
                if not dst.loc_key in self.ircfg.blocks:
                    continue
                node1 = addr_id[loc_key]
                node2 = addr_id[dst.loc_key]
                self.AddEdge(node1, node2)
        return True

    def OnGetText(self, node_id):
        if False:
            while True:
                i = 10
        return str(self[node_id])

    def OnSelect(self, _):
        if False:
            i = 10
            return i + 15
        return True

    def OnClick(self, _):
        if False:
            i = 10
            return i + 15
        return True

    def Show(self):
        if False:
            while True:
                i = 10
        if not idaapi.GraphViewer.Show(self):
            return False
        return True

def is_addr_ro_variable(bs, addr, size):
    if False:
        return 10
    '\n    Return True if address at @addr is a read-only variable.\n    WARNING: Quick & Dirty\n\n    @addr: integer representing the address of the variable\n    @size: size in bits\n\n    '
    try:
        _ = bs.getbytes(addr, size // 8)
    except IOError:
        return False
    return True

def build_graph(start_addr, type_graph, simplify=False, use_ida_stack=True, dontmodstack=False, loadint=False, verbose=False):
    if False:
        while True:
            i = 10
    machine = guess_machine(addr=start_addr)
    (dis_engine, lifter_model_call) = (machine.dis_engine, machine.lifter_model_call)

    class IRADelModCallStack(lifter_model_call):

        def call_effects(self, addr, instr):
            if False:
                i = 10
                return i + 15
            (assignblks, extra) = super(IRADelModCallStack, self).call_effects(addr, instr)
            if use_ida_stack:
                stk_before = idc.get_spd(instr.offset)
                stk_after = idc.get_spd(instr.offset + instr.l)
                stk_diff = stk_after - stk_before
                print(hex(stk_diff))
                call_assignblk = AssignBlock([ExprAssign(self.ret_reg, ExprOp('call_func_ret', addr)), ExprAssign(self.sp, self.sp + ExprInt(stk_diff, self.sp.size))], instr)
                return ([call_assignblk], [])
            else:
                if not dontmodstack:
                    return (assignblks, extra)
                out = []
                for assignblk in assignblks:
                    dct = dict(assignblk)
                    dct = {dst: src for (dst, src) in viewitems(dct) if dst != self.sp}
                    out.append(AssignBlock(dct, assignblk.instr))
            return (out, extra)
    if verbose:
        print('Arch', dis_engine)
    fname = idc.get_root_filename()
    if verbose:
        print(fname)
    bs = bin_stream_ida()
    loc_db = LocationDB()
    mdis = dis_engine(bs, loc_db=loc_db)
    lifter = IRADelModCallStack(loc_db)
    for (addr, name) in idautils.Names():
        if name is None:
            continue
        if loc_db.get_offset_location(addr) or loc_db.get_name_location(name):
            continue
        loc_db.add_location(name, addr)
    if verbose:
        print('start disasm')
    if verbose:
        print(hex(start_addr))
    asmcfg = mdis.dis_multiblock(start_addr)
    entry_points = set([loc_db.get_offset_location(start_addr)])
    if verbose:
        print('generating graph')
        open('asm_flow.dot', 'w').write(asmcfg.dot())
        print('generating IR... %x' % start_addr)
    ircfg = lifter.new_ircfg_from_asmcfg(asmcfg)
    if verbose:
        print('IR ok... %x' % start_addr)
    for irb in list(viewvalues(ircfg.blocks)):
        irs = []
        for assignblk in irb:
            new_assignblk = {expr_simp(dst): expr_simp(src) for (dst, src) in viewitems(assignblk)}
            irs.append(AssignBlock(new_assignblk, instr=assignblk.instr))
        ircfg.blocks[irb.loc_key] = IRBlock(loc_db, irb.loc_key, irs)
    if verbose:
        out = ircfg.dot()
        open(os.path.join(tempfile.gettempdir(), 'graph.dot'), 'wb').write(out)
    title = 'Miasm IR graph'
    head = list(entry_points)[0]
    if simplify:
        ircfg_simplifier = IRCFGSimplifierCommon(lifter)
        ircfg_simplifier.simplify(ircfg, head)
        title += ' (simplified)'
    if type_graph == TYPE_GRAPH_IR:
        graph = GraphMiasmIR(ircfg, title, None)
        graph.Show()
        return

    class IRAOutRegs(lifter_model_call):

        def get_out_regs(self, block):
            if False:
                while True:
                    i = 10
            regs_todo = super(IRAOutRegs, self).get_out_regs(block)
            out = {}
            for assignblk in block:
                for dst in assignblk:
                    reg = self.ssa_var.get(dst, None)
                    if reg is None:
                        continue
                    if reg in regs_todo:
                        out[reg] = dst
            return set(viewvalues(out))
    for loc in ircfg.leaves():
        irblock = ircfg.blocks.get(loc)
        if irblock is None:
            continue
        regs = {}
        for reg in lifter.get_out_regs(irblock):
            regs[reg] = reg
        assignblks = list(irblock)
        new_assiblk = AssignBlock(regs, assignblks[-1].instr)
        assignblks.append(new_assiblk)
        new_irblock = IRBlock(irblock.loc_db, irblock.loc_key, assignblks)
        ircfg.blocks[loc] = new_irblock

    class CustomIRCFGSimplifierSSA(IRCFGSimplifierSSA):

        def do_simplify(self, ssa, head):
            if False:
                print('Hello World!')
            modified = super(CustomIRCFGSimplifierSSA, self).do_simplify(ssa, head)
            if loadint:
                modified |= load_from_int(ssa.graph, bs, is_addr_ro_variable)
            return modified

        def simplify(self, ircfg, head):
            if False:
                for i in range(10):
                    print('nop')
            ssa = self.ircfg_to_ssa(ircfg, head)
            ssa = self.do_simplify_loop(ssa, head)
            if type_graph == TYPE_GRAPH_IRSSA:
                ret = ssa.graph
            elif type_graph == TYPE_GRAPH_IRSSAUNSSA:
                ircfg = self.ssa_to_unssa(ssa, head)
                ircfg_simplifier = IRCFGSimplifierCommon(self.lifter)
                ircfg_simplifier.simplify(ircfg, head)
                ret = ircfg
            else:
                raise ValueError('Unknown option')
            return ret
    head = list(entry_points)[0]
    simplifier = CustomIRCFGSimplifierSSA(lifter)
    ircfg = simplifier.simplify(ircfg, head)
    open('final.dot', 'w').write(ircfg.dot())
    graph = GraphMiasmIR(ircfg, title, None)
    graph.Show()

def function_graph_ir():
    if False:
        i = 10
        return i + 15
    settings = GraphIRForm()
    ret = settings.Execute()
    if not ret:
        return
    func = ida_funcs.get_func(idc.get_screen_ea())
    func_addr = func.start_ea
    build_graph(func_addr, settings.cScope.value, simplify=settings.cOptions.value & OPTION_GRAPH_CODESIMPLIFY, use_ida_stack=settings.cOptions.value & OPTION_GRAPH_USE_IDA_STACK, dontmodstack=settings.cOptions.value & OPTION_GRAPH_DONTMODSTACK, loadint=settings.cOptions.value & OPTION_GRAPH_LOADMEMINT, verbose=False)
    return
if __name__ == '__main__':
    function_graph_ir()