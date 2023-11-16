from functools import cached_property
from numba.core import ir, analysis, transforms, ir_utils

class YieldPoint(object):

    def __init__(self, block, inst):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(block, ir.Block)
        assert isinstance(inst, ir.Yield)
        self.block = block
        self.inst = inst
        self.live_vars = None
        self.weak_live_vars = None

class GeneratorInfo(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.yield_points = {}
        self.state_vars = []

    def get_yield_points(self):
        if False:
            return 10
        '\n        Return an iterable of YieldPoint instances.\n        '
        return self.yield_points.values()

class VariableLifetime(object):
    """
    For lazily building information of variable lifetime
    """

    def __init__(self, blocks):
        if False:
            print('Hello World!')
        self._blocks = blocks

    @cached_property
    def cfg(self):
        if False:
            i = 10
            return i + 15
        return analysis.compute_cfg_from_blocks(self._blocks)

    @cached_property
    def usedefs(self):
        if False:
            print('Hello World!')
        return analysis.compute_use_defs(self._blocks)

    @cached_property
    def livemap(self):
        if False:
            while True:
                i = 10
        return analysis.compute_live_map(self.cfg, self._blocks, self.usedefs.usemap, self.usedefs.defmap)

    @cached_property
    def deadmaps(self):
        if False:
            while True:
                i = 10
        return analysis.compute_dead_maps(self.cfg, self._blocks, self.livemap, self.usedefs.defmap)
ir_extension_insert_dels = {}

class PostProcessor(object):
    """
    A post-processor for Numba IR.
    """

    def __init__(self, func_ir):
        if False:
            while True:
                i = 10
        self.func_ir = func_ir

    def run(self, emit_dels: bool=False, extend_lifetimes: bool=False):
        if False:
            print('Hello World!')
        '\n        Run the following passes over Numba IR:\n        - canonicalize the CFG\n        - emit explicit `del` instructions for variables\n        - compute lifetime of variables\n        - compute generator info (if function is a generator function)\n        '
        self.func_ir.blocks = transforms.canonicalize_cfg(self.func_ir.blocks)
        vlt = VariableLifetime(self.func_ir.blocks)
        self.func_ir.variable_lifetime = vlt
        bev = analysis.compute_live_variables(vlt.cfg, self.func_ir.blocks, vlt.usedefs.defmap, vlt.deadmaps.combined)
        for (offset, ir_block) in self.func_ir.blocks.items():
            self.func_ir.block_entry_vars[ir_block] = bev[offset]
        if self.func_ir.is_generator:
            self.func_ir.generator_info = GeneratorInfo()
            self._compute_generator_info()
        else:
            self.func_ir.generator_info = None
        if emit_dels:
            self._insert_var_dels(extend_lifetimes=extend_lifetimes)

    def _populate_generator_info(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Fill `index` for the Yield instruction and create YieldPoints.\n        '
        dct = self.func_ir.generator_info.yield_points
        assert not dct, 'rerunning _populate_generator_info'
        for block in self.func_ir.blocks.values():
            for inst in block.body:
                if isinstance(inst, ir.Assign):
                    yieldinst = inst.value
                    if isinstance(yieldinst, ir.Yield):
                        index = len(dct) + 1
                        yieldinst.index = index
                        yp = YieldPoint(block, yieldinst)
                        dct[yieldinst.index] = yp

    def _compute_generator_info(self):
        if False:
            i = 10
            return i + 15
        "\n        Compute the generator's state variables as the union of live variables\n        at all yield points.\n        "
        self._insert_var_dels()
        self._populate_generator_info()
        gi = self.func_ir.generator_info
        for yp in gi.get_yield_points():
            live_vars = set(self.func_ir.get_block_entry_vars(yp.block))
            weak_live_vars = set()
            stmts = iter(yp.block.body)
            for stmt in stmts:
                if isinstance(stmt, ir.Assign):
                    if stmt.value is yp.inst:
                        break
                    live_vars.add(stmt.target.name)
                elif isinstance(stmt, ir.Del):
                    live_vars.remove(stmt.value)
            else:
                assert 0, "couldn't find yield point"
            for stmt in stmts:
                if isinstance(stmt, ir.Del):
                    name = stmt.value
                    if name in live_vars:
                        live_vars.remove(name)
                        weak_live_vars.add(name)
                else:
                    break
            yp.live_vars = live_vars
            yp.weak_live_vars = weak_live_vars
        st = set()
        for yp in gi.get_yield_points():
            st |= yp.live_vars
            st |= yp.weak_live_vars
        gi.state_vars = sorted(st)
        self.remove_dels()

    def _insert_var_dels(self, extend_lifetimes=False):
        if False:
            print('Hello World!')
        '\n        Insert del statements for each variable.\n        Returns a 2-tuple of (variable definition map, variable deletion map)\n        which indicates variables defined and deleted in each block.\n\n        The algorithm avoids relying on explicit knowledge on loops and\n        distinguish between variables that are defined locally vs variables that\n        come from incoming blocks.\n        We start with simple usage (variable reference) and definition (variable\n        creation) maps on each block. Propagate the liveness info to predecessor\n        blocks until it stabilize, at which point we know which variables must\n        exist before entering each block. Then, we compute the end of variable\n        lives and insert del statements accordingly. Variables are deleted after\n        the last use. Variable referenced by terminators (e.g. conditional\n        branch and return) are deleted by the successors or the caller.\n        '
        vlt = self.func_ir.variable_lifetime
        self._patch_var_dels(vlt.deadmaps.internal, vlt.deadmaps.escaping, extend_lifetimes=extend_lifetimes)

    def _patch_var_dels(self, internal_dead_map, escaping_dead_map, extend_lifetimes=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Insert delete in each block\n        '
        for (offset, ir_block) in self.func_ir.blocks.items():
            internal_dead_set = internal_dead_map[offset].copy()
            delete_pts = []
            for stmt in reversed(ir_block.body[:-1]):
                live_set = set((v.name for v in stmt.list_vars()))
                dead_set = live_set & internal_dead_set
                for (T, def_func) in ir_extension_insert_dels.items():
                    if isinstance(stmt, T):
                        done_dels = def_func(stmt, dead_set)
                        dead_set -= done_dels
                        internal_dead_set -= done_dels
                delete_pts.append((stmt, dead_set))
                internal_dead_set -= dead_set
            body = []
            lastloc = ir_block.loc
            del_store = []
            for (stmt, delete_set) in reversed(delete_pts):
                if extend_lifetimes:
                    lastloc = ir_block.body[-1].loc
                else:
                    lastloc = stmt.loc
                if not isinstance(stmt, ir.Del):
                    body.append(stmt)
                for var_name in sorted(delete_set, reverse=True):
                    delnode = ir.Del(var_name, loc=lastloc)
                    if extend_lifetimes:
                        del_store.append(delnode)
                    else:
                        body.append(delnode)
            if extend_lifetimes:
                body.extend(del_store)
            body.append(ir_block.body[-1])
            ir_block.body = body
            escape_dead_set = escaping_dead_map[offset]
            for var_name in sorted(escape_dead_set):
                ir_block.prepend(ir.Del(var_name, loc=ir_block.body[0].loc))

    def remove_dels(self):
        if False:
            i = 10
            return i + 15
        '\n        Strips the IR of Del nodes\n        '
        ir_utils.remove_dels(self.func_ir.blocks)