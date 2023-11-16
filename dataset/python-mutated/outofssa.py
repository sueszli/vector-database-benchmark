from future.utils import viewitems, viewvalues
from miasm.expression.expression import ExprId
from miasm.ir.ir import IRBlock, AssignBlock
from miasm.analysis.ssa import get_phi_sources_parent_block, irblock_has_phi

class Varinfo(object):
    """Store liveness information for a variable"""
    __slots__ = ['live_index', 'loc_key', 'index']

    def __init__(self, live_index, loc_key, index):
        if False:
            i = 10
            return i + 15
        self.live_index = live_index
        self.loc_key = loc_key
        self.index = index

class UnSSADiGraph(object):
    """
    Implements unssa algorithm
    Revisiting Out-of-SSA Translation for Correctness, Code Quality, and
    Efficiency
    """

    def __init__(self, ssa, head, cfg_liveness):
        if False:
            while True:
                i = 10
        self.cfg_liveness = cfg_liveness
        self.ssa = ssa
        self.head = head
        self.copy_vars = set()
        self.phi_parent_sources = {}
        self.phi_destinations = {}
        self.phi_new_var = {}
        self.new_var_to_srcs_parents = {}
        self.merge_state = {}
        self.isolate_phi_nodes_block()
        self.init_phis_merge_state()
        self.order_ssa_var_dom()
        self.aggressive_coalesce_block()
        self.insert_parallel_copy()
        self.replace_merge_sets()
        self.remove_assign_eq()

    def insert_parallel_copy(self):
        if False:
            while True:
                i = 10
        "\n        Naive Out-of-SSA from CSSA (without coalescing for now)\n        - Replace Phi\n        - Create room for parallel copies in Phi's parents\n        "
        ircfg = self.ssa.graph
        for irblock in list(viewvalues(ircfg.blocks)):
            if not irblock_has_phi(irblock):
                continue
            parallel_copies = {}
            for dst in self.phi_destinations[irblock.loc_key]:
                new_var = self.phi_new_var[dst]
                parallel_copies[dst] = new_var
            assignblks = list(irblock)
            assignblks[0] = AssignBlock(parallel_copies, irblock[0].instr)
            new_irblock = IRBlock(irblock.loc_db, irblock.loc_key, assignblks)
            ircfg.blocks[irblock.loc_key] = new_irblock
            parent_to_parallel_copies = {}
            parallel_copies = {}
            for dst in irblock[0]:
                new_var = self.phi_new_var[dst]
                for (parent, src) in self.phi_parent_sources[dst]:
                    parent_to_parallel_copies.setdefault(parent, {})[new_var] = src
            for (parent, parallel_copies) in viewitems(parent_to_parallel_copies):
                parent = ircfg.blocks[parent]
                assignblks = list(parent)
                assignblks.append(AssignBlock(parallel_copies, parent[-1].instr))
                new_irblock = IRBlock(parent.loc_db, parent.loc_key, assignblks)
                ircfg.blocks[parent.loc_key] = new_irblock

    def create_copy_var(self, var):
        if False:
            while True:
                i = 10
        '\n        Generate a new var standing for @var\n        @var: variable to replace\n        '
        new_var = ExprId('var%d' % len(self.copy_vars), var.size)
        self.copy_vars.add(new_var)
        return new_var

    def isolate_phi_nodes_block(self):
        if False:
            while True:
                i = 10
        '\n        Init structures and virtually insert parallel copy before/after each phi\n        node\n        '
        ircfg = self.ssa.graph
        for irblock in viewvalues(ircfg.blocks):
            if not irblock_has_phi(irblock):
                continue
            for (dst, sources) in viewitems(irblock[0]):
                assert sources.is_op('Phi')
                new_var = self.create_copy_var(dst)
                self.phi_new_var[dst] = new_var
                var_to_parents = get_phi_sources_parent_block(self.ssa.graph, irblock.loc_key, sources.args)
                for src in sources.args:
                    parents = var_to_parents[src]
                    self.new_var_to_srcs_parents.setdefault(new_var, set()).update(parents)
                    for parent in parents:
                        self.phi_parent_sources.setdefault(dst, set()).add((parent, src))
            self.phi_destinations[irblock.loc_key] = set(irblock[0])

    def init_phis_merge_state(self):
        if False:
            print('Hello World!')
        '\n        Generate trivial coalescing of phi variable and itself\n        '
        for phi_new_var in viewvalues(self.phi_new_var):
            self.merge_state.setdefault(phi_new_var, set([phi_new_var]))

    def order_ssa_var_dom(self):
        if False:
            for i in range(10):
                print('nop')
        'Compute dominance order of each ssa variable'
        ircfg = self.ssa.graph
        dominator_tree = ircfg.compute_dominator_tree(self.head)
        self.var_to_varinfo = {}
        live_index = 0
        for loc_key in dominator_tree.walk_depth_first_forward(self.head):
            irblock = ircfg.blocks.get(loc_key, None)
            if irblock is None:
                continue
            if irblock_has_phi(irblock):
                for dst in irblock[0]:
                    if not dst.is_id():
                        continue
                    new_var = self.phi_new_var[dst]
                    self.var_to_varinfo[new_var] = Varinfo(live_index, loc_key, None)
                live_index += 1
            for (index, assignblk) in enumerate(irblock):
                used = False
                for dst in assignblk:
                    if not dst.is_id():
                        continue
                    if dst in self.ssa.immutable_ids:
                        continue
                    assert dst not in self.var_to_varinfo
                    self.var_to_varinfo[dst] = Varinfo(live_index, loc_key, index)
                    used = True
                if used:
                    live_index += 1

    def ssa_def_dominates(self, node_a, node_b):
        if False:
            while True:
                i = 10
        '\n        Return living index order of @node_a and @node_b\n        @node_a: Varinfo instance\n        @node_b: Varinfo instance\n        '
        ret = self.var_to_varinfo[node_a].live_index <= self.var_to_varinfo[node_b].live_index
        return ret

    def merge_set_sort(self, merge_set):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a sorted list of (live_index, var) from @merge_set in dominance\n        order\n        @merge_set: set of coalescing variables\n        '
        return sorted(((self.var_to_varinfo[var].live_index, var) for var in merge_set))

    def ssa_def_is_live_at(self, node_a, node_b, parent):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return True if @node_a is live during @node_b definition\n        If @parent is None, this is a liveness test for a post phi variable;\n        Else, it is a liveness test for a variable source of the phi node\n\n        @node_a: Varinfo instance\n        @node_b: Varinfo instance\n        @parent: Optional parent location of the phi source\n        '
        (loc_key_b, index_b) = (self.var_to_varinfo[node_b].loc_key, self.var_to_varinfo[node_b].index)
        if parent and index_b is None:
            index_b = 0
        if node_a not in self.new_var_to_srcs_parents:
            liveness_b = self.cfg_liveness.blocks[loc_key_b].infos[index_b]
            return node_a in liveness_b.var_out
        for def_loc_key in self.new_var_to_srcs_parents[node_a]:
            if def_loc_key == parent:
                continue
            liveness_end_block = self.cfg_liveness.blocks[def_loc_key].infos[-1]
            if node_b in liveness_end_block.var_out:
                return True
        return False

    def merge_nodes_interfere(self, node_a, node_b, parent):
        if False:
            i = 10
            return i + 15
        '\n        Return True if @node_a and @node_b interfere\n        @node_a: variable\n        @node_b: variable\n        @parent: Optional parent location of the phi source for liveness tests\n\n        Interference check is: is x live at y definition (or reverse)\n        TODO: add Value-based interference improvement\n        '
        if self.var_to_varinfo[node_a].live_index == self.var_to_varinfo[node_b].live_index:
            return True
        if self.var_to_varinfo[node_a].live_index < self.var_to_varinfo[node_b].live_index:
            return self.ssa_def_is_live_at(node_a, node_b, parent)
        return self.ssa_def_is_live_at(node_b, node_a, parent)

    def merge_sets_interfere(self, merge_a, merge_b, parent):
        if False:
            i = 10
            return i + 15
        '\n        Return True if no variable in @merge_a and @merge_b interferes.\n\n        Implementation of "Algorithm 2: Check intersection in a set of variables"\n\n        @merge_a: a dom ordered list of equivalent variables\n        @merge_b: a dom ordered list of equivalent variables\n        @parent: Optional parent location of the phi source for liveness tests\n        '
        if merge_a == merge_b:
            return False
        merge_a_list = self.merge_set_sort(merge_a)
        merge_b_list = self.merge_set_sort(merge_b)
        dom = []
        while merge_a_list or merge_b_list:
            if not merge_a_list:
                (_, current) = merge_b_list.pop(0)
            elif not merge_b_list:
                (_, current) = merge_a_list.pop(0)
            elif merge_a_list[-1] < merge_b_list[-1]:
                (_, current) = merge_a_list.pop(0)
            else:
                (_, current) = merge_b_list.pop(0)
            while dom and (not self.ssa_def_dominates(dom[-1], current)):
                dom.pop()
            if dom and (not (dom[-1] in merge_a and current in merge_a)) and (not (dom[-1] in merge_b and current in merge_b)) and self.merge_nodes_interfere(current, dom[-1], parent):
                return True
            dom.append(current)
        return False

    def aggressive_coalesce_parallel_copy(self, parallel_copies, parent):
        if False:
            for i in range(10):
                print('nop')
        '\n        Try to coalesce variables each dst/src couple together from\n        @parallel_copies\n\n        @parallel_copies: a dictionary representing dst/src parallel\n        assignments.\n        @parent: Optional parent location of the phi source for liveness tests\n        '
        for (dst, src) in viewitems(parallel_copies):
            dst_merge = self.merge_state.setdefault(dst, set([dst]))
            src_merge = self.merge_state.setdefault(src, set([src]))
            if not self.merge_sets_interfere(dst_merge, src_merge, parent):
                dst_merge.update(src_merge)
                for node in dst_merge:
                    self.merge_state[node] = dst_merge

    def aggressive_coalesce_block(self):
        if False:
            print('Hello World!')
        'Try to coalesce phi var with their pre/post variables'
        ircfg = self.ssa.graph
        for irblock in viewvalues(ircfg.blocks):
            if not irblock_has_phi(irblock):
                continue
            parallel_copies = {}
            for dst in self.phi_destinations[irblock.loc_key]:
                parallel_copies[dst] = self.phi_new_var[dst]
            self.aggressive_coalesce_parallel_copy(parallel_copies, None)
            parent_to_parallel_copies = {}
            for dst in irblock[0]:
                new_var = self.phi_new_var[dst]
                for (parent, src) in self.phi_parent_sources[dst]:
                    parent_to_parallel_copies.setdefault(parent, {})[new_var] = src
            for (parent, parallel_copies) in viewitems(parent_to_parallel_copies):
                self.aggressive_coalesce_parallel_copy(parallel_copies, parent)

    def get_best_merge_set_name(self, merge_set):
        if False:
            print('Hello World!')
        '\n        For a given @merge_set, prefer an original SSA variable instead of a\n        created copy. In other case, take a random name.\n        @merge_set: set of equivalent expressions\n        '
        if not merge_set:
            raise RuntimeError('Merge set should not be empty')
        for var in merge_set:
            if var not in self.copy_vars:
                return var
        return var

    def replace_merge_sets(self):
        if False:
            return 10
        '\n        In the graph, replace all variables from merge state by their\n        representative variable\n        '
        replace = {}
        merge_sets = set()
        merge_set_to_name = {}
        for merge_set in viewvalues(self.merge_state):
            frozen_merge_set = frozenset(merge_set)
            merge_sets.add(frozen_merge_set)
            var_name = self.get_best_merge_set_name(merge_set)
            merge_set_to_name[frozen_merge_set] = var_name
        for merge_set in merge_sets:
            var_name = merge_set_to_name[merge_set]
            merge_set = list(merge_set)
            for var in merge_set:
                replace[var] = var_name
        self.ssa.graph.simplify(lambda x: x.replace_expr(replace))

    def remove_phi(self):
        if False:
            i = 10
            return i + 15
        '\n        Remove phi operators in @ifcfg\n        @ircfg: IRDiGraph instance\n        '
        for irblock in list(viewvalues(self.ssa.graph.blocks)):
            assignblks = list(irblock)
            out = {}
            for (dst, src) in viewitems(assignblks[0]):
                if src.is_op('Phi'):
                    assert set([dst]) == set(src.args)
                    continue
                out[dst] = src
            assignblks[0] = AssignBlock(out, assignblks[0].instr)
            self.ssa.graph.blocks[irblock.loc_key] = IRBlock(irblock.loc_db, irblock.loc_key, assignblks)

    def remove_assign_eq(self):
        if False:
            while True:
                i = 10
        '\n        Remove trivial expressions (a=a) in the current graph\n        '
        for irblock in list(viewvalues(self.ssa.graph.blocks)):
            assignblks = list(irblock)
            for (i, assignblk) in enumerate(assignblks):
                out = {}
                for (dst, src) in viewitems(assignblk):
                    if dst == src:
                        continue
                    out[dst] = src
                assignblks[i] = AssignBlock(out, assignblk.instr)
            self.ssa.graph.blocks[irblock.loc_key] = IRBlock(irblock.loc_db, irblock.loc_key, assignblks)