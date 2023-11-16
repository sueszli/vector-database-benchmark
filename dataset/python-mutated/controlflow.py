import collections
import functools
import sys
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
NEW_BLOCKERS = frozenset(['SETUP_LOOP', 'FOR_ITER', 'SETUP_WITH', 'BEFORE_WITH'])

class CFBlock(object):

    def __init__(self, offset):
        if False:
            i = 10
            return i + 15
        self.offset = offset
        self.body = []
        self.outgoing_jumps = {}
        self.incoming_jumps = {}
        self.terminating = False

    def __repr__(self):
        if False:
            print('Hello World!')
        args = (self.offset, sorted(self.outgoing_jumps), sorted(self.incoming_jumps))
        return 'block(offset:%d, outgoing: %s, incoming: %s)' % args

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return iter(self.body)

class Loop(collections.namedtuple('Loop', ('entries', 'exits', 'header', 'body'))):
    """
    A control flow loop, as detected by a CFGraph object.
    """
    __slots__ = ()

    def __eq__(self, other):
        if False:
            print('Hello World!')
        return isinstance(other, Loop) and other.header == self.header

    def __hash__(self):
        if False:
            print('Hello World!')
        return hash(self.header)

class _DictOfContainers(collections.defaultdict):
    """A defaultdict with customized equality checks that ignore empty values.

    Non-empty value is checked by: `bool(value_item) == True`.
    """

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other, _DictOfContainers):
            mine = self._non_empty_items()
            theirs = other._non_empty_items()
            return mine == theirs
        return NotImplemented

    def __ne__(self, other):
        if False:
            i = 10
            return i + 15
        ret = self.__eq__(other)
        if ret is NotImplemented:
            return ret
        else:
            return not ret

    def _non_empty_items(self):
        if False:
            return 10
        return [(k, vs) for (k, vs) in sorted(self.items()) if vs]

class CFGraph(object):
    """
    Generic (almost) implementation of a Control Flow Graph.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        self._nodes = set()
        self._preds = _DictOfContainers(set)
        self._succs = _DictOfContainers(set)
        self._edge_data = {}
        self._entry_point = None

    def add_node(self, node):
        if False:
            print('Hello World!')
        '\n        Add *node* to the graph.  This is necessary before adding any\n        edges from/to the node.  *node* can be any hashable object.\n        '
        self._nodes.add(node)

    def add_edge(self, src, dest, data=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Add an edge from node *src* to node *dest*, with optional\n        per-edge *data*.\n        If such an edge already exists, it is replaced (duplicate edges\n        are not possible).\n        '
        if src not in self._nodes:
            raise ValueError('Cannot add edge as src node %s not in nodes %s' % (src, self._nodes))
        if dest not in self._nodes:
            raise ValueError('Cannot add edge as dest node %s not in nodes %s' % (dest, self._nodes))
        self._add_edge(src, dest, data)

    def successors(self, src):
        if False:
            return 10
        '\n        Yield (node, data) pairs representing the successors of node *src*.\n        (*data* will be None if no data was specified when adding the edge)\n        '
        for dest in self._succs[src]:
            yield (dest, self._edge_data[src, dest])

    def predecessors(self, dest):
        if False:
            i = 10
            return i + 15
        '\n        Yield (node, data) pairs representing the predecessors of node *dest*.\n        (*data* will be None if no data was specified when adding the edge)\n        '
        for src in self._preds[dest]:
            yield (src, self._edge_data[src, dest])

    def set_entry_point(self, node):
        if False:
            return 10
        '\n        Set the entry point of the graph to *node*.\n        '
        assert node in self._nodes
        self._entry_point = node

    def process(self):
        if False:
            while True:
                i = 10
        '\n        Compute essential properties of the control flow graph.  The graph\n        must have been fully populated, and its entry point specified. Other\n        graph properties are computed on-demand.\n        '
        if self._entry_point is None:
            raise RuntimeError('no entry point defined!')
        self._eliminate_dead_blocks()

    def dominators(self):
        if False:
            print('Hello World!')
        '\n        Return a dictionary of {node -> set(nodes)} mapping each node to\n        the nodes dominating it.\n\n        A node D dominates a node N when any path leading to N must go through D\n        '
        return self._doms

    def post_dominators(self):
        if False:
            print('Hello World!')
        '\n        Return a dictionary of {node -> set(nodes)} mapping each node to\n        the nodes post-dominating it.\n\n        A node P post-dominates a node N when any path starting from N must go\n        through P.\n        '
        return self._post_doms

    def immediate_dominators(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a dictionary of {node -> node} mapping each node to its\n        immediate dominator (idom).\n\n        The idom(B) is the closest strict dominator of V\n        '
        return self._idom

    def dominance_frontier(self):
        if False:
            return 10
        "\n        Return a dictionary of {node -> set(nodes)} mapping each node to\n        the nodes in its dominance frontier.\n\n        The dominance frontier _df(N) is the set of all nodes that are\n        immediate successors to blocks dominated by N but which aren't\n        strictly dominated by N\n        "
        return self._df

    def dominator_tree(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        return a dictionary of {node -> set(nodes)} mapping each node to\n        the set of nodes it immediately dominates\n\n        The domtree(B) is the closest strict set of nodes that B dominates\n        '
        return self._domtree

    @functools.cached_property
    def _exit_points(self):
        if False:
            i = 10
            return i + 15
        return self._find_exit_points()

    @functools.cached_property
    def _doms(self):
        if False:
            for i in range(10):
                print('nop')
        return self._find_dominators()

    @functools.cached_property
    def _back_edges(self):
        if False:
            while True:
                i = 10
        return self._find_back_edges()

    @functools.cached_property
    def _topo_order(self):
        if False:
            i = 10
            return i + 15
        return self._find_topo_order()

    @functools.cached_property
    def _descs(self):
        if False:
            print('Hello World!')
        return self._find_descendents()

    @functools.cached_property
    def _loops(self):
        if False:
            print('Hello World!')
        return self._find_loops()

    @functools.cached_property
    def _in_loops(self):
        if False:
            return 10
        return self._find_in_loops()

    @functools.cached_property
    def _post_doms(self):
        if False:
            print('Hello World!')
        return self._find_post_dominators()

    @functools.cached_property
    def _idom(self):
        if False:
            for i in range(10):
                print('nop')
        return self._find_immediate_dominators()

    @functools.cached_property
    def _df(self):
        if False:
            i = 10
            return i + 15
        return self._find_dominance_frontier()

    @functools.cached_property
    def _domtree(self):
        if False:
            for i in range(10):
                print('nop')
        return self._find_dominator_tree()

    def descendents(self, node):
        if False:
            return 10
        '\n        Return the set of descendents of the given *node*, in topological\n        order (ignoring back edges).\n        '
        return self._descs[node]

    def entry_point(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the entry point node.\n        '
        assert self._entry_point is not None
        return self._entry_point

    def exit_points(self):
        if False:
            while True:
                i = 10
        '\n        Return the computed set of exit nodes (may be empty).\n        '
        return self._exit_points

    def backbone(self):
        if False:
            print('Hello World!')
        "\n        Return the set of nodes constituting the graph's backbone.\n        (i.e. the nodes that every path starting from the entry point\n         must go through).  By construction, it is non-empty: it contains\n         at least the entry point.\n        "
        return self._post_doms[self._entry_point]

    def loops(self):
        if False:
            return 10
        '\n        Return a dictionary of {node -> loop} mapping each loop header\n        to the loop (a Loop instance) starting with it.\n        '
        return self._loops

    def in_loops(self, node):
        if False:
            while True:
                i = 10
        '\n        Return the list of Loop objects the *node* belongs to,\n        from innermost to outermost.\n        '
        return [self._loops[x] for x in self._in_loops.get(node, ())]

    def dead_nodes(self):
        if False:
            while True:
                i = 10
        '\n        Return the set of dead nodes (eliminated from the graph).\n        '
        return self._dead_nodes

    def nodes(self):
        if False:
            i = 10
            return i + 15
        '\n        Return the set of live nodes.\n        '
        return self._nodes

    def topo_order(self):
        if False:
            while True:
                i = 10
        '\n        Return the sequence of nodes in topological order (ignoring back\n        edges).\n        '
        return self._topo_order

    def topo_sort(self, nodes, reverse=False):
        if False:
            return 10
        "\n        Iterate over the *nodes* in topological order (ignoring back edges).\n        The sort isn't guaranteed to be stable.\n        "
        nodes = set(nodes)
        it = self._topo_order
        if reverse:
            it = reversed(it)
        for n in it:
            if n in nodes:
                yield n

    def dump(self, file=None):
        if False:
            return 10
        '\n        Dump extensive debug information.\n        '
        import pprint
        file = file or sys.stdout
        if 1:
            print('CFG adjacency lists:', file=file)
            self._dump_adj_lists(file)
        print('CFG dominators:', file=file)
        pprint.pprint(self._doms, stream=file)
        print('CFG post-dominators:', file=file)
        pprint.pprint(self._post_doms, stream=file)
        print('CFG back edges:', sorted(self._back_edges), file=file)
        print('CFG loops:', file=file)
        pprint.pprint(self._loops, stream=file)
        print('CFG node-to-loops:', file=file)
        pprint.pprint(self._in_loops, stream=file)
        print('CFG backbone:', file=file)
        pprint.pprint(self.backbone(), stream=file)

    def render_dot(self, filename='numba_cfg.dot'):
        if False:
            while True:
                i = 10
        'Render the controlflow graph with GraphViz DOT via the\n        ``graphviz`` python binding.\n\n        Returns\n        -------\n        g : graphviz.Digraph\n            Use `g.view()` to open the graph in the default PDF application.\n        '
        try:
            import graphviz as gv
        except ImportError:
            raise ImportError('The feature requires `graphviz` but it is not available. Please install with `pip install graphviz`')
        g = gv.Digraph(filename=filename)
        for n in self._nodes:
            g.node(str(n))
        for n in self._nodes:
            for edge in self._succs[n]:
                g.edge(str(n), str(edge))
        return g

    def _add_edge(self, from_, to, data=None):
        if False:
            while True:
                i = 10
        self._preds[to].add(from_)
        self._succs[from_].add(to)
        self._edge_data[from_, to] = data

    def _remove_node_edges(self, node):
        if False:
            i = 10
            return i + 15
        for succ in self._succs.pop(node, ()):
            self._preds[succ].remove(node)
            del self._edge_data[node, succ]
        for pred in self._preds.pop(node, ()):
            self._succs[pred].remove(node)
            del self._edge_data[pred, node]

    def _dfs(self, entries=None):
        if False:
            for i in range(10):
                print('nop')
        if entries is None:
            entries = (self._entry_point,)
        seen = set()
        stack = list(entries)
        while stack:
            node = stack.pop()
            if node not in seen:
                yield node
                seen.add(node)
                for succ in self._succs[node]:
                    stack.append(succ)

    def _eliminate_dead_blocks(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Eliminate all blocks not reachable from the entry point, and\n        stash them into self._dead_nodes.\n        '
        live = set()
        for node in self._dfs():
            live.add(node)
        self._dead_nodes = self._nodes - live
        self._nodes = live
        for dead in self._dead_nodes:
            self._remove_node_edges(dead)

    def _find_exit_points(self):
        if False:
            i = 10
            return i + 15
        "\n        Compute the graph's exit points.\n        "
        exit_points = set()
        for n in self._nodes:
            if not self._succs.get(n):
                exit_points.add(n)
        return exit_points

    def _find_postorder(self):
        if False:
            return 10
        succs = self._succs
        back_edges = self._back_edges
        post_order = []
        seen = set()
        post_order = []

        def dfs_rec(node):
            if False:
                while True:
                    i = 10
            if node not in seen:
                seen.add(node)
                stack.append((post_order.append, node))
                for dest in succs[node]:
                    if (node, dest) not in back_edges:
                        stack.append((dfs_rec, dest))
        stack = [(dfs_rec, self._entry_point)]
        while stack:
            (cb, data) = stack.pop()
            cb(data)
        return post_order

    def _find_immediate_dominators(self):
        if False:
            print('Hello World!')

        def intersect(u, v):
            if False:
                while True:
                    i = 10
            while u != v:
                while idx[u] < idx[v]:
                    u = idom[u]
                while idx[u] > idx[v]:
                    v = idom[v]
            return u
        entry = self._entry_point
        preds_table = self._preds
        order = self._find_postorder()
        idx = {e: i for (i, e) in enumerate(order)}
        idom = {entry: entry}
        order.pop()
        order.reverse()
        changed = True
        while changed:
            changed = False
            for u in order:
                new_idom = functools.reduce(intersect, (v for v in preds_table[u] if v in idom))
                if u not in idom or idom[u] != new_idom:
                    idom[u] = new_idom
                    changed = True
        return idom

    def _find_dominator_tree(self):
        if False:
            print('Hello World!')
        idom = self._idom
        domtree = _DictOfContainers(set)
        for (u, v) in idom.items():
            if u not in domtree:
                domtree[u] = set()
            if u != v:
                domtree[v].add(u)
        return domtree

    def _find_dominance_frontier(self):
        if False:
            print('Hello World!')
        idom = self._idom
        preds_table = self._preds
        df = {u: set() for u in idom}
        for u in idom:
            if len(preds_table[u]) < 2:
                continue
            for v in preds_table[u]:
                while v != idom[u]:
                    df[v].add(u)
                    v = idom[v]
        return df

    def _find_dominators_internal(self, post=False):
        if False:
            print('Hello World!')
        if post:
            entries = set(self._exit_points)
            preds_table = self._succs
            succs_table = self._preds
        else:
            entries = set([self._entry_point])
            preds_table = self._preds
            succs_table = self._succs
        if not entries:
            raise RuntimeError('no entry points: dominator algorithm cannot be seeded')
        doms = {}
        for e in entries:
            doms[e] = set([e])
        todo = []
        for n in self._nodes:
            if n not in entries:
                doms[n] = set(self._nodes)
                todo.append(n)
        while todo:
            n = todo.pop()
            if n in entries:
                continue
            new_doms = set([n])
            preds = preds_table[n]
            if preds:
                new_doms |= functools.reduce(set.intersection, [doms[p] for p in preds])
            if new_doms != doms[n]:
                assert len(new_doms) < len(doms[n])
                doms[n] = new_doms
                todo.extend(succs_table[n])
        return doms

    def _find_dominators(self):
        if False:
            return 10
        return self._find_dominators_internal(post=False)

    def _find_post_dominators(self):
        if False:
            while True:
                i = 10
        dummy_exit = object()
        self._exit_points.add(dummy_exit)
        for loop in self._loops.values():
            if not loop.exits:
                for b in loop.body:
                    self._add_edge(b, dummy_exit)
        pdoms = self._find_dominators_internal(post=True)
        del pdoms[dummy_exit]
        for doms in pdoms.values():
            doms.discard(dummy_exit)
        self._remove_node_edges(dummy_exit)
        self._exit_points.remove(dummy_exit)
        return pdoms

    def _find_back_edges(self, stats=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Find back edges.  An edge (src, dest) is a back edge if and\n        only if *dest* dominates *src*.\n        '
        if stats is not None:
            if not isinstance(stats, dict):
                raise TypeError(f'*stats* must be a dict; got {type(stats)}')
            stats.setdefault('iteration_count', 0)
        back_edges = set()
        stack = []
        succs_state = {}
        entry_point = self.entry_point()
        checked = set()

        def push_state(node):
            if False:
                while True:
                    i = 10
            stack.append(node)
            succs_state[node] = [dest for dest in self._succs[node]]
        push_state(entry_point)
        iter_ct = 0
        while stack:
            iter_ct += 1
            tos = stack[-1]
            tos_succs = succs_state[tos]
            if tos_succs:
                cur_node = tos_succs.pop()
                if cur_node in stack:
                    back_edges.add((tos, cur_node))
                elif cur_node not in checked:
                    push_state(cur_node)
            else:
                stack.pop()
                checked.add(tos)
        if stats is not None:
            stats['iteration_count'] += iter_ct
        return back_edges

    def _find_topo_order(self):
        if False:
            i = 10
            return i + 15
        succs = self._succs
        back_edges = self._back_edges
        post_order = []
        seen = set()

        def _dfs_rec(node):
            if False:
                for i in range(10):
                    print('nop')
            if node not in seen:
                seen.add(node)
                for dest in succs[node]:
                    if (node, dest) not in back_edges:
                        _dfs_rec(dest)
                post_order.append(node)
        _dfs_rec(self._entry_point)
        post_order.reverse()
        return post_order

    def _find_descendents(self):
        if False:
            for i in range(10):
                print('nop')
        descs = {}
        for node in reversed(self._topo_order):
            descs[node] = node_descs = set()
            for succ in self._succs[node]:
                if (node, succ) not in self._back_edges:
                    node_descs.add(succ)
                    node_descs.update(descs[succ])
        return descs

    def _find_loops(self):
        if False:
            while True:
                i = 10
        "\n        Find the loops defined by the graph's back edges.\n        "
        bodies = {}
        for (src, dest) in self._back_edges:
            header = dest
            body = set([header])
            queue = [src]
            while queue:
                n = queue.pop()
                if n not in body:
                    body.add(n)
                    queue.extend(self._preds[n])
            if header in bodies:
                bodies[header].update(body)
            else:
                bodies[header] = body
        loops = {}
        for (header, body) in bodies.items():
            entries = set()
            exits = set()
            for n in body:
                entries.update(self._preds[n] - body)
                exits.update(self._succs[n] - body)
            loop = Loop(header=header, body=body, entries=entries, exits=exits)
            loops[header] = loop
        return loops

    def _find_in_loops(self):
        if False:
            return 10
        loops = self._loops
        in_loops = dict(((n, []) for n in self._nodes))
        for loop in sorted(loops.values(), key=lambda loop: len(loop.body)):
            for n in loop.body:
                in_loops[n].append(loop.header)
        return in_loops

    def _dump_adj_lists(self, file):
        if False:
            i = 10
            return i + 15
        adj_lists = dict(((src, sorted(list(dests))) for (src, dests) in self._succs.items()))
        import pprint
        pprint.pprint(adj_lists, stream=file)

    def __eq__(self, other):
        if False:
            print('Hello World!')
        if not isinstance(other, CFGraph):
            return NotImplemented
        for x in ['_nodes', '_edge_data', '_entry_point', '_preds', '_succs']:
            this = getattr(self, x, None)
            that = getattr(other, x, None)
            if this != that:
                return False
        return True

    def __ne__(self, other):
        if False:
            while True:
                i = 10
        return not self.__eq__(other)

class ControlFlowAnalysis(object):
    """
    Attributes
    ----------
    - bytecode

    - blocks

    - blockseq

    - doms: dict of set
        Dominators

    - backbone: set of block offsets
        The set of block that is common to all possible code path.

    """

    def __init__(self, bytecode):
        if False:
            for i in range(10):
                print('nop')
        self.bytecode = bytecode
        self.blocks = {}
        self.liveblocks = {}
        self.blockseq = []
        self.doms = None
        self.backbone = None
        self._force_new_block = True
        self._curblock = None
        self._blockstack = []
        self._loops = []
        self._withs = []

    def iterblocks(self):
        if False:
            while True:
                i = 10
        '\n        Return all blocks in sequence of occurrence\n        '
        for i in self.blockseq:
            yield self.blocks[i]

    def iterliveblocks(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return all live blocks in sequence of occurrence\n        '
        for i in self.blockseq:
            if i in self.liveblocks:
                yield self.blocks[i]

    def incoming_blocks(self, block):
        if False:
            print('Hello World!')
        '\n        Yield (incoming block, number of stack pops) pairs for *block*.\n        '
        for (i, pops) in block.incoming_jumps.items():
            if i in self.liveblocks:
                yield (self.blocks[i], pops)

    def dump(self, file=None):
        if False:
            for i in range(10):
                print('nop')
        self.graph.dump(file=None)

    def run(self):
        if False:
            return 10
        for inst in self._iter_inst():
            fname = 'op_%s' % inst.opname
            fn = getattr(self, fname, None)
            if fn is not None:
                fn(inst)
            elif inst.is_jump:
                l = Loc(self.bytecode.func_id.filename, inst.lineno)
                if inst.opname in {'SETUP_FINALLY'}:
                    msg = "'try' block not supported until python3.7 or later"
                else:
                    msg = 'Use of unsupported opcode (%s) found' % inst.opname
                raise UnsupportedError(msg, loc=l)
            else:
                pass
        for (cur, nxt) in zip(self.blockseq, self.blockseq[1:]):
            blk = self.blocks[cur]
            if not blk.outgoing_jumps and (not blk.terminating):
                blk.outgoing_jumps[nxt] = 0
        graph = CFGraph()
        for b in self.blocks:
            graph.add_node(b)
        for b in self.blocks.values():
            for (out, pops) in b.outgoing_jumps.items():
                graph.add_edge(b.offset, out, pops)
        graph.set_entry_point(min(self.blocks))
        graph.process()
        self.graph = graph
        for b in self.blocks.values():
            for (out, pops) in b.outgoing_jumps.items():
                self.blocks[out].incoming_jumps[b.offset] = pops
        self.liveblocks = dict(((i, self.blocks[i]) for i in self.graph.nodes()))
        for lastblk in reversed(self.blockseq):
            if lastblk in self.liveblocks:
                break
        else:
            raise AssertionError('No live block that exits!?')
        backbone = self.graph.backbone()
        inloopblocks = set()
        for b in self.blocks.keys():
            if self.graph.in_loops(b):
                inloopblocks.add(b)
        self.backbone = backbone - inloopblocks

    def jump(self, target, pops=0):
        if False:
            return 10
        '\n        Register a jump (conditional or not) to *target* offset.\n        *pops* is the number of stack pops implied by the jump (default 0).\n        '
        self._curblock.outgoing_jumps[target] = pops

    def _iter_inst(self):
        if False:
            while True:
                i = 10
        for inst in self.bytecode:
            if self._use_new_block(inst):
                self._guard_with_as(inst)
                self._start_new_block(inst)
            self._curblock.body.append(inst.offset)
            yield inst

    def _use_new_block(self, inst):
        if False:
            i = 10
            return i + 15
        if inst.offset in self.bytecode.labels:
            res = True
        elif inst.opname in NEW_BLOCKERS:
            res = True
        else:
            res = self._force_new_block
        self._force_new_block = False
        return res

    def _start_new_block(self, inst):
        if False:
            while True:
                i = 10
        self._curblock = CFBlock(inst.offset)
        self.blocks[inst.offset] = self._curblock
        self.blockseq.append(inst.offset)

    def _guard_with_as(self, current_inst):
        if False:
            while True:
                i = 10
        "Checks if the next instruction after a SETUP_WITH is something other\n        than a POP_TOP, if it is something else it'll be some sort of store\n        which is not supported (this corresponds to `with CTXMGR as VAR(S)`)."
        if current_inst.opname == 'SETUP_WITH':
            next_op = self.bytecode[current_inst.next].opname
            if next_op != 'POP_TOP':
                msg = "The 'with (context manager) as (variable):' construct is not supported."
                raise UnsupportedError(msg)

    def op_SETUP_LOOP(self, inst):
        if False:
            print('Hello World!')
        end = inst.get_jump_target()
        self._blockstack.append(end)
        self._loops.append((inst.offset, end))
        self.jump(inst.next)
        self._force_new_block = True

    def op_SETUP_WITH(self, inst):
        if False:
            return 10
        end = inst.get_jump_target()
        self._blockstack.append(end)
        self._withs.append((inst.offset, end))
        self.jump(inst.next)
        self._force_new_block = True

    def op_POP_BLOCK(self, inst):
        if False:
            return 10
        self._blockstack.pop()

    def op_FOR_ITER(self, inst):
        if False:
            while True:
                i = 10
        self.jump(inst.get_jump_target())
        self.jump(inst.next)
        self._force_new_block = True

    def _op_ABSOLUTE_JUMP_IF(self, inst):
        if False:
            return 10
        self.jump(inst.get_jump_target())
        self.jump(inst.next)
        self._force_new_block = True
    op_POP_JUMP_IF_FALSE = _op_ABSOLUTE_JUMP_IF
    op_POP_JUMP_IF_TRUE = _op_ABSOLUTE_JUMP_IF
    op_JUMP_IF_FALSE = _op_ABSOLUTE_JUMP_IF
    op_JUMP_IF_TRUE = _op_ABSOLUTE_JUMP_IF
    op_POP_JUMP_FORWARD_IF_FALSE = _op_ABSOLUTE_JUMP_IF
    op_POP_JUMP_BACKWARD_IF_FALSE = _op_ABSOLUTE_JUMP_IF
    op_POP_JUMP_FORWARD_IF_TRUE = _op_ABSOLUTE_JUMP_IF
    op_POP_JUMP_BACKWARD_IF_TRUE = _op_ABSOLUTE_JUMP_IF

    def _op_ABSOLUTE_JUMP_OR_POP(self, inst):
        if False:
            i = 10
            return i + 15
        self.jump(inst.get_jump_target())
        self.jump(inst.next, pops=1)
        self._force_new_block = True
    op_JUMP_IF_FALSE_OR_POP = _op_ABSOLUTE_JUMP_OR_POP
    op_JUMP_IF_TRUE_OR_POP = _op_ABSOLUTE_JUMP_OR_POP

    def op_JUMP_ABSOLUTE(self, inst):
        if False:
            print('Hello World!')
        self.jump(inst.get_jump_target())
        self._force_new_block = True

    def op_JUMP_FORWARD(self, inst):
        if False:
            for i in range(10):
                print('nop')
        self.jump(inst.get_jump_target())
        self._force_new_block = True
    op_JUMP_BACKWARD = op_JUMP_FORWARD

    def op_RETURN_VALUE(self, inst):
        if False:
            while True:
                i = 10
        self._curblock.terminating = True
        self._force_new_block = True

    def op_RAISE_VARARGS(self, inst):
        if False:
            return 10
        self._curblock.terminating = True
        self._force_new_block = True

    def op_BREAK_LOOP(self, inst):
        if False:
            print('Hello World!')
        self.jump(self._blockstack[-1])
        self._force_new_block = True