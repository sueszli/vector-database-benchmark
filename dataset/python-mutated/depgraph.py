"""Provide dependency graph"""
from functools import total_ordering
from future.utils import viewitems
from miasm.expression.expression import ExprInt, ExprLoc, ExprAssign, ExprWalk, canonize_to_exprloc
from miasm.core.graph import DiGraph
from miasm.expression.simplifications import expr_simp_explicit
from miasm.ir.symbexec import SymbolicExecutionEngine
from miasm.ir.ir import IRBlock, AssignBlock
from miasm.ir.translators import Translator
from miasm.expression.expression_helper import possible_values
try:
    import z3
except:
    pass

@total_ordering
class DependencyNode(object):
    """Node elements of a DependencyGraph

    A dependency node stands for the dependency on the @element at line number
    @line_nb in the IRblock named @loc_key, *before* the evaluation of this
    line.
    """
    __slots__ = ['_loc_key', '_element', '_line_nb', '_hash']

    def __init__(self, loc_key, element, line_nb):
        if False:
            for i in range(10):
                print('nop')
        'Create a dependency node with:\n        @loc_key: LocKey instance\n        @element: Expr instance\n        @line_nb: int\n        '
        self._loc_key = loc_key
        self._element = element
        self._line_nb = line_nb
        self._hash = hash((self._loc_key, self._element, self._line_nb))

    def __hash__(self):
        if False:
            print('Hello World!')
        'Returns a hash of @self to uniquely identify @self'
        return self._hash

    def __eq__(self, depnode):
        if False:
            while True:
                i = 10
        'Returns True if @self and @depnode are equals.'
        if not isinstance(depnode, self.__class__):
            return False
        return self.loc_key == depnode.loc_key and self.element == depnode.element and (self.line_nb == depnode.line_nb)

    def __ne__(self, depnode):
        if False:
            return 10
        return not self == depnode

    def __lt__(self, node):
        if False:
            print('Hello World!')
        'Compares @self with @node.'
        if not isinstance(node, self.__class__):
            return NotImplemented
        return (self.loc_key, self.element, self.line_nb) < (node.loc_key, node.element, node.line_nb)

    def __str__(self):
        if False:
            return 10
        'Returns a string representation of DependencyNode'
        return '<%s %s %s %s>' % (self.__class__.__name__, self.loc_key, self.element, self.line_nb)

    def __repr__(self):
        if False:
            return 10
        'Returns a string representation of DependencyNode'
        return self.__str__()

    @property
    def loc_key(self):
        if False:
            i = 10
            return i + 15
        'Name of the current IRBlock'
        return self._loc_key

    @property
    def element(self):
        if False:
            for i in range(10):
                print('nop')
        'Current tracked Expr'
        return self._element

    @property
    def line_nb(self):
        if False:
            for i in range(10):
                print('nop')
        'Line in the current IRBlock'
        return self._line_nb

class DependencyState(object):
    """
    Store intermediate depnodes states during dependencygraph analysis
    """

    def __init__(self, loc_key, pending, line_nb=None):
        if False:
            return 10
        self.loc_key = loc_key
        self.history = [loc_key]
        self.pending = {k: set(v) for (k, v) in viewitems(pending)}
        self.line_nb = line_nb
        self.links = set()
        self._graph = None

    def __repr__(self):
        if False:
            print('Hello World!')
        return '<State: %r (%r) (%r)>' % (self.loc_key, self.pending, self.links)

    def extend(self, loc_key):
        if False:
            return 10
        "Return a copy of itself, with itself in history\n        @loc_key: LocKey instance for the new DependencyState's loc_key\n        "
        new_state = self.__class__(loc_key, self.pending)
        new_state.links = set(self.links)
        new_state.history = self.history + [loc_key]
        return new_state

    def get_done_state(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns immutable object representing current state'
        return (self.loc_key, frozenset(self.links))

    def as_graph(self):
        if False:
            while True:
                i = 10
        'Generates a Digraph of dependencies'
        graph = DiGraph()
        for (node_a, node_b) in self.links:
            if not node_b:
                graph.add_node(node_a)
            else:
                graph.add_edge(node_a, node_b)
        for (parent, sons) in viewitems(self.pending):
            for son in sons:
                graph.add_edge(parent, son)
        return graph

    @property
    def graph(self):
        if False:
            return 10
        'Returns a DiGraph instance representing the DependencyGraph'
        if self._graph is None:
            self._graph = self.as_graph()
        return self._graph

    def remove_pendings(self, nodes):
        if False:
            for i in range(10):
                print('nop')
        'Remove resolved @nodes'
        for node in nodes:
            del self.pending[node]

    def add_pendings(self, future_pending):
        if False:
            i = 10
            return i + 15
        'Add @future_pending to the state'
        for (node, depnodes) in viewitems(future_pending):
            if node not in self.pending:
                self.pending[node] = depnodes
            else:
                self.pending[node].update(depnodes)

    def link_element(self, element, line_nb):
        if False:
            print('Hello World!')
        "Link element to its dependencies\n        @element: the element to link\n        @line_nb: the element's line\n        "
        depnode = DependencyNode(self.loc_key, element, line_nb)
        if not self.pending[element]:
            self.links.add((depnode, None))
        else:
            for node_son in self.pending[element]:
                self.links.add((depnode, node_son))

    def link_dependencies(self, element, line_nb, dependencies, future_pending):
        if False:
            return 10
        "Link unfollowed dependencies and create remaining pending elements.\n        @element: the element to link\n        @line_nb: the element's line\n        @dependencies: the element's dependencies\n        @future_pending: the future dependencies\n        "
        depnode = DependencyNode(self.loc_key, element, line_nb)
        for dependency in dependencies:
            if not dependency.follow:
                parent = DependencyNode(self.loc_key, dependency.element, line_nb)
                self.links.add((parent, depnode))
                continue
            future_pending.setdefault(dependency.element, set()).add(depnode)

class DependencyResult(DependencyState):
    """Container and methods for DependencyGraph results"""

    def __init__(self, ircfg, initial_state, state, inputs):
        if False:
            return 10
        super(DependencyResult, self).__init__(state.loc_key, state.pending)
        self.initial_state = initial_state
        self.history = state.history
        self.pending = state.pending
        self.line_nb = state.line_nb
        self.inputs = inputs
        self.links = state.links
        self._ircfg = ircfg
        self._has_loop = None

    @property
    def unresolved(self):
        if False:
            for i in range(10):
                print('nop')
        "Set of nodes whose dependencies weren't found"
        return set((element for element in self.pending if element != self._ircfg.IRDst))

    @property
    def relevant_nodes(self):
        if False:
            return 10
        'Set of nodes directly and indirectly influencing inputs'
        output = set()
        for (node_a, node_b) in self.links:
            output.add(node_a)
            if node_b is not None:
                output.add(node_b)
        return output

    @property
    def relevant_loc_keys(self):
        if False:
            for i in range(10):
                print('nop')
        'List of loc_keys containing nodes influencing inputs.\n        The history order is preserved.'
        used_loc_keys = set((depnode.loc_key for depnode in self.relevant_nodes))
        output = []
        for loc_key in self.history:
            if loc_key in used_loc_keys:
                output.append(loc_key)
        return output

    @property
    def has_loop(self):
        if False:
            while True:
                i = 10
        'True iff there is at least one data dependencies cycle (regarding\n        the associated depgraph)'
        if self._has_loop is None:
            self._has_loop = self.graph.has_loop()
        return self._has_loop

    def irblock_slice(self, irb, max_line=None):
        if False:
            return 10
        'Slice of the dependency nodes on the irblock @irb\n        @irb: irbloc instance\n        '
        assignblks = []
        line2elements = {}
        for depnode in self.relevant_nodes:
            if depnode.loc_key != irb.loc_key:
                continue
            line2elements.setdefault(depnode.line_nb, set()).add(depnode.element)
        for (line_nb, elements) in sorted(viewitems(line2elements)):
            if max_line is not None and line_nb >= max_line:
                break
            assignmnts = {}
            for element in elements:
                if element in irb[line_nb]:
                    assignmnts[element] = irb[line_nb][element]
            assignblks.append(AssignBlock(assignmnts))
        return IRBlock(irb.loc_db, irb.loc_key, assignblks)

    def emul(self, lifter, ctx=None, step=False):
        if False:
            print('Hello World!')
        "Symbolic execution of relevant nodes according to the history\n        Return the values of inputs nodes' elements\n        @lifter: Lifter instance\n        @ctx: (optional) Initial context as dictionary\n        @step: (optional) Verbose execution\n        Warning: The emulation is not sound if the inputs nodes depend on loop\n        variant.\n        "
        ctx_init = {}
        if ctx is not None:
            ctx_init.update(ctx)
        assignblks = []
        last_index = len(self.relevant_loc_keys)
        for (index, loc_key) in enumerate(reversed(self.relevant_loc_keys), 1):
            if index == last_index and loc_key == self.initial_state.loc_key:
                line_nb = self.initial_state.line_nb
            else:
                line_nb = None
            assignblks += self.irblock_slice(self._ircfg.blocks[loc_key], line_nb).assignblks
        loc_db = lifter.loc_db
        temp_loc = loc_db.get_or_create_name_location('Temp')
        symb_exec = SymbolicExecutionEngine(lifter, ctx_init)
        symb_exec.eval_updt_irblock(IRBlock(loc_db, temp_loc, assignblks), step=step)
        return {element: symb_exec.symbols[element] for element in self.inputs}

class DependencyResultImplicit(DependencyResult):
    """Stand for a result of a DependencyGraph with implicit option

    Provide path constraints using the z3 solver"""
    _solver = None
    unsat_expr = ExprAssign(ExprInt(0, 1), ExprInt(1, 1))

    def _gen_path_constraints(self, translator, expr, expected):
        if False:
            i = 10
            return i + 15
        'Generate path constraint from @expr. Handle special case with\n        generated loc_keys\n        '
        out = []
        expected = canonize_to_exprloc(self._ircfg.loc_db, expected)
        expected_is_loc_key = expected.is_loc()
        for consval in possible_values(expr):
            value = canonize_to_exprloc(self._ircfg.loc_db, consval.value)
            if expected_is_loc_key and value != expected:
                continue
            if not expected_is_loc_key and value.is_loc_key():
                continue
            conds = z3.And(*[translator.from_expr(cond.to_constraint()) for cond in consval.constraints])
            if expected != value:
                conds = z3.And(conds, translator.from_expr(ExprAssign(value, expected)))
            out.append(conds)
        if out:
            conds = z3.Or(*out)
        else:
            conds = translator.from_expr(self.unsat_expr)
        return conds

    def emul(self, lifter, ctx=None, step=False):
        if False:
            for i in range(10):
                print('nop')
        ctx_init = {}
        if ctx is not None:
            ctx_init.update(ctx)
        solver = z3.Solver()
        symb_exec = SymbolicExecutionEngine(lifter, ctx_init)
        history = self.history[::-1]
        history_size = len(history)
        translator = Translator.to_language('z3')
        size = self._ircfg.IRDst.size
        for (hist_nb, loc_key) in enumerate(history, 1):
            if hist_nb == history_size and loc_key == self.initial_state.loc_key:
                line_nb = self.initial_state.line_nb
            else:
                line_nb = None
            irb = self.irblock_slice(self._ircfg.blocks[loc_key], line_nb)
            dst = symb_exec.eval_updt_irblock(irb, step=step)
            if hist_nb < history_size:
                next_loc_key = history[hist_nb]
                expected = symb_exec.eval_expr(ExprLoc(next_loc_key, size))
                solver.add(self._gen_path_constraints(translator, dst, expected))
        self._solver = solver
        return {element: symb_exec.eval_expr(element) for element in self.inputs}

    @property
    def is_satisfiable(self):
        if False:
            return 10
        "Return True iff the solution path admits at least one solution\n        PRE: 'emul'\n        "
        return self._solver.check() == z3.sat

    @property
    def constraints(self):
        if False:
            print('Hello World!')
        'If satisfiable, return a valid solution as a Z3 Model instance'
        if not self.is_satisfiable:
            raise ValueError('Unsatisfiable')
        return self._solver.model()

class FollowExpr(object):
    """Stand for an element (expression, depnode, ...) to follow or not"""
    __slots__ = ['follow', 'element']

    def __init__(self, follow, element):
        if False:
            while True:
                i = 10
        self.follow = follow
        self.element = element

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '%s(%r, %r)' % (self.__class__.__name__, self.follow, self.element)

    @staticmethod
    def to_depnodes(follow_exprs, loc_key, line):
        if False:
            while True:
                i = 10
        'Build a set of FollowExpr(DependencyNode) from the @follow_exprs set\n        of FollowExpr\n        @follow_exprs: set of FollowExpr\n        @loc_key: LocKey instance\n        @line: integer\n        '
        dependencies = set()
        for follow_expr in follow_exprs:
            dependencies.add(FollowExpr(follow_expr.follow, DependencyNode(loc_key, follow_expr.element, line)))
        return dependencies

    @staticmethod
    def extract_depnodes(follow_exprs, only_follow=False):
        if False:
            print('Hello World!')
        'Extract depnodes from a set of FollowExpr(Depnodes)\n        @only_follow: (optional) extract only elements to follow'
        return set((follow_expr.element for follow_expr in follow_exprs if not only_follow or follow_expr.follow))

class FilterExprSources(ExprWalk):
    """
    Walk Expression to find sources to track
    @follow_mem: (optional) Track memory syntactically
    @follow_call: (optional) Track through "call"
    """

    def __init__(self, follow_mem, follow_call):
        if False:
            return 10
        super(FilterExprSources, self).__init__(lambda x: None)
        self.follow_mem = follow_mem
        self.follow_call = follow_call
        self.nofollow = set()
        self.follow = set()

    def visit(self, expr, *args, **kwargs):
        if False:
            return 10
        if expr in self.cache:
            return None
        ret = self.visit_inner(expr, *args, **kwargs)
        self.cache.add(expr)
        return ret

    def visit_inner(self, expr, *args, **kwargs):
        if False:
            while True:
                i = 10
        if expr.is_id():
            self.follow.add(expr)
        elif expr.is_int():
            self.nofollow.add(expr)
        elif expr.is_loc():
            self.nofollow.add(expr)
        elif expr.is_mem():
            if self.follow_mem:
                self.follow.add(expr)
            else:
                self.nofollow.add(expr)
                return None
        elif expr.is_function_call():
            if self.follow_call:
                self.follow.add(expr)
            else:
                self.nofollow.add(expr)
                return None
        ret = super(FilterExprSources, self).visit(expr, *args, **kwargs)
        return ret

class DependencyGraph(object):
    """Implementation of a dependency graph

    A dependency graph contains DependencyNode as nodes. The oriented edges
    stand for a dependency.
    The dependency graph is made of the lines of a group of IRblock
    *explicitly* or *implicitly* involved in the equation of given element.
    """

    def __init__(self, ircfg, implicit=False, apply_simp=True, follow_mem=True, follow_call=True):
        if False:
            while True:
                i = 10
        'Create a DependencyGraph linked to @ircfg\n\n        @ircfg: IRCFG instance\n        @implicit: (optional) Track IRDst for each block in the resulting path\n\n        Following arguments define filters used to generate dependencies\n        @apply_simp: (optional) Apply expr_simp_explicit\n        @follow_mem: (optional) Track memory syntactically\n        @follow_call: (optional) Track through "call"\n        '
        self._ircfg = ircfg
        self._implicit = implicit
        self._cb_follow = []
        if apply_simp:
            self._cb_follow.append(self._follow_simp_expr)
        self._cb_follow.append(lambda exprs: self.do_follow(exprs, follow_mem, follow_call))

    @staticmethod
    def do_follow(exprs, follow_mem, follow_call):
        if False:
            while True:
                i = 10
        visitor = FilterExprSources(follow_mem, follow_call)
        for expr in exprs:
            visitor.visit(expr)
        return (visitor.follow, visitor.nofollow)

    @staticmethod
    def _follow_simp_expr(exprs):
        if False:
            print('Hello World!')
        'Simplify expression so avoid tracking useless elements,\n        as: XOR EAX, EAX\n        '
        follow = set()
        for expr in exprs:
            follow.add(expr_simp_explicit(expr))
        return (follow, set())

    def _follow_apply_cb(self, expr):
        if False:
            while True:
                i = 10
        'Apply callback functions to @expr\n        @expr : FollowExpr instance'
        follow = set([expr])
        nofollow = set()
        for callback in self._cb_follow:
            (follow, nofollow_tmp) = callback(follow)
            nofollow.update(nofollow_tmp)
        out = set((FollowExpr(True, expr) for expr in follow))
        out.update(set((FollowExpr(False, expr) for expr in nofollow)))
        return out

    def _track_exprs(self, state, assignblk, line_nb):
        if False:
            return 10
        'Track pending expression in an assignblock'
        future_pending = {}
        node_resolved = set()
        for (dst, src) in viewitems(assignblk):
            if dst not in state.pending:
                continue
            if dst == self._ircfg.IRDst and (not self._implicit):
                continue
            assert dst not in node_resolved
            node_resolved.add(dst)
            dependencies = self._follow_apply_cb(src)
            state.link_element(dst, line_nb)
            state.link_dependencies(dst, line_nb, dependencies, future_pending)
        state.remove_pendings(node_resolved)
        state.add_pendings(future_pending)

    def _compute_intrablock(self, state):
        if False:
            print('Hello World!')
        'Follow dependencies tracked in @state in the current irbloc\n        @state: instance of DependencyState'
        irb = self._ircfg.blocks[state.loc_key]
        line_nb = len(irb) if state.line_nb is None else state.line_nb
        for (cur_line_nb, assignblk) in reversed(list(enumerate(irb[:line_nb]))):
            self._track_exprs(state, assignblk, cur_line_nb)

    def get(self, loc_key, elements, line_nb, heads):
        if False:
            i = 10
            return i + 15
        'Compute the dependencies of @elements at line number @line_nb in\n        the block named @loc_key in the current IRCFG, before the execution of\n        this line. Dependency check stop if one of @heads is reached\n        @loc_key: LocKey instance\n        @element: set of Expr instances\n        @line_nb: int\n        @heads: set of LocKey instances\n        Return an iterator on DiGraph(DependencyNode)\n        '
        inputs = {element: set() for element in elements}
        initial_state = DependencyState(loc_key, inputs, line_nb)
        todo = set([initial_state])
        done = set()
        dpResultcls = DependencyResultImplicit if self._implicit else DependencyResult
        while todo:
            state = todo.pop()
            self._compute_intrablock(state)
            done_state = state.get_done_state()
            if done_state in done:
                continue
            done.add(done_state)
            if not state.pending or state.loc_key in heads or (not self._ircfg.predecessors(state.loc_key)):
                yield dpResultcls(self._ircfg, initial_state, state, elements)
                if not state.pending:
                    continue
            if self._implicit:
                state.pending[self._ircfg.IRDst] = set()
            for pred in self._ircfg.predecessors_iter(state.loc_key):
                todo.add(state.extend(pred))

    def get_from_depnodes(self, depnodes, heads):
        if False:
            print('Hello World!')
        'Alias for the get() method. Use the attributes of @depnodes as\n        argument.\n        PRE: Loc_Keys and lines of depnodes have to be equals\n        @depnodes: set of DependencyNode instances\n        @heads: set of LocKey instances\n        '
        lead = list(depnodes)[0]
        elements = set((depnode.element for depnode in depnodes))
        return self.get(lead.loc_key, elements, lead.line_nb, heads)

    def address_to_location(self, address):
        if False:
            for i in range(10):
                print('nop')
        "Helper to retrieve the .get() arguments, ie.\n        assembly address -> irblock's location key and line number\n        "
        current_loc_key = next(iter(self._ircfg.getby_offset(address)))
        assignblk_index = 0
        current_block = self._ircfg.get_block(current_loc_key)
        for (assignblk_index, assignblk) in enumerate(current_block):
            if assignblk.instr.offset == address:
                break
        else:
            return None
        return {'loc_key': current_block.loc_key, 'line_nb': assignblk_index}