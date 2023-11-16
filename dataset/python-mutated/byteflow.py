"""
Implement python 3.8+ bytecode analysis
"""
import dis
from pprint import pformat
import logging
from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from numba.core.utils import UniqueDict, PYVERSION, ALL_BINOPS_TO_OPERATORS
from numba.core.controlflow import NEW_BLOCKERS, CFGraph
from numba.core.ir import Loc
from numba.core.errors import UnsupportedError
_logger = logging.getLogger(__name__)
_EXCEPT_STACK_OFFSET = 6
_FINALLY_POP = _EXCEPT_STACK_OFFSET
_NO_RAISE_OPS = frozenset({'LOAD_CONST', 'NOP', 'LOAD_DEREF', 'PRECALL'})

@total_ordering
class BlockKind(object):
    """Kinds of block to make related code safer than just `str`.
    """
    _members = frozenset({'LOOP', 'TRY', 'EXCEPT', 'FINALLY', 'WITH', 'WITH_FINALLY'})

    def __init__(self, value):
        if False:
            print('Hello World!')
        assert value in self._members
        self._value = value

    def __hash__(self):
        if False:
            print('Hello World!')
        return hash((type(self), self._value))

    def __lt__(self, other):
        if False:
            while True:
                i = 10
        if isinstance(other, BlockKind):
            return self._value < other._value
        else:
            raise TypeError('cannot compare to {!r}'.format(type(other)))

    def __eq__(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, BlockKind):
            return self._value == other._value
        else:
            raise TypeError('cannot compare to {!r}'.format(type(other)))

    def __repr__(self):
        if False:
            while True:
                i = 10
        return 'BlockKind({})'.format(self._value)

class _lazy_pformat(object):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return pformat(*self.args, **self.kwargs)

class Flow(object):
    """Data+Control Flow analysis.

    Simulate execution to recover dataflow and controlflow information.
    """

    def __init__(self, bytecode):
        if False:
            print('Hello World!')
        _logger.debug('bytecode dump:\n%s', bytecode.dump())
        self._bytecode = bytecode
        self.block_infos = UniqueDict()

    def run(self):
        if False:
            print('Hello World!')
        'Run a trace over the bytecode over all reachable path.\n\n        The trace starts at bytecode offset 0 and gathers stack and control-\n        flow information by partially interpreting each bytecode.\n        Each ``State`` instance in the trace corresponds to a basic-block.\n        The State instances forks when a jump instruction is encountered.\n        A newly forked state is then added to the list of pending states.\n        The trace ends when there are no more pending states.\n        '
        firststate = State(bytecode=self._bytecode, pc=0, nstack=0, blockstack=())
        runner = TraceRunner(debug_filename=self._bytecode.func_id.filename)
        runner.pending.append(firststate)
        first_encounter = UniqueDict()
        while runner.pending:
            _logger.debug('pending: %s', runner.pending)
            state = runner.pending.popleft()
            if state not in runner.finished:
                _logger.debug('stack: %s', state._stack)
                _logger.debug('state.pc_initial: %s', state)
                first_encounter[state.pc_initial] = state
                while True:
                    runner.dispatch(state)
                    if state.has_terminated():
                        break
                    else:
                        if self._run_handle_exception(runner, state):
                            break
                        if self._is_implicit_new_block(state):
                            self._guard_with_as(state)
                            state.split_new_block()
                            break
                _logger.debug('end state. edges=%s', state.outgoing_edges)
                runner.finished.add(state)
                out_states = state.get_outgoing_states()
                runner.pending.extend(out_states)
        self._build_cfg(runner.finished)
        self._prune_phis(runner)
        for state in sorted(runner.finished, key=lambda x: x.pc_initial):
            self.block_infos[state.pc_initial] = si = adapt_state_infos(state)
            _logger.debug('block_infos %s:\n%s', state, si)
    if PYVERSION == (3, 11):

        def _run_handle_exception(self, runner, state):
            if False:
                return 10
            if not state.in_with() and (state.has_active_try() and state.get_inst().opname not in _NO_RAISE_OPS):
                state.fork(pc=state.get_inst().next)
                runner._adjust_except_stack(state)
                return True
            else:
                state.advance_pc()
                if not state.in_with() and state.is_in_exception():
                    _logger.debug('3.11 exception %s PC=%s', state.get_exception(), state._pc)
                    eh = state.get_exception()
                    eh_top = state.get_top_block('TRY')
                    if eh_top and eh_top['end'] == eh.target:
                        eh_block = None
                    else:
                        eh_block = state.make_block('TRY', end=eh.target)
                        eh_block['end_offset'] = eh.end
                        eh_block['stack_depth'] = eh.depth
                        eh_block['push_lasti'] = eh.lasti
                        state.fork(pc=state._pc, extra_block=eh_block)
                        return True
    elif PYVERSION < (3, 11):

        def _run_handle_exception(self, runner, state):
            if False:
                print('Hello World!')
            if state.has_active_try() and state.get_inst().opname not in _NO_RAISE_OPS:
                state.fork(pc=state.get_inst().next)
                tryblk = state.get_top_block('TRY')
                state.pop_block_and_above(tryblk)
                nstack = state.stack_depth
                kwargs = {}
                if nstack > tryblk['entry_stack']:
                    kwargs['npop'] = nstack - tryblk['entry_stack']
                handler = tryblk['handler']
                kwargs['npush'] = {BlockKind('EXCEPT'): _EXCEPT_STACK_OFFSET, BlockKind('FINALLY'): _FINALLY_POP}[handler['kind']]
                kwargs['extra_block'] = handler
                state.fork(pc=tryblk['end'], **kwargs)
                return True
            else:
                state.advance_pc()
    else:
        raise NotImplementedError(PYVERSION)

    def _build_cfg(self, all_states):
        if False:
            print('Hello World!')
        graph = CFGraph()
        for state in all_states:
            b = state.pc_initial
            graph.add_node(b)
        for state in all_states:
            for edge in state.outgoing_edges:
                graph.add_edge(state.pc_initial, edge.pc, 0)
        graph.set_entry_point(0)
        graph.process()
        self.cfgraph = graph

    def _prune_phis(self, runner):
        if False:
            for i in range(10):
                print('nop')
        _logger.debug('Prune PHIs'.center(60, '-'))

        def get_used_phis_per_state():
            if False:
                i = 10
                return i + 15
            used_phis = defaultdict(set)
            phi_set = set()
            for state in runner.finished:
                used = set(state._used_regs)
                phis = set(state._phis)
                used_phis[state] |= phis & used
                phi_set |= phis
            return (used_phis, phi_set)

        def find_use_defs():
            if False:
                print('Hello World!')
            defmap = {}
            phismap = defaultdict(set)
            for state in runner.finished:
                for (phi, rhs) in state._outgoing_phis.items():
                    if rhs not in phi_set:
                        defmap[phi] = state
                    phismap[phi].add((rhs, state))
            _logger.debug('defmap: %s', _lazy_pformat(defmap))
            _logger.debug('phismap: %s', _lazy_pformat(phismap))
            return (defmap, phismap)

        def propagate_phi_map(phismap):
            if False:
                for i in range(10):
                    print('nop')
            'An iterative dataflow algorithm to find the definition\n            (the source) of each PHI node.\n            '
            blacklist = defaultdict(set)
            while True:
                changing = False
                for (phi, defsites) in sorted(list(phismap.items())):
                    for (rhs, state) in sorted(list(defsites)):
                        if rhs in phi_set:
                            defsites |= phismap[rhs]
                            blacklist[phi].add((rhs, state))
                    to_remove = blacklist[phi]
                    if to_remove & defsites:
                        defsites -= to_remove
                        changing = True
                _logger.debug('changing phismap: %s', _lazy_pformat(phismap))
                if not changing:
                    break

        def apply_changes(used_phis, phismap):
            if False:
                for i in range(10):
                    print('nop')
            keep = {}
            for (state, used_set) in used_phis.items():
                for phi in used_set:
                    keep[phi] = phismap[phi]
            _logger.debug('keep phismap: %s', _lazy_pformat(keep))
            new_out = defaultdict(dict)
            for phi in keep:
                for (rhs, state) in keep[phi]:
                    new_out[state][phi] = rhs
            _logger.debug('new_out: %s', _lazy_pformat(new_out))
            for state in runner.finished:
                state._outgoing_phis.clear()
                state._outgoing_phis.update(new_out[state])
        (used_phis, phi_set) = get_used_phis_per_state()
        _logger.debug('Used_phis: %s', _lazy_pformat(used_phis))
        (defmap, phismap) = find_use_defs()
        propagate_phi_map(phismap)
        apply_changes(used_phis, phismap)
        _logger.debug('DONE Prune PHIs'.center(60, '-'))

    def _is_implicit_new_block(self, state):
        if False:
            i = 10
            return i + 15
        inst = state.get_inst()
        if inst.offset in self._bytecode.labels:
            return True
        elif inst.opname in NEW_BLOCKERS:
            return True
        else:
            return False

    def _guard_with_as(self, state):
        if False:
            i = 10
            return i + 15
        "Checks if the next instruction after a SETUP_WITH is something other\n        than a POP_TOP, if it is something else it'll be some sort of store\n        which is not supported (this corresponds to `with CTXMGR as VAR(S)`)."
        current_inst = state.get_inst()
        if current_inst.opname in {'SETUP_WITH', 'BEFORE_WITH'}:
            next_op = self._bytecode[current_inst.next].opname
            if next_op != 'POP_TOP':
                msg = "The 'with (context manager) as (variable):' construct is not supported."
                raise UnsupportedError(msg)

def _is_null_temp_reg(reg):
    if False:
        while True:
            i = 10
    return reg.startswith('$null$')

class TraceRunner(object):
    """Trace runner contains the states for the trace and the opcode dispatch.
    """

    def __init__(self, debug_filename):
        if False:
            return 10
        self.debug_filename = debug_filename
        self.pending = deque()
        self.finished = set()

    def get_debug_loc(self, lineno):
        if False:
            i = 10
            return i + 15
        return Loc(self.debug_filename, lineno)

    def dispatch(self, state):
        if False:
            print('Hello World!')
        if PYVERSION > (3, 11):
            raise NotImplementedError(PYVERSION)
        elif PYVERSION == (3, 11) and state._blockstack:
            state: State
            while state._blockstack:
                topblk = state._blockstack[-1]
                blk_end = topblk['end']
                if blk_end is not None and blk_end <= state.pc_initial:
                    state._blockstack.pop()
                else:
                    break
        inst = state.get_inst()
        if inst.opname != 'CACHE':
            _logger.debug('dispatch pc=%s, inst=%s', state._pc, inst)
            _logger.debug('stack %s', state._stack)
        fn = getattr(self, 'op_{}'.format(inst.opname), None)
        if fn is not None:
            fn(state, inst)
        else:
            msg = 'Use of unsupported opcode (%s) found' % inst.opname
            raise UnsupportedError(msg, loc=self.get_debug_loc(inst.lineno))

    def _adjust_except_stack(self, state):
        if False:
            while True:
                i = 10
        '\n        Adjust stack when entering an exception handler to match expectation\n        by the bytecode.\n        '
        tryblk = state.get_top_block('TRY')
        state.pop_block_and_above(tryblk)
        nstack = state.stack_depth
        kwargs = {}
        expected_depth = tryblk['stack_depth']
        if nstack > expected_depth:
            kwargs['npop'] = nstack - expected_depth
        extra_stack = 1
        if tryblk['push_lasti']:
            extra_stack += 1
        kwargs['npush'] = extra_stack
        state.fork(pc=tryblk['end'], **kwargs)

    def op_NOP(self, state, inst):
        if False:
            while True:
                i = 10
        state.append(inst)

    def op_RESUME(self, state, inst):
        if False:
            print('Hello World!')
        state.append(inst)

    def op_CACHE(self, state, inst):
        if False:
            for i in range(10):
                print('nop')
        state.append(inst)

    def op_PRECALL(self, state, inst):
        if False:
            while True:
                i = 10
        state.append(inst)

    def op_PUSH_NULL(self, state, inst):
        if False:
            print('Hello World!')
        state.push(state.make_null())
        state.append(inst)

    def op_RETURN_GENERATOR(self, state, inst):
        if False:
            print('Hello World!')
        state.push(state.make_temp())
        state.append(inst)

    def op_FORMAT_VALUE(self, state, inst):
        if False:
            for i in range(10):
                print('nop')
        '\n        FORMAT_VALUE(flags): flags argument specifies format spec which is\n        not supported yet. Currently, we just call str() on the value.\n        Pops a value from stack and pushes results back.\n        Required for supporting f-strings.\n        https://docs.python.org/3/library/dis.html#opcode-FORMAT_VALUE\n        '
        if inst.arg != 0:
            msg = 'format spec in f-strings not supported yet'
            raise UnsupportedError(msg, loc=self.get_debug_loc(inst.lineno))
        value = state.pop()
        strvar = state.make_temp()
        res = state.make_temp()
        state.append(inst, value=value, res=res, strvar=strvar)
        state.push(res)

    def op_BUILD_STRING(self, state, inst):
        if False:
            print('Hello World!')
        '\n        BUILD_STRING(count): Concatenates count strings from the stack and\n        pushes the resulting string onto the stack.\n        Required for supporting f-strings.\n        https://docs.python.org/3/library/dis.html#opcode-BUILD_STRING\n        '
        count = inst.arg
        strings = list(reversed([state.pop() for _ in range(count)]))
        if count == 0:
            tmps = [state.make_temp()]
        else:
            tmps = [state.make_temp() for _ in range(count - 1)]
        state.append(inst, strings=strings, tmps=tmps)
        state.push(tmps[-1])

    def op_POP_TOP(self, state, inst):
        if False:
            return 10
        state.pop()
    if PYVERSION == (3, 11):

        def op_LOAD_GLOBAL(self, state, inst):
            if False:
                i = 10
                return i + 15
            res = state.make_temp()
            idx = inst.arg >> 1
            state.append(inst, idx=idx, res=res)
            if inst.arg & 1:
                state.push(state.make_null())
            state.push(res)
    elif PYVERSION < (3, 11):

        def op_LOAD_GLOBAL(self, state, inst):
            if False:
                for i in range(10):
                    print('nop')
            res = state.make_temp()
            state.append(inst, res=res)
            state.push(res)
    else:
        raise NotImplementedError(PYVERSION)

    def op_COPY_FREE_VARS(self, state, inst):
        if False:
            print('Hello World!')
        state.append(inst)

    def op_MAKE_CELL(self, state, inst):
        if False:
            for i in range(10):
                print('nop')
        state.append(inst)

    def op_LOAD_DEREF(self, state, inst):
        if False:
            i = 10
            return i + 15
        res = state.make_temp()
        state.append(inst, res=res)
        state.push(res)

    def op_LOAD_CONST(self, state, inst):
        if False:
            for i in range(10):
                print('nop')
        res = state.make_temp('const')
        state.push(res)
        state.append(inst, res=res)

    def op_LOAD_ATTR(self, state, inst):
        if False:
            while True:
                i = 10
        item = state.pop()
        res = state.make_temp()
        state.append(inst, item=item, res=res)
        state.push(res)

    def op_LOAD_FAST(self, state, inst):
        if False:
            while True:
                i = 10
        name = state.get_varname(inst)
        res = state.make_temp(name)
        state.append(inst, res=res)
        state.push(res)

    def op_DELETE_FAST(self, state, inst):
        if False:
            return 10
        state.append(inst)

    def op_DELETE_ATTR(self, state, inst):
        if False:
            print('Hello World!')
        target = state.pop()
        state.append(inst, target=target)

    def op_STORE_ATTR(self, state, inst):
        if False:
            for i in range(10):
                print('nop')
        target = state.pop()
        value = state.pop()
        state.append(inst, target=target, value=value)

    def op_STORE_DEREF(self, state, inst):
        if False:
            while True:
                i = 10
        value = state.pop()
        state.append(inst, value=value)

    def op_STORE_FAST(self, state, inst):
        if False:
            while True:
                i = 10
        value = state.pop()
        state.append(inst, value=value)

    def op_SLICE_1(self, state, inst):
        if False:
            for i in range(10):
                print('nop')
        '\n        TOS = TOS1[TOS:]\n        '
        tos = state.pop()
        tos1 = state.pop()
        res = state.make_temp()
        slicevar = state.make_temp()
        indexvar = state.make_temp()
        nonevar = state.make_temp()
        state.append(inst, base=tos1, start=tos, res=res, slicevar=slicevar, indexvar=indexvar, nonevar=nonevar)
        state.push(res)

    def op_SLICE_2(self, state, inst):
        if False:
            for i in range(10):
                print('nop')
        '\n        TOS = TOS1[:TOS]\n        '
        tos = state.pop()
        tos1 = state.pop()
        res = state.make_temp()
        slicevar = state.make_temp()
        indexvar = state.make_temp()
        nonevar = state.make_temp()
        state.append(inst, base=tos1, stop=tos, res=res, slicevar=slicevar, indexvar=indexvar, nonevar=nonevar)
        state.push(res)

    def op_SLICE_3(self, state, inst):
        if False:
            for i in range(10):
                print('nop')
        '\n        TOS = TOS2[TOS1:TOS]\n        '
        tos = state.pop()
        tos1 = state.pop()
        tos2 = state.pop()
        res = state.make_temp()
        slicevar = state.make_temp()
        indexvar = state.make_temp()
        state.append(inst, base=tos2, start=tos1, stop=tos, res=res, slicevar=slicevar, indexvar=indexvar)
        state.push(res)

    def op_STORE_SLICE_0(self, state, inst):
        if False:
            while True:
                i = 10
        '\n        TOS[:] = TOS1\n        '
        tos = state.pop()
        value = state.pop()
        slicevar = state.make_temp()
        indexvar = state.make_temp()
        nonevar = state.make_temp()
        state.append(inst, base=tos, value=value, slicevar=slicevar, indexvar=indexvar, nonevar=nonevar)

    def op_STORE_SLICE_1(self, state, inst):
        if False:
            return 10
        '\n        TOS1[TOS:] = TOS2\n        '
        tos = state.pop()
        tos1 = state.pop()
        value = state.pop()
        slicevar = state.make_temp()
        indexvar = state.make_temp()
        nonevar = state.make_temp()
        state.append(inst, base=tos1, start=tos, slicevar=slicevar, value=value, indexvar=indexvar, nonevar=nonevar)

    def op_STORE_SLICE_2(self, state, inst):
        if False:
            while True:
                i = 10
        '\n        TOS1[:TOS] = TOS2\n        '
        tos = state.pop()
        tos1 = state.pop()
        value = state.pop()
        slicevar = state.make_temp()
        indexvar = state.make_temp()
        nonevar = state.make_temp()
        state.append(inst, base=tos1, stop=tos, value=value, slicevar=slicevar, indexvar=indexvar, nonevar=nonevar)

    def op_STORE_SLICE_3(self, state, inst):
        if False:
            while True:
                i = 10
        '\n        TOS2[TOS1:TOS] = TOS3\n        '
        tos = state.pop()
        tos1 = state.pop()
        tos2 = state.pop()
        value = state.pop()
        slicevar = state.make_temp()
        indexvar = state.make_temp()
        state.append(inst, base=tos2, start=tos1, stop=tos, value=value, slicevar=slicevar, indexvar=indexvar)

    def op_DELETE_SLICE_0(self, state, inst):
        if False:
            for i in range(10):
                print('nop')
        '\n        del TOS[:]\n        '
        tos = state.pop()
        slicevar = state.make_temp()
        indexvar = state.make_temp()
        nonevar = state.make_temp()
        state.append(inst, base=tos, slicevar=slicevar, indexvar=indexvar, nonevar=nonevar)

    def op_DELETE_SLICE_1(self, state, inst):
        if False:
            print('Hello World!')
        '\n        del TOS1[TOS:]\n        '
        tos = state.pop()
        tos1 = state.pop()
        slicevar = state.make_temp()
        indexvar = state.make_temp()
        nonevar = state.make_temp()
        state.append(inst, base=tos1, start=tos, slicevar=slicevar, indexvar=indexvar, nonevar=nonevar)

    def op_DELETE_SLICE_2(self, state, inst):
        if False:
            for i in range(10):
                print('nop')
        '\n        del TOS1[:TOS]\n        '
        tos = state.pop()
        tos1 = state.pop()
        slicevar = state.make_temp()
        indexvar = state.make_temp()
        nonevar = state.make_temp()
        state.append(inst, base=tos1, stop=tos, slicevar=slicevar, indexvar=indexvar, nonevar=nonevar)

    def op_DELETE_SLICE_3(self, state, inst):
        if False:
            print('Hello World!')
        '\n        del TOS2[TOS1:TOS]\n        '
        tos = state.pop()
        tos1 = state.pop()
        tos2 = state.pop()
        slicevar = state.make_temp()
        indexvar = state.make_temp()
        state.append(inst, base=tos2, start=tos1, stop=tos, slicevar=slicevar, indexvar=indexvar)

    def op_BUILD_SLICE(self, state, inst):
        if False:
            i = 10
            return i + 15
        '\n        slice(TOS1, TOS) or slice(TOS2, TOS1, TOS)\n        '
        argc = inst.arg
        if argc == 2:
            tos = state.pop()
            tos1 = state.pop()
            start = tos1
            stop = tos
            step = None
        elif argc == 3:
            tos = state.pop()
            tos1 = state.pop()
            tos2 = state.pop()
            start = tos2
            stop = tos1
            step = tos
        else:
            raise Exception('unreachable')
        slicevar = state.make_temp()
        res = state.make_temp()
        state.append(inst, start=start, stop=stop, step=step, res=res, slicevar=slicevar)
        state.push(res)

    def _op_POP_JUMP_IF(self, state, inst):
        if False:
            i = 10
            return i + 15
        pred = state.pop()
        state.append(inst, pred=pred)
        target_inst = inst.get_jump_target()
        next_inst = inst.next
        state.fork(pc=next_inst)
        if target_inst != next_inst:
            state.fork(pc=target_inst)
    op_POP_JUMP_IF_TRUE = _op_POP_JUMP_IF
    op_POP_JUMP_IF_FALSE = _op_POP_JUMP_IF

    def _op_JUMP_IF_OR_POP(self, state, inst):
        if False:
            i = 10
            return i + 15
        pred = state.get_tos()
        state.append(inst, pred=pred)
        state.fork(pc=inst.next, npop=1)
        state.fork(pc=inst.get_jump_target())
    op_JUMP_IF_FALSE_OR_POP = _op_JUMP_IF_OR_POP
    op_JUMP_IF_TRUE_OR_POP = _op_JUMP_IF_OR_POP

    def op_POP_JUMP_FORWARD_IF_NONE(self, state, inst):
        if False:
            while True:
                i = 10
        self._op_POP_JUMP_IF(state, inst)

    def op_POP_JUMP_FORWARD_IF_NOT_NONE(self, state, inst):
        if False:
            for i in range(10):
                print('nop')
        self._op_POP_JUMP_IF(state, inst)

    def op_POP_JUMP_BACKWARD_IF_NONE(self, state, inst):
        if False:
            i = 10
            return i + 15
        self._op_POP_JUMP_IF(state, inst)

    def op_POP_JUMP_BACKWARD_IF_NOT_NONE(self, state, inst):
        if False:
            return 10
        self._op_POP_JUMP_IF(state, inst)

    def op_POP_JUMP_FORWARD_IF_FALSE(self, state, inst):
        if False:
            for i in range(10):
                print('nop')
        self._op_POP_JUMP_IF(state, inst)

    def op_POP_JUMP_FORWARD_IF_TRUE(self, state, inst):
        if False:
            print('Hello World!')
        self._op_POP_JUMP_IF(state, inst)

    def op_POP_JUMP_BACKWARD_IF_FALSE(self, state, inst):
        if False:
            for i in range(10):
                print('nop')
        self._op_POP_JUMP_IF(state, inst)

    def op_POP_JUMP_BACKWARD_IF_TRUE(self, state, inst):
        if False:
            print('Hello World!')
        self._op_POP_JUMP_IF(state, inst)

    def op_JUMP_FORWARD(self, state, inst):
        if False:
            for i in range(10):
                print('nop')
        state.append(inst)
        state.fork(pc=inst.get_jump_target())

    def op_JUMP_BACKWARD(self, state, inst):
        if False:
            print('Hello World!')
        state.append(inst)
        state.fork(pc=inst.get_jump_target())

    def op_JUMP_ABSOLUTE(self, state, inst):
        if False:
            while True:
                i = 10
        state.append(inst)
        state.fork(pc=inst.get_jump_target())

    def op_BREAK_LOOP(self, state, inst):
        if False:
            while True:
                i = 10
        end = state.get_top_block('LOOP')['end']
        state.append(inst, end=end)
        state.pop_block()
        state.fork(pc=end)

    def op_RETURN_VALUE(self, state, inst):
        if False:
            print('Hello World!')
        state.append(inst, retval=state.pop(), castval=state.make_temp())
        state.terminate()

    def op_YIELD_VALUE(self, state, inst):
        if False:
            while True:
                i = 10
        val = state.pop()
        res = state.make_temp()
        state.append(inst, value=val, res=res)
        state.push(res)
    if PYVERSION == (3, 11):

        def op_RAISE_VARARGS(self, state, inst):
            if False:
                while True:
                    i = 10
            if inst.arg == 0:
                exc = None
                if state.has_active_try():
                    raise UnsupportedError('The re-raising of an exception is not yet supported.', loc=self.get_debug_loc(inst.lineno))
            elif inst.arg == 1:
                exc = state.pop()
            else:
                raise ValueError('Multiple argument raise is not supported.')
            state.append(inst, exc=exc)
            if state.has_active_try():
                self._adjust_except_stack(state)
            else:
                state.terminate()
    elif PYVERSION < (3, 11):

        def op_RAISE_VARARGS(self, state, inst):
            if False:
                return 10
            in_exc_block = any([state.get_top_block('EXCEPT') is not None, state.get_top_block('FINALLY') is not None])
            if inst.arg == 0:
                exc = None
                if in_exc_block:
                    raise UnsupportedError('The re-raising of an exception is not yet supported.', loc=self.get_debug_loc(inst.lineno))
            elif inst.arg == 1:
                exc = state.pop()
            else:
                raise ValueError('Multiple argument raise is not supported.')
            state.append(inst, exc=exc)
            state.terminate()
    else:
        raise NotImplementedError

    def op_BEGIN_FINALLY(self, state, inst):
        if False:
            while True:
                i = 10
        temps = []
        for i in range(_EXCEPT_STACK_OFFSET):
            tmp = state.make_temp()
            temps.append(tmp)
            state.push(tmp)
        state.append(inst, temps=temps)

    def op_END_FINALLY(self, state, inst):
        if False:
            i = 10
            return i + 15
        blk = state.pop_block()
        state.reset_stack(blk['entry_stack'])

    def op_POP_FINALLY(self, state, inst):
        if False:
            while True:
                i = 10
        if inst.arg != 0:
            msg = 'Unsupported use of a bytecode related to try..finally or a with-context'
            raise UnsupportedError(msg, loc=self.get_debug_loc(inst.lineno))

    def op_CALL_FINALLY(self, state, inst):
        if False:
            for i in range(10):
                print('nop')
        pass

    def op_WITH_EXCEPT_START(self, state, inst):
        if False:
            return 10
        state.terminate()

    def op_WITH_CLEANUP_START(self, state, inst):
        if False:
            for i in range(10):
                print('nop')
        state.append(inst)

    def op_WITH_CLEANUP_FINISH(self, state, inst):
        if False:
            return 10
        state.append(inst)

    def op_SETUP_LOOP(self, state, inst):
        if False:
            i = 10
            return i + 15
        state.push_block(state.make_block(kind='LOOP', end=inst.get_jump_target()))

    def op_BEFORE_WITH(self, state, inst):
        if False:
            while True:
                i = 10
        cm = state.pop()
        yielded = state.make_temp()
        exitfn = state.make_temp(prefix='setup_with_exitfn')
        state.push(exitfn)
        state.push(yielded)
        bc = state._bytecode
        ehhead = bc.find_exception_entry(inst.next)
        ehrelated = [ehhead]
        for eh in bc.exception_entries:
            if eh.target == ehhead.target:
                ehrelated.append(eh)
        end = max((eh.end for eh in ehrelated))
        state.append(inst, contextmanager=cm, exitfn=exitfn, end=end)
        state.push_block(state.make_block(kind='WITH', end=end))
        state.fork(pc=inst.next)

    def op_SETUP_WITH(self, state, inst):
        if False:
            i = 10
            return i + 15
        cm = state.pop()
        yielded = state.make_temp()
        exitfn = state.make_temp(prefix='setup_with_exitfn')
        state.append(inst, contextmanager=cm, exitfn=exitfn)
        if PYVERSION < (3, 9):
            state.push_block(state.make_block(kind='WITH_FINALLY', end=inst.get_jump_target()))
        state.push(exitfn)
        state.push(yielded)
        state.push_block(state.make_block(kind='WITH', end=inst.get_jump_target()))
        state.fork(pc=inst.next)

    def _setup_try(self, kind, state, next, end):
        if False:
            for i in range(10):
                print('nop')
        handler_block = state.make_block(kind=kind, end=None, reset_stack=False)
        state.fork(pc=next, extra_block=state.make_block(kind='TRY', end=end, reset_stack=False, handler=handler_block))

    def op_PUSH_EXC_INFO(self, state, inst):
        if False:
            return 10
        tos = state.pop()
        state.push(state.make_temp('exception'))
        state.push(tos)

    def op_SETUP_FINALLY(self, state, inst):
        if False:
            print('Hello World!')
        state.append(inst)
        self._setup_try('FINALLY', state, next=inst.next, end=inst.get_jump_target())
    if PYVERSION == (3, 11):

        def op_POP_EXCEPT(self, state, inst):
            if False:
                i = 10
                return i + 15
            state.pop()
    elif PYVERSION < (3, 11):

        def op_POP_EXCEPT(self, state, inst):
            if False:
                i = 10
                return i + 15
            blk = state.pop_block()
            if blk['kind'] not in {BlockKind('EXCEPT'), BlockKind('FINALLY')}:
                raise UnsupportedError(f"POP_EXCEPT got an unexpected block: {blk['kind']}", loc=self.get_debug_loc(inst.lineno))
            state.pop()
            state.pop()
            state.pop()
            state.fork(pc=inst.next)
    else:
        raise NotImplementedError(PYVERSION)

    def op_POP_BLOCK(self, state, inst):
        if False:
            while True:
                i = 10
        blk = state.pop_block()
        if blk['kind'] == BlockKind('TRY'):
            state.append(inst, kind='try')
        elif blk['kind'] == BlockKind('WITH'):
            state.append(inst, kind='with')
        state.fork(pc=inst.next)

    def op_BINARY_SUBSCR(self, state, inst):
        if False:
            return 10
        index = state.pop()
        target = state.pop()
        res = state.make_temp()
        state.append(inst, index=index, target=target, res=res)
        state.push(res)

    def op_STORE_SUBSCR(self, state, inst):
        if False:
            i = 10
            return i + 15
        index = state.pop()
        target = state.pop()
        value = state.pop()
        state.append(inst, target=target, index=index, value=value)

    def op_DELETE_SUBSCR(self, state, inst):
        if False:
            i = 10
            return i + 15
        index = state.pop()
        target = state.pop()
        state.append(inst, target=target, index=index)

    def op_CALL(self, state, inst):
        if False:
            while True:
                i = 10
        narg = inst.arg
        args = list(reversed([state.pop() for _ in range(narg)]))
        callable_or_firstarg = state.pop()
        null_or_callable = state.pop()
        if _is_null_temp_reg(null_or_callable):
            callable = callable_or_firstarg
        else:
            callable = null_or_callable
            args = [callable_or_firstarg, *args]
        res = state.make_temp()
        kw_names = state.pop_kw_names()
        state.append(inst, func=callable, args=args, kw_names=kw_names, res=res)
        state.push(res)

    def op_KW_NAMES(self, state, inst):
        if False:
            i = 10
            return i + 15
        state.set_kw_names(inst.arg)

    def op_CALL_FUNCTION(self, state, inst):
        if False:
            print('Hello World!')
        narg = inst.arg
        args = list(reversed([state.pop() for _ in range(narg)]))
        func = state.pop()
        res = state.make_temp()
        state.append(inst, func=func, args=args, res=res)
        state.push(res)

    def op_CALL_FUNCTION_KW(self, state, inst):
        if False:
            while True:
                i = 10
        narg = inst.arg
        names = state.pop()
        args = list(reversed([state.pop() for _ in range(narg)]))
        func = state.pop()
        res = state.make_temp()
        state.append(inst, func=func, args=args, names=names, res=res)
        state.push(res)

    def op_CALL_FUNCTION_EX(self, state, inst):
        if False:
            for i in range(10):
                print('nop')
        if inst.arg & 1 and PYVERSION < (3, 10):
            errmsg = 'CALL_FUNCTION_EX with **kwargs not supported'
            raise UnsupportedError(errmsg)
        if inst.arg & 1:
            varkwarg = state.pop()
        else:
            varkwarg = None
        vararg = state.pop()
        func = state.pop()
        if PYVERSION == (3, 11):
            if _is_null_temp_reg(state.peek(1)):
                state.pop()
        res = state.make_temp()
        state.append(inst, func=func, vararg=vararg, varkwarg=varkwarg, res=res)
        state.push(res)

    def _dup_topx(self, state, inst, count):
        if False:
            print('Hello World!')
        orig = [state.pop() for _ in range(count)]
        orig.reverse()
        duped = [state.make_temp() for _ in range(count)]
        state.append(inst, orig=orig, duped=duped)
        for val in orig:
            state.push(val)
        for val in duped:
            state.push(val)

    def op_DUP_TOPX(self, state, inst):
        if False:
            for i in range(10):
                print('nop')
        count = inst.arg
        assert 1 <= count <= 5, 'Invalid DUP_TOPX count'
        self._dup_topx(state, inst, count)

    def op_DUP_TOP(self, state, inst):
        if False:
            while True:
                i = 10
        self._dup_topx(state, inst, count=1)

    def op_DUP_TOP_TWO(self, state, inst):
        if False:
            return 10
        self._dup_topx(state, inst, count=2)

    def op_COPY(self, state, inst):
        if False:
            return 10
        state.push(state.peek(inst.arg))

    def op_SWAP(self, state, inst):
        if False:
            print('Hello World!')
        state.swap(inst.arg)

    def op_ROT_TWO(self, state, inst):
        if False:
            while True:
                i = 10
        first = state.pop()
        second = state.pop()
        state.push(first)
        state.push(second)

    def op_ROT_THREE(self, state, inst):
        if False:
            i = 10
            return i + 15
        first = state.pop()
        second = state.pop()
        third = state.pop()
        state.push(first)
        state.push(third)
        state.push(second)

    def op_ROT_FOUR(self, state, inst):
        if False:
            while True:
                i = 10
        first = state.pop()
        second = state.pop()
        third = state.pop()
        forth = state.pop()
        state.push(first)
        state.push(forth)
        state.push(third)
        state.push(second)

    def op_UNPACK_SEQUENCE(self, state, inst):
        if False:
            for i in range(10):
                print('nop')
        count = inst.arg
        iterable = state.pop()
        stores = [state.make_temp() for _ in range(count)]
        tupleobj = state.make_temp()
        state.append(inst, iterable=iterable, stores=stores, tupleobj=tupleobj)
        for st in reversed(stores):
            state.push(st)

    def op_BUILD_TUPLE(self, state, inst):
        if False:
            print('Hello World!')
        count = inst.arg
        items = list(reversed([state.pop() for _ in range(count)]))
        tup = state.make_temp()
        state.append(inst, items=items, res=tup)
        state.push(tup)

    def _build_tuple_unpack(self, state, inst):
        if False:
            return 10
        tuples = list(reversed([state.pop() for _ in range(inst.arg)]))
        temps = [state.make_temp() for _ in range(len(tuples) - 1)]
        is_assign = len(tuples) == 1
        if is_assign:
            temps = [state.make_temp()]
        state.append(inst, tuples=tuples, temps=temps, is_assign=is_assign)
        state.push(temps[-1])

    def op_BUILD_TUPLE_UNPACK_WITH_CALL(self, state, inst):
        if False:
            for i in range(10):
                print('nop')
        self._build_tuple_unpack(state, inst)

    def op_BUILD_TUPLE_UNPACK(self, state, inst):
        if False:
            return 10
        self._build_tuple_unpack(state, inst)

    def op_LIST_TO_TUPLE(self, state, inst):
        if False:
            while True:
                i = 10
        tos = state.pop()
        res = state.make_temp()
        state.append(inst, const_list=tos, res=res)
        state.push(res)

    def op_BUILD_CONST_KEY_MAP(self, state, inst):
        if False:
            print('Hello World!')
        keys = state.pop()
        vals = list(reversed([state.pop() for _ in range(inst.arg)]))
        keytmps = [state.make_temp() for _ in range(inst.arg)]
        res = state.make_temp()
        state.append(inst, keys=keys, keytmps=keytmps, values=vals, res=res)
        state.push(res)

    def op_BUILD_LIST(self, state, inst):
        if False:
            return 10
        count = inst.arg
        items = list(reversed([state.pop() for _ in range(count)]))
        lst = state.make_temp()
        state.append(inst, items=items, res=lst)
        state.push(lst)

    def op_LIST_APPEND(self, state, inst):
        if False:
            while True:
                i = 10
        value = state.pop()
        index = inst.arg
        target = state.peek(index)
        appendvar = state.make_temp()
        res = state.make_temp()
        state.append(inst, target=target, value=value, appendvar=appendvar, res=res)

    def op_LIST_EXTEND(self, state, inst):
        if False:
            print('Hello World!')
        value = state.pop()
        index = inst.arg
        target = state.peek(index)
        extendvar = state.make_temp()
        res = state.make_temp()
        state.append(inst, target=target, value=value, extendvar=extendvar, res=res)

    def op_BUILD_MAP(self, state, inst):
        if False:
            print('Hello World!')
        dct = state.make_temp()
        count = inst.arg
        items = []
        for i in range(count):
            (v, k) = (state.pop(), state.pop())
            items.append((k, v))
        state.append(inst, items=items[::-1], size=count, res=dct)
        state.push(dct)

    def op_MAP_ADD(self, state, inst):
        if False:
            return 10
        TOS = state.pop()
        TOS1 = state.pop()
        (key, value) = (TOS1, TOS)
        index = inst.arg
        target = state.peek(index)
        setitemvar = state.make_temp()
        res = state.make_temp()
        state.append(inst, target=target, key=key, value=value, setitemvar=setitemvar, res=res)

    def op_BUILD_SET(self, state, inst):
        if False:
            return 10
        count = inst.arg
        items = list(reversed([state.pop() for _ in range(count)]))
        res = state.make_temp()
        state.append(inst, items=items, res=res)
        state.push(res)

    def op_SET_UPDATE(self, state, inst):
        if False:
            while True:
                i = 10
        value = state.pop()
        index = inst.arg
        target = state.peek(index)
        updatevar = state.make_temp()
        res = state.make_temp()
        state.append(inst, target=target, value=value, updatevar=updatevar, res=res)

    def op_DICT_UPDATE(self, state, inst):
        if False:
            i = 10
            return i + 15
        value = state.pop()
        index = inst.arg
        target = state.peek(index)
        updatevar = state.make_temp()
        res = state.make_temp()
        state.append(inst, target=target, value=value, updatevar=updatevar, res=res)

    def op_GET_ITER(self, state, inst):
        if False:
            while True:
                i = 10
        value = state.pop()
        res = state.make_temp()
        state.append(inst, value=value, res=res)
        state.push(res)

    def op_FOR_ITER(self, state, inst):
        if False:
            return 10
        iterator = state.get_tos()
        pair = state.make_temp()
        indval = state.make_temp()
        pred = state.make_temp()
        state.append(inst, iterator=iterator, pair=pair, indval=indval, pred=pred)
        state.push(indval)
        end = inst.get_jump_target()
        state.fork(pc=end, npop=2)
        state.fork(pc=inst.next)

    def op_GEN_START(self, state, inst):
        if False:
            return 10
        'Pops TOS. If TOS was not None, raises an exception. The kind\n        operand corresponds to the type of generator or coroutine and\n        determines the error message. The legal kinds are 0 for generator,\n        1 for coroutine, and 2 for async generator.\n\n        New in version 3.10.\n        '
        pass

    def op_BINARY_OP(self, state, inst):
        if False:
            while True:
                i = 10
        op = dis._nb_ops[inst.arg][1]
        rhs = state.pop()
        lhs = state.pop()
        op_name = ALL_BINOPS_TO_OPERATORS[op].__name__
        res = state.make_temp(prefix=f'binop_{op_name}')
        state.append(inst, op=op, lhs=lhs, rhs=rhs, res=res)
        state.push(res)

    def _unaryop(self, state, inst):
        if False:
            return 10
        val = state.pop()
        res = state.make_temp()
        state.append(inst, value=val, res=res)
        state.push(res)
    op_UNARY_NEGATIVE = _unaryop
    op_UNARY_POSITIVE = _unaryop
    op_UNARY_NOT = _unaryop
    op_UNARY_INVERT = _unaryop

    def _binaryop(self, state, inst):
        if False:
            for i in range(10):
                print('nop')
        rhs = state.pop()
        lhs = state.pop()
        res = state.make_temp()
        state.append(inst, lhs=lhs, rhs=rhs, res=res)
        state.push(res)
    op_COMPARE_OP = _binaryop
    op_IS_OP = _binaryop
    op_CONTAINS_OP = _binaryop
    op_INPLACE_ADD = _binaryop
    op_INPLACE_SUBTRACT = _binaryop
    op_INPLACE_MULTIPLY = _binaryop
    op_INPLACE_DIVIDE = _binaryop
    op_INPLACE_TRUE_DIVIDE = _binaryop
    op_INPLACE_FLOOR_DIVIDE = _binaryop
    op_INPLACE_MODULO = _binaryop
    op_INPLACE_POWER = _binaryop
    op_INPLACE_MATRIX_MULTIPLY = _binaryop
    op_INPLACE_LSHIFT = _binaryop
    op_INPLACE_RSHIFT = _binaryop
    op_INPLACE_AND = _binaryop
    op_INPLACE_OR = _binaryop
    op_INPLACE_XOR = _binaryop
    op_BINARY_ADD = _binaryop
    op_BINARY_SUBTRACT = _binaryop
    op_BINARY_MULTIPLY = _binaryop
    op_BINARY_DIVIDE = _binaryop
    op_BINARY_TRUE_DIVIDE = _binaryop
    op_BINARY_FLOOR_DIVIDE = _binaryop
    op_BINARY_MODULO = _binaryop
    op_BINARY_POWER = _binaryop
    op_BINARY_MATRIX_MULTIPLY = _binaryop
    op_BINARY_LSHIFT = _binaryop
    op_BINARY_RSHIFT = _binaryop
    op_BINARY_AND = _binaryop
    op_BINARY_OR = _binaryop
    op_BINARY_XOR = _binaryop

    def op_MAKE_FUNCTION(self, state, inst, MAKE_CLOSURE=False):
        if False:
            return 10
        if PYVERSION == (3, 11):
            name = None
        elif PYVERSION < (3, 11):
            name = state.pop()
        else:
            raise NotImplementedError(PYVERSION)
        code = state.pop()
        closure = annotations = kwdefaults = defaults = None
        if inst.arg & 8:
            closure = state.pop()
        if inst.arg & 4:
            annotations = state.pop()
        if inst.arg & 2:
            kwdefaults = state.pop()
        if inst.arg & 1:
            defaults = state.pop()
        res = state.make_temp()
        state.append(inst, name=name, code=code, closure=closure, annotations=annotations, kwdefaults=kwdefaults, defaults=defaults, res=res)
        state.push(res)

    def op_MAKE_CLOSURE(self, state, inst):
        if False:
            for i in range(10):
                print('nop')
        self.op_MAKE_FUNCTION(state, inst, MAKE_CLOSURE=True)

    def op_LOAD_CLOSURE(self, state, inst):
        if False:
            while True:
                i = 10
        res = state.make_temp()
        state.append(inst, res=res)
        state.push(res)

    def op_LOAD_ASSERTION_ERROR(self, state, inst):
        if False:
            for i in range(10):
                print('nop')
        res = state.make_temp('assertion_error')
        state.append(inst, res=res)
        state.push(res)

    def op_CHECK_EXC_MATCH(self, state, inst):
        if False:
            return 10
        pred = state.make_temp('predicate')
        tos = state.pop()
        tos1 = state.get_tos()
        state.append(inst, pred=pred, tos=tos, tos1=tos1)
        state.push(pred)

    def op_JUMP_IF_NOT_EXC_MATCH(self, state, inst):
        if False:
            print('Hello World!')
        pred = state.make_temp('predicate')
        tos = state.pop()
        tos1 = state.pop()
        state.append(inst, pred=pred, tos=tos, tos1=tos1)
        state.fork(pc=inst.next)
        state.fork(pc=inst.get_jump_target())
    if PYVERSION == (3, 11):

        def op_RERAISE(self, state, inst):
            if False:
                print('Hello World!')
            exc = state.pop()
            if inst.arg != 0:
                state.pop()
            state.append(inst, exc=exc)
            if state.has_active_try():
                self._adjust_except_stack(state)
            else:
                state.terminate()
    elif PYVERSION < (3, 11):

        def op_RERAISE(self, state, inst):
            if False:
                i = 10
                return i + 15
            exc = state.pop()
            state.append(inst, exc=exc)
            state.terminate()
    else:
        raise NotImplementedError(PYVERSION)
    if PYVERSION == (3, 11):

        def op_LOAD_METHOD(self, state, inst):
            if False:
                for i in range(10):
                    print('nop')
            item = state.pop()
            extra = state.make_null()
            state.push(extra)
            res = state.make_temp()
            state.append(inst, item=item, res=res)
            state.push(res)
    elif PYVERSION < (3, 11):

        def op_LOAD_METHOD(self, state, inst):
            if False:
                i = 10
                return i + 15
            self.op_LOAD_ATTR(state, inst)
    else:
        raise NotImplementedError(PYVERSION)

    def op_CALL_METHOD(self, state, inst):
        if False:
            i = 10
            return i + 15
        self.op_CALL_FUNCTION(state, inst)

@total_ordering
class _State(object):
    """State of the trace
    """

    def __init__(self, bytecode, pc, nstack, blockstack, nullvals=()):
        if False:
            print('Hello World!')
        '\n        Parameters\n        ----------\n        bytecode : numba.bytecode.ByteCode\n            function bytecode\n        pc : int\n            program counter\n        nstack : int\n            stackdepth at entry\n        blockstack : Sequence[Dict]\n            A sequence of dictionary denoting entries on the blockstack.\n        '
        self._bytecode = bytecode
        self._pc_initial = pc
        self._pc = pc
        self._nstack_initial = nstack
        self._stack = []
        self._blockstack_initial = tuple(blockstack)
        self._blockstack = list(blockstack)
        self._temp_registers = []
        self._insts = []
        self._outedges = []
        self._terminated = False
        self._phis = {}
        self._outgoing_phis = UniqueDict()
        self._used_regs = set()
        for i in range(nstack):
            if i in nullvals:
                phi = self.make_temp('null$')
            else:
                phi = self.make_temp('phi')
            self._phis[phi] = i
            self.push(phi)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'State(pc_initial={} nstack_initial={})'.format(self._pc_initial, self._nstack_initial)

    def get_identity(self):
        if False:
            i = 10
            return i + 15
        return (self._pc_initial, self._nstack_initial)

    def __hash__(self):
        if False:
            while True:
                i = 10
        return hash(self.get_identity())

    def __lt__(self, other):
        if False:
            while True:
                i = 10
        return self.get_identity() < other.get_identity()

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        return self.get_identity() == other.get_identity()

    @property
    def pc_initial(self):
        if False:
            print('Hello World!')
        'The starting bytecode offset of this State.\n        The PC given to the constructor.\n        '
        return self._pc_initial

    @property
    def instructions(self):
        if False:
            print('Hello World!')
        'The list of instructions information as a 2-tuple of\n        ``(pc : int, register_map : Dict)``\n        '
        return self._insts

    @property
    def outgoing_edges(self):
        if False:
            i = 10
            return i + 15
        'The list of outgoing edges.\n\n        Returns\n        -------\n        edges : List[State]\n        '
        return self._outedges

    @property
    def outgoing_phis(self):
        if False:
            i = 10
            return i + 15
        'The dictionary of outgoing phi nodes.\n\n        The keys are the name of the PHI nodes.\n        The values are the outgoing states.\n        '
        return self._outgoing_phis

    @property
    def blockstack_initial(self):
        if False:
            return 10
        'A copy of the initial state of the blockstack\n        '
        return self._blockstack_initial

    @property
    def stack_depth(self):
        if False:
            i = 10
            return i + 15
        'The current size of the stack\n\n        Returns\n        -------\n        res : int\n        '
        return len(self._stack)

    def find_initial_try_block(self):
        if False:
            while True:
                i = 10
        'Find the initial *try* block.\n        '
        for blk in reversed(self._blockstack_initial):
            if blk['kind'] == BlockKind('TRY'):
                return blk

    def has_terminated(self):
        if False:
            print('Hello World!')
        return self._terminated

    def get_inst(self):
        if False:
            while True:
                i = 10
        return self._bytecode[self._pc]

    def advance_pc(self):
        if False:
            while True:
                i = 10
        inst = self.get_inst()
        self._pc = inst.next

    def make_temp(self, prefix=''):
        if False:
            while True:
                i = 10
        if not prefix:
            name = '${prefix}{offset}{opname}.{tempct}'.format(prefix=prefix, offset=self._pc, opname=self.get_inst().opname.lower(), tempct=len(self._temp_registers))
        else:
            name = '${prefix}{offset}.{tempct}'.format(prefix=prefix, offset=self._pc, tempct=len(self._temp_registers))
        self._temp_registers.append(name)
        return name

    def append(self, inst, **kwargs):
        if False:
            print('Hello World!')
        'Append new inst'
        self._insts.append((inst.offset, kwargs))
        self._used_regs |= set(_flatten_inst_regs(kwargs.values()))

    def get_tos(self):
        if False:
            i = 10
            return i + 15
        return self.peek(1)

    def peek(self, k):
        if False:
            print('Hello World!')
        "Return the k'th element on the stack\n        "
        return self._stack[-k]

    def push(self, item):
        if False:
            return 10
        'Push to stack'
        self._stack.append(item)

    def pop(self):
        if False:
            for i in range(10):
                print('nop')
        'Pop the stack'
        return self._stack.pop()

    def swap(self, idx):
        if False:
            i = 10
            return i + 15
        'Swap stack[idx] with the tos'
        s = self._stack
        (s[-1], s[-idx]) = (s[-idx], s[-1])

    def push_block(self, synblk):
        if False:
            print('Hello World!')
        'Push a block to blockstack\n        '
        assert 'stack_depth' in synblk
        self._blockstack.append(synblk)

    def reset_stack(self, depth):
        if False:
            while True:
                i = 10
        'Reset the stack to the given stack depth.\n        Returning the popped items.\n        '
        (self._stack, popped) = (self._stack[:depth], self._stack[depth:])
        return popped

    def make_block(self, kind, end, reset_stack=True, handler=None):
        if False:
            return 10
        'Make a new block\n        '
        d = {'kind': BlockKind(kind), 'end': end, 'entry_stack': len(self._stack)}
        if reset_stack:
            d['stack_depth'] = len(self._stack)
        else:
            d['stack_depth'] = None
        d['handler'] = handler
        return d

    def pop_block(self):
        if False:
            while True:
                i = 10
        'Pop a block and unwind the stack\n        '
        b = self._blockstack.pop()
        self.reset_stack(b['stack_depth'])
        return b

    def pop_block_and_above(self, blk):
        if False:
            for i in range(10):
                print('nop')
        'Find *blk* in the blockstack and remove it and all blocks above it\n        from the stack.\n        '
        idx = self._blockstack.index(blk)
        assert 0 <= idx < len(self._blockstack)
        self._blockstack = self._blockstack[:idx]

    def get_top_block(self, kind):
        if False:
            for i in range(10):
                print('nop')
        'Find the first block that matches *kind*\n        '
        kind = BlockKind(kind)
        for bs in reversed(self._blockstack):
            if bs['kind'] == kind:
                return bs

    def get_top_block_either(self, *kinds):
        if False:
            i = 10
            return i + 15
        'Find the first block that matches *kind*\n        '
        kinds = {BlockKind(kind) for kind in kinds}
        for bs in reversed(self._blockstack):
            if bs['kind'] in kinds:
                return bs

    def has_active_try(self):
        if False:
            i = 10
            return i + 15
        'Returns a boolean indicating if the top-block is a *try* block\n        '
        return self.get_top_block('TRY') is not None

    def get_varname(self, inst):
        if False:
            return 10
        'Get referenced variable name from the oparg\n        '
        return self._bytecode.co_varnames[inst.arg]

    def terminate(self):
        if False:
            return 10
        'Mark block as terminated\n        '
        self._terminated = True

    def fork(self, pc, npop=0, npush=0, extra_block=None):
        if False:
            return 10
        'Fork the state\n        '
        stack = list(self._stack)
        if npop:
            assert 0 <= npop <= len(self._stack)
            nstack = len(self._stack) - npop
            stack = stack[:nstack]
        if npush:
            assert 0 <= npush
            for i in range(npush):
                stack.append(self.make_temp())
        blockstack = list(self._blockstack)
        if PYVERSION == (3, 11):
            while blockstack:
                top = blockstack[-1]
                end = top.get('end_offset') or top['end']
                if pc >= end:
                    blockstack.pop()
                else:
                    break
        elif PYVERSION < (3, 11):
            pass
        else:
            raise NotImplementedError(PYVERSION)
        if extra_block:
            blockstack.append(extra_block)
        self._outedges.append(Edge(pc=pc, stack=tuple(stack), npush=npush, blockstack=tuple(blockstack)))
        self.terminate()

    def split_new_block(self):
        if False:
            return 10
        'Split the state\n        '
        self.fork(pc=self._pc)

    def get_outgoing_states(self):
        if False:
            while True:
                i = 10
        'Get states for each outgoing edges\n        '
        assert not self._outgoing_phis
        ret = []
        for edge in self._outedges:
            state = State(bytecode=self._bytecode, pc=edge.pc, nstack=len(edge.stack), blockstack=edge.blockstack, nullvals=[i for (i, v) in enumerate(edge.stack) if _is_null_temp_reg(v)])
            ret.append(state)
            for (phi, i) in state._phis.items():
                self._outgoing_phis[phi] = edge.stack[i]
        return ret

    def get_outgoing_edgepushed(self):
        if False:
            while True:
                i = 10
        '\n        Returns\n        -------\n        Dict[int, int]\n            where keys are the PC\n            values are the edge-pushed stack values\n        '
        return {edge.pc: tuple(edge.stack[-edge.npush:]) for edge in self._outedges}

class StatePy311(_State):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super().__init__(*args, **kwargs)
        self._kw_names = None

    def pop_kw_names(self):
        if False:
            for i in range(10):
                print('nop')
        out = self._kw_names
        self._kw_names = None
        return out

    def set_kw_names(self, val):
        if False:
            for i in range(10):
                print('nop')
        assert self._kw_names is None
        self._kw_names = val

    def is_in_exception(self):
        if False:
            while True:
                i = 10
        bc = self._bytecode
        return bc.find_exception_entry(self._pc) is not None

    def get_exception(self):
        if False:
            print('Hello World!')
        bc = self._bytecode
        return bc.find_exception_entry(self._pc)

    def in_with(self):
        if False:
            i = 10
            return i + 15
        for ent in self._blockstack_initial:
            if ent['kind'] == BlockKind('WITH'):
                return True

    def make_null(self):
        if False:
            while True:
                i = 10
        return self.make_temp(prefix='null$')
if PYVERSION == (3, 11):
    State = StatePy311
elif PYVERSION < (3, 11):
    State = _State
else:
    raise NotImplementedError(PYVERSION)
Edge = namedtuple('Edge', ['pc', 'stack', 'blockstack', 'npush'])

class AdaptDFA(object):
    """Adapt Flow to the old DFA class expected by Interpreter
    """

    def __init__(self, flow):
        if False:
            print('Hello World!')
        self._flow = flow

    @property
    def infos(self):
        if False:
            while True:
                i = 10
        return self._flow.block_infos
AdaptBlockInfo = namedtuple('AdaptBlockInfo', ['insts', 'outgoing_phis', 'blockstack', 'active_try_block', 'outgoing_edgepushed'])

def adapt_state_infos(state):
    if False:
        print('Hello World!')
    return AdaptBlockInfo(insts=tuple(state.instructions), outgoing_phis=state.outgoing_phis, blockstack=state.blockstack_initial, active_try_block=state.find_initial_try_block(), outgoing_edgepushed=state.get_outgoing_edgepushed())

def _flatten_inst_regs(iterable):
    if False:
        for i in range(10):
            print('nop')
    'Flatten an iterable of registers used in an instruction\n    '
    for item in iterable:
        if isinstance(item, str):
            yield item
        elif isinstance(item, (tuple, list)):
            for x in _flatten_inst_regs(item):
                yield x

class AdaptCFA(object):
    """Adapt Flow to the old CFA class expected by Interpreter
    """

    def __init__(self, flow):
        if False:
            i = 10
            return i + 15
        self._flow = flow
        self._blocks = {}
        for (offset, blockinfo) in flow.block_infos.items():
            self._blocks[offset] = AdaptCFBlock(blockinfo, offset)
        backbone = self._flow.cfgraph.backbone()
        graph = flow.cfgraph
        backbone = graph.backbone()
        inloopblocks = set()
        for b in self.blocks.keys():
            if graph.in_loops(b):
                inloopblocks.add(b)
        self._backbone = backbone - inloopblocks

    @property
    def graph(self):
        if False:
            return 10
        return self._flow.cfgraph

    @property
    def backbone(self):
        if False:
            for i in range(10):
                print('nop')
        return self._backbone

    @property
    def blocks(self):
        if False:
            i = 10
            return i + 15
        return self._blocks

    def iterliveblocks(self):
        if False:
            return 10
        for b in sorted(self.blocks):
            yield self.blocks[b]

    def dump(self):
        if False:
            while True:
                i = 10
        self._flow.cfgraph.dump()

class AdaptCFBlock(object):

    def __init__(self, blockinfo, offset):
        if False:
            return 10
        self.offset = offset
        self.body = tuple((i for (i, _) in blockinfo.insts))