"""
Implement Dominance-Fronter-based SSA by Choi et al described in Inria SSA book

References:

- Static Single Assignment Book by Inria
  http://ssabook.gforge.inria.fr/latest/book.pdf
- Choi et al. Incremental computation of static single assignment form.
"""
import logging
import operator
import warnings
from functools import reduce
from copy import copy
from pprint import pformat
from collections import defaultdict
from numba import config
from numba.core import ir, ir_utils, errors
from numba.core.analysis import compute_cfg_from_blocks
_logger = logging.getLogger(__name__)

def reconstruct_ssa(func_ir):
    if False:
        print('Hello World!')
    'Apply SSA reconstruction algorithm on the given IR.\n\n    Produces minimal SSA using Choi et al algorithm.\n    '
    func_ir.blocks = _run_ssa(func_ir.blocks)
    return func_ir

class _CacheListVars:

    def __init__(self):
        if False:
            return 10
        self._saved = {}

    def get(self, inst):
        if False:
            for i in range(10):
                print('nop')
        got = self._saved.get(inst)
        if got is None:
            self._saved[inst] = got = inst.list_vars()
        return got

def _run_ssa(blocks):
    if False:
        for i in range(10):
            print('nop')
    'Run SSA reconstruction on IR blocks of a function.\n    '
    if not blocks:
        return {}
    cfg = compute_cfg_from_blocks(blocks)
    df_plus = _iterated_domfronts(cfg)
    violators = _find_defs_violators(blocks)
    cache_list_vars = _CacheListVars()
    for varname in violators:
        _logger.debug('Fix SSA violator on var %s', varname)
        (blocks, defmap) = _fresh_vars(blocks, varname)
        _logger.debug('Replaced assignments: %s', pformat(defmap))
        blocks = _fix_ssa_vars(blocks, varname, defmap, cfg, df_plus, cache_list_vars)
    cfg_post = compute_cfg_from_blocks(blocks)
    if cfg_post != cfg:
        raise errors.CompilerError('CFG mutated in SSA pass')
    return blocks

def _fix_ssa_vars(blocks, varname, defmap, cfg, df_plus, cache_list_vars):
    if False:
        while True:
            i = 10
    'Rewrite all uses to ``varname`` given the definition map\n    '
    states = _make_states(blocks)
    states['varname'] = varname
    states['defmap'] = defmap
    states['phimap'] = phimap = defaultdict(list)
    states['cfg'] = cfg
    states['phi_locations'] = _compute_phi_locations(df_plus, defmap)
    newblocks = _run_block_rewrite(blocks, states, _FixSSAVars(cache_list_vars))
    for (label, philist) in phimap.items():
        curblk = newblocks[label]
        curblk.body = philist + curblk.body
    return newblocks

def _iterated_domfronts(cfg):
    if False:
        while True:
            i = 10
    'Compute the iterated dominance frontiers (DF+ in literatures).\n\n    Returns a dictionary which maps block label to the set of labels of its\n    iterated dominance frontiers.\n    '
    domfronts = {k: set(vs) for (k, vs) in cfg.dominance_frontier().items()}
    keep_going = True
    while keep_going:
        keep_going = False
        for (k, vs) in domfronts.items():
            inner = reduce(operator.or_, [domfronts[v] for v in vs], set())
            if inner.difference(vs):
                vs |= inner
                keep_going = True
    return domfronts

def _compute_phi_locations(iterated_df, defmap):
    if False:
        return 10
    phi_locations = set()
    for (deflabel, defstmts) in defmap.items():
        if defstmts:
            phi_locations |= iterated_df[deflabel]
    return phi_locations

def _fresh_vars(blocks, varname):
    if False:
        i = 10
        return i + 15
    'Rewrite to put fresh variable names\n    '
    states = _make_states(blocks)
    states['varname'] = varname
    states['defmap'] = defmap = defaultdict(list)
    newblocks = _run_block_rewrite(blocks, states, _FreshVarHandler())
    return (newblocks, defmap)

def _get_scope(blocks):
    if False:
        i = 10
        return i + 15
    (first, *_) = blocks.values()
    return first.scope

def _find_defs_violators(blocks):
    if False:
        return 10
    '\n    Returns\n    -------\n    res : Set[str]\n        The SSA violators in a dictionary of variable names.\n    '
    defs = defaultdict(list)
    _run_block_analysis(blocks, defs, _GatherDefsHandler())
    _logger.debug('defs %s', pformat(defs))
    violators = {k for (k, vs) in defs.items() if len(vs) > 1}
    _logger.debug('SSA violators %s', pformat(violators))
    return violators

def _run_block_analysis(blocks, states, handler):
    if False:
        for i in range(10):
            print('nop')
    for (label, blk) in blocks.items():
        _logger.debug('==== SSA block analysis pass on %s', label)
        for _ in _run_ssa_block_pass(states, blk, handler):
            pass

def _run_block_rewrite(blocks, states, handler):
    if False:
        for i in range(10):
            print('nop')
    newblocks = {}
    for (label, blk) in blocks.items():
        _logger.debug('==== SSA block rewrite pass on %s', label)
        newblk = ir.Block(scope=blk.scope, loc=blk.loc)
        newbody = []
        states['label'] = label
        states['block'] = blk
        for stmt in _run_ssa_block_pass(states, blk, handler):
            assert stmt is not None
            newbody.append(stmt)
        newblk.body = newbody
        newblocks[label] = newblk
    return newblocks

def _make_states(blocks):
    if False:
        for i in range(10):
            print('nop')
    return dict(scope=_get_scope(blocks))

def _run_ssa_block_pass(states, blk, handler):
    if False:
        print('Hello World!')
    _logger.debug('Running %s', handler)
    for stmt in blk.body:
        _logger.debug('on stmt: %s', stmt)
        if isinstance(stmt, ir.Assign):
            ret = handler.on_assign(states, stmt)
        else:
            ret = handler.on_other(states, stmt)
        if ret is not stmt and ret is not None:
            _logger.debug('replaced with: %s', ret)
        yield ret

class _BaseHandler:
    """A base handler for all the passes used here for the SSA algorithm.
    """

    def on_assign(self, states, assign):
        if False:
            return 10
        '\n        Called when the pass sees an ``ir.Assign``.\n\n        Subclasses should override this for custom behavior\n\n        Parameters\n        -----------\n        states : dict\n        assign : numba.ir.Assign\n\n        Returns\n        -------\n        stmt : numba.ir.Assign or None\n            For rewrite passes, the return value is used as the replacement\n            for the given statement.\n        '

    def on_other(self, states, stmt):
        if False:
            for i in range(10):
                print('nop')
        "\n        Called when the pass sees an ``ir.Stmt`` that's not an assignment.\n\n        Subclasses should override this for custom behavior\n\n        Parameters\n        -----------\n        states : dict\n        assign : numba.ir.Stmt\n\n        Returns\n        -------\n        stmt : numba.ir.Stmt or None\n            For rewrite passes, the return value is used as the replacement\n            for the given statement.\n        "

class _GatherDefsHandler(_BaseHandler):
    """Find all defs

    ``states`` is a Mapping[str, List[ir.Assign]]
    """

    def on_assign(self, states, assign):
        if False:
            return 10
        states[assign.target.name].append(assign)

class UndefinedVariable:

    def __init__(self):
        if False:
            return 10
        raise NotImplementedError('Not intended for instantiation')
    target = ir.UNDEFINED

class _FreshVarHandler(_BaseHandler):
    """Replaces assignment target with new fresh variables.
    """

    def on_assign(self, states, assign):
        if False:
            i = 10
            return i + 15
        if assign.target.name == states['varname']:
            scope = states['scope']
            defmap = states['defmap']
            if len(defmap) == 0:
                newtarget = assign.target
                _logger.debug('first assign: %s', newtarget)
                if newtarget.name not in scope.localvars:
                    wmsg = f'variable {newtarget.name!r} is not in scope.'
                    warnings.warn(errors.NumbaIRAssumptionWarning(wmsg, loc=assign.loc))
            else:
                newtarget = scope.redefine(assign.target.name, loc=assign.loc)
            assign = ir.Assign(target=newtarget, value=assign.value, loc=assign.loc)
            defmap[states['label']].append(assign)
        return assign

    def on_other(self, states, stmt):
        if False:
            print('Hello World!')
        return stmt

class _FixSSAVars(_BaseHandler):
    """Replace variable uses in IR nodes to the correct reaching variable
    and introduce Phi nodes if necessary. This class contains the core of
    the SSA reconstruction algorithm.

    See Ch 5 of the Inria SSA book for reference. The method names used here
    are similar to the names used in the pseudocode in the book.
    """

    def __init__(self, cache_list_vars):
        if False:
            while True:
                i = 10
        self._cache_list_vars = cache_list_vars

    def on_assign(self, states, assign):
        if False:
            for i in range(10):
                print('nop')
        rhs = assign.value
        if isinstance(rhs, ir.Inst):
            newdef = self._fix_var(states, assign, self._cache_list_vars.get(assign.value))
            if newdef is not None and newdef.target is not ir.UNDEFINED:
                if states['varname'] != newdef.target.name:
                    replmap = {states['varname']: newdef.target}
                    rhs = copy(rhs)
                    ir_utils.replace_vars_inner(rhs, replmap)
                    return ir.Assign(target=assign.target, value=rhs, loc=assign.loc)
        elif isinstance(rhs, ir.Var):
            newdef = self._fix_var(states, assign, [rhs])
            if newdef is not None and newdef.target is not ir.UNDEFINED:
                if states['varname'] != newdef.target.name:
                    return ir.Assign(target=assign.target, value=newdef.target, loc=assign.loc)
        return assign

    def on_other(self, states, stmt):
        if False:
            return 10
        newdef = self._fix_var(states, stmt, self._cache_list_vars.get(stmt))
        if newdef is not None and newdef.target is not ir.UNDEFINED:
            if states['varname'] != newdef.target.name:
                replmap = {states['varname']: newdef.target}
                stmt = copy(stmt)
                ir_utils.replace_vars_stmt(stmt, replmap)
        return stmt

    def _fix_var(self, states, stmt, used_vars):
        if False:
            return 10
        'Fix all variable uses in ``used_vars``.\n        '
        varnames = [k.name for k in used_vars]
        phivar = states['varname']
        if phivar in varnames:
            return self._find_def(states, stmt)

    def _find_def(self, states, stmt):
        if False:
            print('Hello World!')
        'Find definition of ``stmt`` for the statement ``stmt``\n        '
        _logger.debug('find_def var=%r stmt=%s', states['varname'], stmt)
        selected_def = None
        label = states['label']
        local_defs = states['defmap'][label]
        local_phis = states['phimap'][label]
        block = states['block']
        cur_pos = self._stmt_index(stmt, block)
        for defstmt in reversed(local_defs):
            def_pos = self._stmt_index(defstmt, block, stop=cur_pos)
            if def_pos < cur_pos:
                selected_def = defstmt
                break
            elif defstmt in local_phis:
                selected_def = local_phis[-1]
                break
        if selected_def is None:
            selected_def = self._find_def_from_top(states, label, loc=stmt.loc)
        return selected_def

    def _find_def_from_top(self, states, label, loc):
        if False:
            return 10
        'Find definition reaching block of ``label``.\n\n        This method would look at all dominance frontiers.\n        Insert phi node if necessary.\n        '
        _logger.debug('find_def_from_top label %r', label)
        cfg = states['cfg']
        defmap = states['defmap']
        phimap = states['phimap']
        phi_locations = states['phi_locations']
        if label in phi_locations:
            scope = states['scope']
            loc = states['block'].loc
            freshvar = scope.redefine(states['varname'], loc=loc)
            phinode = ir.Assign(target=freshvar, value=ir.Expr.phi(loc=loc), loc=loc)
            _logger.debug('insert phi node %s at %s', phinode, label)
            defmap[label].insert(0, phinode)
            phimap[label].append(phinode)
            for (pred, _) in cfg.predecessors(label):
                incoming_def = self._find_def_from_bottom(states, pred, loc=loc)
                _logger.debug('incoming_def %s', incoming_def)
                phinode.value.incoming_values.append(incoming_def.target)
                phinode.value.incoming_blocks.append(pred)
            return phinode
        else:
            idom = cfg.immediate_dominators()[label]
            if idom == label:
                _warn_about_uninitialized_variable(states['varname'], loc)
                return UndefinedVariable
            _logger.debug('idom %s from label %s', idom, label)
            return self._find_def_from_bottom(states, idom, loc=loc)

    def _find_def_from_bottom(self, states, label, loc):
        if False:
            i = 10
            return i + 15
        'Find definition from within the block at ``label``.\n        '
        _logger.debug('find_def_from_bottom label %r', label)
        defmap = states['defmap']
        defs = defmap[label]
        if defs:
            lastdef = defs[-1]
            return lastdef
        else:
            return self._find_def_from_top(states, label, loc=loc)

    def _stmt_index(self, defstmt, block, stop=-1):
        if False:
            print('Hello World!')
        'Find the positional index of the statement at ``block``.\n\n        Assumptions:\n        - no two statements can point to the same object.\n        '
        for i in range(len(block.body))[:stop]:
            if block.body[i] is defstmt:
                return i
        return len(block.body)

def _warn_about_uninitialized_variable(varname, loc):
    if False:
        while True:
            i = 10
    if config.ALWAYS_WARN_UNINIT_VAR:
        warnings.warn(errors.NumbaWarning(f'Detected uninitialized variable {varname}', loc=loc))