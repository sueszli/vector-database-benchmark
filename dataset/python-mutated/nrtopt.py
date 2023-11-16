"""
NRT specific optimizations
"""
import re
from collections import defaultdict, deque
from llvmlite import binding as ll
from numba.core import cgutils
_regex_incref = re.compile('\\s*(?:tail)?\\s*call void @NRT_incref\\((.*)\\)')
_regex_decref = re.compile('\\s*(?:tail)?\\s*call void @NRT_decref\\((.*)\\)')
_regex_bb = re.compile('|'.join(['[0-9]+:', '[\\\'"]?[-a-zA-Z$._0-9][-a-zA-Z$._0-9]*[\\\'"]?:', '^define', '^;\\s*<label>']))

def _remove_redundant_nrt_refct(llvmir):
    if False:
        i = 10
        return i + 15

    def _extract_functions(module):
        if False:
            return 10
        cur = []
        for line in str(module).splitlines():
            if line.startswith('define'):
                assert not cur
                cur.append(line)
            elif line.startswith('}'):
                assert cur
                cur.append(line)
                yield (True, cur)
                cur = []
            elif cur:
                cur.append(line)
            else:
                yield (False, [line])

    def _process_function(func_lines):
        if False:
            print('Hello World!')
        out = []
        for (is_bb, bb_lines) in _extract_basic_blocks(func_lines):
            if is_bb and bb_lines:
                bb_lines = _process_basic_block(bb_lines)
            out += bb_lines
        return out

    def _extract_basic_blocks(func_lines):
        if False:
            return 10
        assert func_lines[0].startswith('define')
        assert func_lines[-1].startswith('}')
        yield (False, [func_lines[0]])
        cur = []
        for ln in func_lines[1:-1]:
            m = _regex_bb.match(ln)
            if m is not None:
                yield (True, cur)
                cur = []
                yield (False, [ln])
            elif ln:
                cur.append(ln)
        yield (True, cur)
        yield (False, [func_lines[-1]])

    def _process_basic_block(bb_lines):
        if False:
            print('Hello World!')
        bb_lines = _move_and_group_decref_after_all_increfs(bb_lines)
        bb_lines = _prune_redundant_refct_ops(bb_lines)
        return bb_lines

    def _examine_refct_op(bb_lines):
        if False:
            for i in range(10):
                print('nop')
        for (num, ln) in enumerate(bb_lines):
            m = _regex_incref.match(ln)
            if m is not None:
                yield (num, m.group(1), None)
                continue
            m = _regex_decref.match(ln)
            if m is not None:
                yield (num, None, m.group(1))
                continue
            yield (ln, None, None)

    def _prune_redundant_refct_ops(bb_lines):
        if False:
            i = 10
            return i + 15
        incref_map = defaultdict(deque)
        decref_map = defaultdict(deque)
        to_remove = set()
        for (num, incref_var, decref_var) in _examine_refct_op(bb_lines):
            assert not (incref_var and decref_var)
            if incref_var:
                if incref_var == 'i8* null':
                    to_remove.add(num)
                else:
                    incref_map[incref_var].append(num)
            elif decref_var:
                if decref_var == 'i8* null':
                    to_remove.add(num)
                else:
                    decref_map[decref_var].append(num)
        for (var, decops) in decref_map.items():
            incops = incref_map[var]
            ct = min(len(incops), len(decops))
            for _ in range(ct):
                to_remove.add(incops.pop())
                to_remove.add(decops.popleft())
        return [ln for (num, ln) in enumerate(bb_lines) if num not in to_remove]

    def _move_and_group_decref_after_all_increfs(bb_lines):
        if False:
            while True:
                i = 10
        last_incref_pos = 0
        for (pos, ln) in enumerate(bb_lines):
            if _regex_incref.match(ln) is not None:
                last_incref_pos = pos + 1
        last_decref_pos = 0
        for (pos, ln) in enumerate(bb_lines):
            if _regex_decref.match(ln) is not None:
                last_decref_pos = pos + 1
        last_pos = max(last_incref_pos, last_decref_pos)
        decrefs = []
        head = []
        for ln in bb_lines[:last_pos]:
            if _regex_decref.match(ln) is not None:
                decrefs.append(ln)
            else:
                head.append(ln)
        return head + decrefs + bb_lines[last_pos:]
    processed = []
    for (is_func, lines) in _extract_functions(llvmir):
        if is_func:
            lines = _process_function(lines)
        processed += lines
    return '\n'.join(processed)

def remove_redundant_nrt_refct(ll_module):
    if False:
        return 10
    "\n    Remove redundant reference count operations from the\n    `llvmlite.binding.ModuleRef`. This parses the ll_module as a string and\n    line by line to remove the unnecessary nrt refct pairs within each block.\n    Decref calls are moved after the last incref call in the block to avoid\n    temporarily decref'ing to zero (which can happen due to hidden decref from\n    alias).\n\n    Note: non-threadsafe due to usage of global LLVMcontext\n    "
    try:
        ll_module.get_function('NRT_incref')
    except NameError:
        return ll_module
    name = ll_module.name
    newll = _remove_redundant_nrt_refct(str(ll_module))
    new_mod = ll.parse_assembly(newll)
    new_mod.name = cgutils.normalize_ir_text(name)
    return new_mod