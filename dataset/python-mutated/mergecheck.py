"""Check for duplicate AST nodes after merge."""
from __future__ import annotations
from typing import Final
from mypy.nodes import Decorator, FakeInfo, FuncDef, SymbolNode, Var
from mypy.server.objgraph import get_path, get_reachable_graph
DUMP_MISMATCH_NODES: Final = False

def check_consistency(o: object) -> None:
    if False:
        return 10
    "Fail if there are two AST nodes with the same fullname reachable from 'o'.\n\n    Raise AssertionError on failure and print some debugging output.\n    "
    (seen, parents) = get_reachable_graph(o)
    reachable = list(seen.values())
    syms = [x for x in reachable if isinstance(x, SymbolNode)]
    m: dict[str, SymbolNode] = {}
    for sym in syms:
        if isinstance(sym, FakeInfo):
            continue
        fn = sym.fullname
        if fn is None:
            continue
        if isinstance(sym, (Var, Decorator)):
            continue
        if isinstance(sym, FuncDef) and sym.is_overload:
            continue
        if fn not in m:
            m[sym.fullname] = sym
            continue
        (sym1, sym2) = (sym, m[fn])
        if type(sym1) is not type(sym2):
            continue
        path1 = get_path(sym1, seen, parents)
        path2 = get_path(sym2, seen, parents)
        if fn in m:
            print(f'\nDuplicate {type(sym).__name__!r} nodes with fullname {fn!r} found:')
            print('[1] %d: %s' % (id(sym1), path_to_str(path1)))
            print('[2] %d: %s' % (id(sym2), path_to_str(path2)))
        if DUMP_MISMATCH_NODES and fn in m:
            print('---')
            print(id(sym1), sym1)
            print('---')
            print(id(sym2), sym2)
        assert sym.fullname not in m

def path_to_str(path: list[tuple[object, object]]) -> str:
    if False:
        for i in range(10):
            print('nop')
    result = '<root>'
    for (attr, obj) in path:
        t = type(obj).__name__
        if t in ('dict', 'tuple', 'SymbolTable', 'list'):
            result += f'[{repr(attr)}]'
        elif isinstance(obj, Var):
            result += f'.{attr}({t}:{obj.name})'
        elif t in ('BuildManager', 'FineGrainedBuildManager'):
            result += f'.{attr}'
        else:
            result += f'.{attr}({t})'
    return result