"""Topological sorting implementation."""
from functools import reduce as _reduce
from logging import getLogger
log = getLogger(__name__)

def _toposort(data):
    if False:
        i = 10
        return i + 15
    'Dependencies are expressed as a dictionary whose keys are items\n    and whose values are a set of dependent items. Output is a list of\n    sets in topological order. The first set consists of items with no\n    dependences, each subsequent set consists of items that depend upon\n    items in the preceding sets.\n    '
    if len(data) == 0:
        return
    for (k, v) in data.items():
        v.discard(k)
    extra_items_in_deps = _reduce(set.union, data.values()) - set(data.keys())
    data.update({item: set() for item in extra_items_in_deps})
    while True:
        ordered = sorted({item for (item, dep) in data.items() if len(dep) == 0})
        if not ordered:
            break
        for item in ordered:
            yield item
            data.pop(item, None)
        for dep in sorted(data.values()):
            dep -= set(ordered)
    if len(data) != 0:
        from ..exceptions import CondaValueError
        msg = 'Cyclic dependencies exist among these items: {}'
        raise CondaValueError(msg.format(' -> '.join((repr(x) for x in data.keys()))))

def pop_key(data):
    if False:
        i = 10
        return i + 15
    '\n    Pop an item from the graph that has the fewest dependencies in the case of a tie\n    The winners will be sorted alphabetically\n    '
    items = sorted(data.items(), key=lambda item: (len(item[1]), item[0]))
    key = items[0][0]
    data.pop(key)
    for dep in data.values():
        dep.discard(key)
    return key

def _safe_toposort(data):
    if False:
        return 10
    'Dependencies are expressed as a dictionary whose keys are items\n    and whose values are a set of dependent items. Output is a list of\n    sets in topological order. The first set consists of items with no\n    dependencies, each subsequent set consists of items that depend upon\n    items in the preceding sets.\n    '
    if len(data) == 0:
        return
    t = _toposort(data)
    while True:
        try:
            value = next(t)
            yield value
        except ValueError as err:
            log.debug(err.args[0])
            if not data:
                return
            yield pop_key(data)
            t = _toposort(data)
            continue
        except StopIteration:
            return

def toposort(data, safe=True):
    if False:
        i = 10
        return i + 15
    data = {k: set(v) for (k, v) in data.items()}
    if 'python' in data:
        data['python'].discard('pip')
    if safe:
        return list(_safe_toposort(data))
    else:
        return list(_toposort(data))