__all__ = ['parent', 'reference', 'at', 'parents', 'children']
import gc
import sys
from ._dill import _proxy_helper as reference
from ._dill import _locate_object as at

def parent(obj, objtype, ignore=()):
    if False:
        return 10
    "\n>>> listiter = iter([4,5,6,7])\n>>> obj = parent(listiter, list)\n>>> obj == [4,5,6,7]  # actually 'is', but don't have handle any longer\nTrue\n\nNOTE: objtype can be a single type (e.g. int or list) or a tuple of types.\n\nWARNING: if obj is a sequence (e.g. list), may produce unexpected results.\nParent finds *one* parent (e.g. the last member of the sequence).\n    "
    depth = 1
    chain = parents(obj, objtype, depth, ignore)
    parent = chain.pop()
    if parent is obj:
        return None
    return parent

def parents(obj, objtype, depth=1, ignore=()):
    if False:
        while True:
            i = 10
    "Find the chain of referents for obj. Chain will end with obj.\n\n    objtype: an object type or tuple of types to search for\n    depth: search depth (e.g. depth=2 is 'grandparents')\n    ignore: an object or tuple of objects to ignore in the search\n    "
    edge_func = gc.get_referents
    predicate = lambda x: isinstance(x, objtype)
    ignore = (ignore,) if not hasattr(ignore, '__len__') else ignore
    ignore = (id(obj) for obj in ignore)
    chain = find_chain(obj, predicate, edge_func, depth)[::-1]
    return chain

def children(obj, objtype, depth=1, ignore=()):
    if False:
        print('Hello World!')
    "Find the chain of referrers for obj. Chain will start with obj.\n\n    objtype: an object type or tuple of types to search for\n    depth: search depth (e.g. depth=2 is 'grandchildren')\n    ignore: an object or tuple of objects to ignore in the search\n\n    NOTE: a common thing to ignore is all globals, 'ignore=(globals(),)'\n\n    NOTE: repeated calls may yield different results, as python stores\n    the last value in the special variable '_'; thus, it is often good\n    to execute something to replace '_' (e.g. >>> 1+1).\n    "
    edge_func = gc.get_referrers
    predicate = lambda x: isinstance(x, objtype)
    ignore = (ignore,) if not hasattr(ignore, '__len__') else ignore
    ignore = (id(obj) for obj in ignore)
    chain = find_chain(obj, predicate, edge_func, depth, ignore)
    return chain

def find_chain(obj, predicate, edge_func, max_depth=20, extra_ignore=()):
    if False:
        print('Hello World!')
    queue = [obj]
    depth = {id(obj): 0}
    parent = {id(obj): None}
    ignore = set(extra_ignore)
    ignore.add(id(extra_ignore))
    ignore.add(id(queue))
    ignore.add(id(depth))
    ignore.add(id(parent))
    ignore.add(id(ignore))
    ignore.add(id(sys._getframe()))
    ignore.add(id(sys._getframe(1)))
    gc.collect()
    while queue:
        target = queue.pop(0)
        if predicate(target):
            chain = [target]
            while parent[id(target)] is not None:
                target = parent[id(target)]
                chain.append(target)
            return chain
        tdepth = depth[id(target)]
        if tdepth < max_depth:
            referrers = edge_func(target)
            ignore.add(id(referrers))
            for source in referrers:
                if id(source) in ignore:
                    continue
                if id(source) not in depth:
                    depth[id(source)] = tdepth + 1
                    parent[id(source)] = target
                    queue.append(source)
    return [obj]
refobject = at