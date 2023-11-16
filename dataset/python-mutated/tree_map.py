from functorch._C import dim
tree_flatten = dim.tree_flatten

def tree_map(fn, tree):
    if False:
        return 10
    (vs, unflatten) = tree_flatten(tree)
    return unflatten((fn(v) for v in vs))