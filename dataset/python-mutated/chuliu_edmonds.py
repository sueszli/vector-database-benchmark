import numpy as np

def tarjan(tree):
    if False:
        for i in range(10):
            print('nop')
    'Finds the cycles in a dependency graph\n\n    The input should be a numpy array of integers,\n    where in the standard use case,\n    tree[i] is the head of node i.\n\n    tree[0] == 0 to represent the root\n\n    so for example, for the English sentence "This is a test",\n    the input is\n\n    [0 4 4 4 0]\n\n    "Arthritis makes my hip hurt"\n\n    [0 2 0 4 2 2]\n\n    The return is a list of cycles, where in cycle has True if the\n    node at that index is participating in the cycle.\n    So, for example, the previous examples both return empty lists,\n    whereas an input of\n      np.array([0, 3, 1, 2])\n    has an output of\n      [np.array([False,  True,  True,  True])]\n    '
    indices = -np.ones_like(tree)
    lowlinks = -np.ones_like(tree)
    onstack = np.zeros_like(tree, dtype=bool)
    stack = list()
    _index = [0]
    cycles = []

    def maybe_pop_cycle(i):
        if False:
            i = 10
            return i + 15
        if lowlinks[i] == indices[i]:
            cycle = np.zeros_like(indices, dtype=bool)
            while stack[-1] != i:
                j = stack.pop()
                onstack[j] = False
                cycle[j] = True
            stack.pop()
            onstack[i] = False
            cycle[i] = True
            if cycle.sum() > 1:
                cycles.append(cycle)

    def initialize_strong_connect(i):
        if False:
            i = 10
            return i + 15
        _index[0] += 1
        index = _index[-1]
        indices[i] = lowlinks[i] = index - 1
        stack.append(i)
        onstack[i] = True

    def strong_connect(i):
        if False:
            return 10
        call_stack = [(i, None, None)]
        while len(call_stack) > 0:
            (i, dependents_iterator, j) = call_stack.pop()
            if dependents_iterator is None:
                initialize_strong_connect(i)
                dependents_iterator = iter(np.where(np.equal(tree, i))[0])
            else:
                lowlinks[i] = min(lowlinks[i], lowlinks[j])
            for j in dependents_iterator:
                if indices[j] == -1:
                    call_stack.append((i, dependents_iterator, j))
                    call_stack.append((j, None, None))
                    break
                elif onstack[j]:
                    lowlinks[i] = min(lowlinks[i], indices[j])
            else:
                maybe_pop_cycle(i)
    for i in range(len(tree)):
        if indices[i] == -1:
            strong_connect(i)
    return cycles

def process_cycle(tree, cycle, scores):
    if False:
        print('Hello World!')
    '\n    Build a subproblem with one cycle broken\n    '
    cycle_locs = np.where(cycle)[0]
    cycle_subtree = tree[cycle]
    cycle_scores = scores[cycle, cycle_subtree]
    cycle_score = cycle_scores.sum()
    noncycle = np.logical_not(cycle)
    noncycle_locs = np.where(noncycle)[0]
    metanode_head_scores = scores[cycle][:, noncycle] - cycle_scores[:, None] + cycle_score
    metanode_dep_scores = scores[noncycle][:, cycle]
    metanode_heads = np.argmax(metanode_head_scores, axis=0)
    metanode_deps = np.argmax(metanode_dep_scores, axis=1)
    subscores = scores[noncycle][:, noncycle]
    subscores = np.pad(subscores, ((0, 1), (0, 1)), 'constant')
    subscores[-1, :-1] = metanode_head_scores[metanode_heads, np.arange(len(noncycle_locs))]
    subscores[:-1, -1] = metanode_dep_scores[np.arange(len(noncycle_locs)), metanode_deps]
    return (subscores, cycle_locs, noncycle_locs, metanode_heads, metanode_deps)

def expand_contracted_tree(tree, contracted_tree, cycle_locs, noncycle_locs, metanode_heads, metanode_deps):
    if False:
        for i in range(10):
            print('nop')
    '\n    Given a partially solved tree with a cycle and a solved subproblem\n    for the cycle, build a larger solution without the cycle\n    '
    cycle_head = contracted_tree[-1]
    contracted_tree = contracted_tree[:-1]
    new_tree = -np.ones_like(tree)
    contracted_subtree = contracted_tree < len(contracted_tree)
    new_tree[noncycle_locs[contracted_subtree]] = noncycle_locs[contracted_tree[contracted_subtree]]
    contracted_subtree = np.logical_not(contracted_subtree)
    new_tree[noncycle_locs[contracted_subtree]] = cycle_locs[metanode_deps[contracted_subtree]]
    new_tree[cycle_locs] = tree[cycle_locs]
    cycle_root = metanode_heads[cycle_head]
    new_tree[cycle_locs[cycle_root]] = noncycle_locs[cycle_head]
    return new_tree

def prepare_scores(scores):
    if False:
        i = 10
        return i + 15
    '\n    Alter the scores matrix to avoid self loops and handle the root\n    '
    np.fill_diagonal(scores, -float('inf'))
    scores[0] = -float('inf')
    scores[0, 0] = 0

def chuliu_edmonds(scores):
    if False:
        print('Hello World!')
    subtree_stack = []
    prepare_scores(scores)
    tree = np.argmax(scores, axis=1)
    cycles = tarjan(tree)
    while cycles:
        (subscores, cycle_locs, noncycle_locs, metanode_heads, metanode_deps) = process_cycle(tree, cycles.pop(), scores)
        subtree_stack.append((tree, cycles, scores, subscores, cycle_locs, noncycle_locs, metanode_heads, metanode_deps))
        scores = subscores
        prepare_scores(scores)
        tree = np.argmax(scores, axis=1)
        cycles = tarjan(tree)
    while len(subtree_stack) > 0:
        contracted_tree = tree
        (tree, cycles, scores, subscores, cycle_locs, noncycle_locs, metanode_heads, metanode_deps) = subtree_stack.pop()
        tree = expand_contracted_tree(tree, contracted_tree, cycle_locs, noncycle_locs, metanode_heads, metanode_deps)
    return tree

def chuliu_edmonds_one_root(scores):
    if False:
        while True:
            i = 10
    ''
    scores = scores.astype(np.float64)
    tree = chuliu_edmonds(scores)
    roots_to_try = np.where(np.equal(tree[1:], 0))[0] + 1
    if len(roots_to_try) == 1:
        return tree

    def set_root(scores, root):
        if False:
            while True:
                i = 10
        root_score = scores[root, 0]
        scores = np.array(scores)
        scores[1:, 0] = -float('inf')
        scores[root] = -float('inf')
        scores[root, 0] = 0
        return (scores, root_score)
    (best_score, best_tree) = (-np.inf, None)
    for root in roots_to_try:
        (_scores, root_score) = set_root(scores, root)
        _tree = chuliu_edmonds(_scores)
        tree_probs = _scores[np.arange(len(_scores)), _tree]
        tree_score = tree_probs.sum() + root_score if (tree_probs > -np.inf).all() else -np.inf
        if tree_score > best_score:
            best_score = tree_score
            best_tree = _tree
    try:
        assert best_tree is not None
    except:
        with open('debug.log', 'w') as f:
            f.write('{}: {}, {}\n'.format(tree, scores, roots_to_try))
            f.write('{}: {}, {}, {}\n'.format(_tree, _scores, tree_probs, tree_score))
        raise
    return best_tree