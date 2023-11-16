"""Utilities for working with the CFG."""
import collections
import itertools
DEEP_VARIABLE_LIMIT = 1024

def variable_product(variables):
    if False:
        i = 10
        return i + 15
    'Take the Cartesian product of a number of Variables.\n\n  Args:\n    variables: A sequence of Variables.\n\n  Returns:\n    A list of lists of Values, where each sublist has one element from each\n    of the given Variables.\n  '
    return itertools.product(*(v.bindings for v in variables))

def _variable_product_items(variableitems, complexity_limit):
    if False:
        for i in range(10):
            print('nop')
    "Take the Cartesian product of a list of (key, value) tuples.\n\n  See variable_product_dict below.\n\n  Args:\n    variableitems: A dict mapping object to cfg.Variable.\n    complexity_limit: A counter that tracks how many combinations we've yielded\n      and aborts if we go over the limit.\n\n  Yields:\n    A sequence of [(key, cfg.Binding), ...] lists.\n  "
    variableitems_iter = iter(variableitems)
    try:
        (headkey, headvar) = next(variableitems_iter)
    except StopIteration:
        yield []
    else:
        for tail in _variable_product_items(variableitems_iter, complexity_limit):
            for headvalue in headvar.bindings:
                complexity_limit.inc()
                yield ([(headkey, headvalue)] + tail)

class TooComplexError(Exception):
    """Thrown if we determine that something in our program is too complex."""

class ComplexityLimit:
    """A class that raises TooComplexError if we hit a limit."""

    def __init__(self, limit):
        if False:
            print('Hello World!')
        self.limit = limit
        self.count = 0

    def inc(self, add=1):
        if False:
            print('Hello World!')
        self.count += add
        if self.count >= self.limit:
            raise TooComplexError()

def deep_variable_product(variables, limit=DEEP_VARIABLE_LIMIT):
    if False:
        for i in range(10):
            print('nop')
    "Take the deep Cartesian product of a list of Variables.\n\n  For example:\n    x1.children = {v2, v3}\n    v1 = {x1, x2}\n    v2 = {x3}\n    v3 = {x4, x5}\n    v4 = {x6}\n  then\n    deep_variable_product([v1, v4]) will return:\n      [[x1, x3, x4, x6],\n       [x1, x3, x5, x6],\n       [x2, x6]]\n  .\n  Args:\n    variables: A sequence of Variables.\n    limit: How many results we allow before aborting.\n\n  Returns:\n    A list of lists of Values, where each sublist has one Value from each\n    of the corresponding Variables and the Variables of their Values' children.\n\n  Raises:\n    TooComplexError: If we expanded too many values.\n  "
    return _deep_values_list_product([v.bindings for v in variables], set(), ComplexityLimit(limit))

def _deep_values_list_product(values_list, seen, complexity_limit):
    if False:
        return 10
    'Take the deep Cartesian product of a list of list of Values.'
    result = []
    for row in itertools.product(*(values for values in values_list if values)):
        extra_params = [value for entry in row if entry not in seen for value in entry.data.unique_parameter_values()]
        extra_values = extra_params and _deep_values_list_product(extra_params, seen.union(row), complexity_limit)
        if extra_values:
            for new_row in extra_values:
                result.append(row + new_row)
        else:
            complexity_limit.inc()
            result.append(row)
    return result

def variable_product_dict(variabledict, limit=DEEP_VARIABLE_LIMIT):
    if False:
        i = 10
        return i + 15
    'Take the Cartesian product of variables in the values of a dict.\n\n  This Cartesian product is taken using the dict keys as the indices into the\n  input and output dicts. So:\n    variable_product_dict({"x": Variable(a, b), "y": Variable(c, d)})\n      ==\n    [{"x": a, "y": c}, {"x": a, "y": d}, {"x": b, "y": c}, {"x": b, "y": d}]\n  This is exactly analogous to a traditional Cartesian product except that\n  instead of trying each possible value of a numbered position, we are trying\n  each possible value of a named position.\n\n  Args:\n    variabledict: A dict with variable values.\n    limit: How many results to allow before aborting.\n\n  Returns:\n    A list of dicts with Value values.\n  '
    return [dict(d) for d in _variable_product_items(variabledict.items(), ComplexityLimit(limit))]

def merge_variables(program, node, variables):
    if False:
        return 10
    'Create a combined Variable for a list of variables.\n\n  The purpose of this function is to create a final result variable for\n  functions that return a list of "temporary" variables. (E.g. function\n  calls).\n\n  Args:\n    program: A cfg.Program instance.\n    node: The current CFG node.\n    variables: A list of cfg.Variables.\n  Returns:\n    A cfg.Variable.\n  '
    if not variables:
        return program.NewVariable()
    elif all((v is variables[0] for v in variables)):
        return variables[0].AssignToNewVariable(node)
    else:
        v = program.NewVariable()
        for r in variables:
            v.PasteVariable(r, node)
        return v

def merge_bindings(program, node, bindings):
    if False:
        print('Hello World!')
    'Create a combined Variable for a list of bindings.\n\n  Args:\n    program: A cfg.Program instance.\n    node: The current CFG node.\n    bindings: A list of cfg.Bindings.\n  Returns:\n    A cfg.Variable.\n  '
    v = program.NewVariable()
    for b in bindings:
        v.PasteBinding(b, node)
    return v

def walk_binding(binding, keep_binding=lambda _: True):
    if False:
        return 10
    "Helper function to walk a binding's origins.\n\n  Args:\n    binding: A cfg.Binding.\n    keep_binding: Optionally, a function, cfg.Binding -> bool, specifying\n      whether to keep each binding found.\n\n  Yields:\n    A cfg.Origin. The caller must send the origin back into the generator. To\n    stop exploring the origin, send None back.\n  "
    bindings = [binding]
    seen = set()
    while bindings:
        b = bindings.pop(0)
        if b in seen or not keep_binding(b):
            continue
        seen.add(b)
        for o in b.origins:
            o = (yield o)
            if o:
                bindings.extend(itertools.chain(*o.source_sets))

def compute_predecessors(nodes):
    if False:
        return 10
    'Build a transitive closure.\n\n  For a list of nodes, compute all the predecessors of each node.\n\n  Args:\n    nodes: A list of nodes or blocks.\n  Returns:\n    A dictionary that maps each node to a set of all the nodes that can reach\n    that node.\n  '
    predecessors = {n: {n} for n in nodes}
    discovered = set()
    for start in nodes:
        if start in discovered:
            continue
        unprocessed = [(start, n) for n in start.outgoing]
        while unprocessed:
            (from_node, node) = unprocessed.pop(0)
            node_predecessors = predecessors[node]
            length_before = len(node_predecessors)
            node_predecessors |= predecessors[from_node]
            if length_before != len(node_predecessors):
                unprocessed.extend(((node, n) for n in node.outgoing))
                discovered.add(node)
    return predecessors

def order_nodes(nodes):
    if False:
        return 10
    'Build an ancestors first traversal of CFG nodes.\n\n  This guarantees that at least one predecessor of a block is scheduled before\n  the block itself, and it also tries to schedule as many of them before the\n  block as possible (so e.g. if two branches merge in a node, it prefers to\n  process both the branches before that node).\n\n  Args:\n    nodes: A list of nodes or blocks. They have two attributes: "incoming" and\n      "outgoing". Both are lists of other nodes.\n  Returns:\n    A list of nodes in the proper order.\n  '
    if not nodes:
        return []
    root = nodes[0]
    predecessor_map = compute_predecessors(nodes)
    dead = {node for (node, predecessors) in predecessor_map.items() if root not in predecessors}
    queue = {root: predecessor_map[root]}
    order = []
    seen = set()
    while queue:
        (_, _, node) = min(((len(predecessors), node.id, node) for (node, predecessors) in queue.items()))
        del queue[node]
        if node in seen:
            continue
        order.append(node)
        seen.add(node)
        for (_, predecessors) in queue.items():
            predecessors.discard(node)
        for n in node.outgoing:
            if n not in queue:
                queue[n] = predecessor_map[n] - seen
    assert len(set(order) | dead) == len(set(nodes))
    return order

def topological_sort(nodes):
    if False:
        while True:
            i = 10
    'Sort a list of nodes topologically.\n\n  This will order the nodes so that any node that appears in the "incoming"\n  list of another node n2 will appear in the output before n2. It assumes that\n  the graph doesn\'t have any cycles.\n  If there are multiple ways to sort the list, a random one is picked.\n\n  Args:\n    nodes: A sequence of nodes. Each node may have an attribute "incoming",\n      a list of nodes (every node in this list needs to be in "nodes"). If\n      "incoming" is not there, it\'s assumed to be empty. The list of nodes\n      can\'t have duplicates.\n  Yields:\n    The nodes in their topological order.\n  Raises:\n    ValueError: If the graph contains a cycle.\n  '
    incoming = {node: set(getattr(node, 'incoming', ())) for node in nodes}
    outgoing = collections.defaultdict(set)
    for node in nodes:
        for inc in incoming[node]:
            outgoing[inc].add(node)
    stack = [node for node in nodes if not incoming[node]]
    for _ in nodes:
        if not stack:
            raise ValueError('Circular graph')
        leaf = stack.pop()
        yield leaf
        for out in outgoing[leaf]:
            incoming[out].remove(leaf)
            if not incoming[out]:
                stack.append(out)
    assert not stack