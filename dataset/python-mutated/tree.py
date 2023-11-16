def pprint_nodes(subtrees):
    if False:
        for i in range(10):
            print('nop')
    '\n    Prettyprints systems of nodes.\n\n    Examples\n    ========\n\n    >>> from sympy.printing.tree import pprint_nodes\n    >>> print(pprint_nodes(["a", "b1\\nb2", "c"]))\n    +-a\n    +-b1\n    | b2\n    +-c\n\n    '

    def indent(s, type=1):
        if False:
            while True:
                i = 10
        x = s.split('\n')
        r = '+-%s\n' % x[0]
        for a in x[1:]:
            if a == '':
                continue
            if type == 1:
                r += '| %s\n' % a
            else:
                r += '  %s\n' % a
        return r
    if not subtrees:
        return ''
    f = ''
    for a in subtrees[:-1]:
        f += indent(a)
    f += indent(subtrees[-1], 2)
    return f

def print_node(node, assumptions=True):
    if False:
        return 10
    '\n    Returns information about the "node".\n\n    This includes class name, string representation and assumptions.\n\n    Parameters\n    ==========\n\n    assumptions : bool, optional\n        See the ``assumptions`` keyword in ``tree``\n    '
    s = '%s: %s\n' % (node.__class__.__name__, str(node))
    if assumptions:
        d = node._assumptions
    else:
        d = None
    if d:
        for a in sorted(d):
            v = d[a]
            if v is None:
                continue
            s += '%s: %s\n' % (a, v)
    return s

def tree(node, assumptions=True):
    if False:
        print('Hello World!')
    '\n    Returns a tree representation of "node" as a string.\n\n    It uses print_node() together with pprint_nodes() on node.args recursively.\n\n    Parameters\n    ==========\n\n    asssumptions : bool, optional\n        The flag to decide whether to print out all the assumption data\n        (such as ``is_integer`, ``is_real``) associated with the\n        expression or not.\n\n        Enabling the flag makes the result verbose, and the printed\n        result may not be determinisitic because of the randomness used\n        in backtracing the assumptions.\n\n    See Also\n    ========\n\n    print_tree\n\n    '
    subtrees = []
    for arg in node.args:
        subtrees.append(tree(arg, assumptions=assumptions))
    s = print_node(node, assumptions=assumptions) + pprint_nodes(subtrees)
    return s

def print_tree(node, assumptions=True):
    if False:
        while True:
            i = 10
    '\n    Prints a tree representation of "node".\n\n    Parameters\n    ==========\n\n    asssumptions : bool, optional\n        The flag to decide whether to print out all the assumption data\n        (such as ``is_integer`, ``is_real``) associated with the\n        expression or not.\n\n        Enabling the flag makes the result verbose, and the printed\n        result may not be determinisitic because of the randomness used\n        in backtracing the assumptions.\n\n    Examples\n    ========\n\n    >>> from sympy.printing import print_tree\n    >>> from sympy import Symbol\n    >>> x = Symbol(\'x\', odd=True)\n    >>> y = Symbol(\'y\', even=True)\n\n    Printing with full assumptions information:\n\n    >>> print_tree(y**x)\n    Pow: y**x\n    +-Symbol: y\n    | algebraic: True\n    | commutative: True\n    | complex: True\n    | even: True\n    | extended_real: True\n    | finite: True\n    | hermitian: True\n    | imaginary: False\n    | infinite: False\n    | integer: True\n    | irrational: False\n    | noninteger: False\n    | odd: False\n    | rational: True\n    | real: True\n    | transcendental: False\n    +-Symbol: x\n      algebraic: True\n      commutative: True\n      complex: True\n      even: False\n      extended_nonzero: True\n      extended_real: True\n      finite: True\n      hermitian: True\n      imaginary: False\n      infinite: False\n      integer: True\n      irrational: False\n      noninteger: False\n      nonzero: True\n      odd: True\n      rational: True\n      real: True\n      transcendental: False\n      zero: False\n\n    Hiding the assumptions:\n\n    >>> print_tree(y**x, assumptions=False)\n    Pow: y**x\n    +-Symbol: y\n    +-Symbol: x\n\n    See Also\n    ========\n\n    tree\n\n    '
    print(tree(node, assumptions=assumptions))