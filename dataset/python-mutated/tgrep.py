"""
============================================
 TGrep search implementation for NLTK trees
============================================

This module supports TGrep2 syntax for matching parts of NLTK Trees.
Note that many tgrep operators require the tree passed to be a
``ParentedTree``.

External links:

- `Tgrep tutorial <https://www.stanford.edu/dept/linguistics/corpora/cas-tut-tgrep.html>`_
- `Tgrep2 manual <http://tedlab.mit.edu/~dr/Tgrep2/tgrep2.pdf>`_
- `Tgrep2 source <http://tedlab.mit.edu/~dr/Tgrep2/>`_

Usage
=====

>>> from nltk.tree import ParentedTree
>>> from nltk.tgrep import tgrep_nodes, tgrep_positions
>>> tree = ParentedTree.fromstring('(S (NP (DT the) (JJ big) (NN dog)) (VP bit) (NP (DT a) (NN cat)))')
>>> list(tgrep_nodes('NN', [tree]))
[[ParentedTree('NN', ['dog']), ParentedTree('NN', ['cat'])]]
>>> list(tgrep_positions('NN', [tree]))
[[(0, 2), (2, 1)]]
>>> list(tgrep_nodes('DT', [tree]))
[[ParentedTree('DT', ['the']), ParentedTree('DT', ['a'])]]
>>> list(tgrep_nodes('DT $ JJ', [tree]))
[[ParentedTree('DT', ['the'])]]

This implementation adds syntax to select nodes based on their NLTK
tree position.  This syntax is ``N`` plus a Python tuple representing
the tree position.  For instance, ``N()``, ``N(0,)``, ``N(0,0)`` are
valid node selectors.  Example:

>>> tree = ParentedTree.fromstring('(S (NP (DT the) (JJ big) (NN dog)) (VP bit) (NP (DT a) (NN cat)))')
>>> tree[0,0]
ParentedTree('DT', ['the'])
>>> tree[0,0].treeposition()
(0, 0)
>>> list(tgrep_nodes('N(0,0)', [tree]))
[[ParentedTree('DT', ['the'])]]

Caveats:
========

- Link modifiers: "?" and "=" are not implemented.
- Tgrep compatibility: Using "@" for "!", "{" for "<", "}" for ">" are
  not implemented.
- The "=" and "~" links are not implemented.

Known Issues:
=============

- There are some issues with link relations involving leaf nodes
  (which are represented as bare strings in NLTK trees).  For
  instance, consider the tree::

      (S (A x))

  The search string ``* !>> S`` should select all nodes which are not
  dominated in some way by an ``S`` node (i.e., all nodes which are
  not descendants of an ``S``).  Clearly, in this tree, the only node
  which fulfills this criterion is the top node (since it is not
  dominated by anything).  However, the code here will find both the
  top node and the leaf node ``x``.  This is because we cannot recover
  the parent of the leaf, since it is stored as a bare string.

  A possible workaround, when performing this kind of search, would be
  to filter out all leaf nodes.

Implementation notes
====================

This implementation is (somewhat awkwardly) based on lambda functions
which are predicates on a node.  A predicate is a function which is
either True or False; using a predicate function, we can identify sets
of nodes with particular properties.  A predicate function, could, for
instance, return True only if a particular node has a label matching a
particular regular expression, and has a daughter node which has no
sisters.  Because tgrep2 search strings can do things statefully (such
as substituting in macros, and binding nodes with node labels), the
actual predicate function is declared with three arguments::

    pred = lambda n, m, l: return True # some logic here

``n``
    is a node in a tree; this argument must always be given

``m``
    contains a dictionary, mapping macro names onto predicate functions

``l``
    is a dictionary to map node labels onto nodes in the tree

``m`` and ``l`` are declared to default to ``None``, and so need not be
specified in a call to a predicate.  Predicates which call other
predicates must always pass the value of these arguments on.  The
top-level predicate (constructed by ``_tgrep_exprs_action``) binds the
macro definitions to ``m`` and initialises ``l`` to an empty dictionary.
"""
import functools
import re
try:
    import pyparsing
except ImportError:
    print('Warning: nltk.tgrep will not work without the `pyparsing` package')
    print('installed.')
import nltk.tree

class TgrepException(Exception):
    """Tgrep exception type."""
    pass

def ancestors(node):
    if False:
        while True:
            i = 10
    '\n    Returns the list of all nodes dominating the given tree node.\n    This method will not work with leaf nodes, since there is no way\n    to recover the parent.\n    '
    results = []
    try:
        current = node.parent()
    except AttributeError:
        return results
    while current:
        results.append(current)
        current = current.parent()
    return results

def unique_ancestors(node):
    if False:
        return 10
    '\n    Returns the list of all nodes dominating the given node, where\n    there is only a single path of descent.\n    '
    results = []
    try:
        current = node.parent()
    except AttributeError:
        return results
    while current and len(current) == 1:
        results.append(current)
        current = current.parent()
    return results

def _descendants(node):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns the list of all nodes which are descended from the given\n    tree node in some way.\n    '
    try:
        treepos = node.treepositions()
    except AttributeError:
        return []
    return [node[x] for x in treepos[1:]]

def _leftmost_descendants(node):
    if False:
        print('Hello World!')
    '\n    Returns the set of all nodes descended in some way through\n    left branches from this node.\n    '
    try:
        treepos = node.treepositions()
    except AttributeError:
        return []
    return [node[x] for x in treepos[1:] if all((y == 0 for y in x))]

def _rightmost_descendants(node):
    if False:
        while True:
            i = 10
    '\n    Returns the set of all nodes descended in some way through\n    right branches from this node.\n    '
    try:
        rightmost_leaf = max(node.treepositions())
    except AttributeError:
        return []
    return [node[rightmost_leaf[:i]] for i in range(1, len(rightmost_leaf) + 1)]

def _istree(obj):
    if False:
        return 10
    'Predicate to check whether `obj` is a nltk.tree.Tree.'
    return isinstance(obj, nltk.tree.Tree)

def _unique_descendants(node):
    if False:
        return 10
    '\n    Returns the list of all nodes descended from the given node, where\n    there is only a single path of descent.\n    '
    results = []
    current = node
    while current and _istree(current) and (len(current) == 1):
        current = current[0]
        results.append(current)
    return results

def _before(node):
    if False:
        while True:
            i = 10
    '\n    Returns the set of all nodes that are before the given node.\n    '
    try:
        pos = node.treeposition()
        tree = node.root()
    except AttributeError:
        return []
    return [tree[x] for x in tree.treepositions() if x[:len(pos)] < pos[:len(x)]]

def _immediately_before(node):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns the set of all nodes that are immediately before the given\n    node.\n\n    Tree node A immediately precedes node B if the last terminal\n    symbol (word) produced by A immediately precedes the first\n    terminal symbol produced by B.\n    '
    try:
        pos = node.treeposition()
        tree = node.root()
    except AttributeError:
        return []
    idx = len(pos) - 1
    while 0 <= idx and pos[idx] == 0:
        idx -= 1
    if idx < 0:
        return []
    pos = list(pos[:idx + 1])
    pos[-1] -= 1
    before = tree[pos]
    return [before] + _rightmost_descendants(before)

def _after(node):
    if False:
        while True:
            i = 10
    '\n    Returns the set of all nodes that are after the given node.\n    '
    try:
        pos = node.treeposition()
        tree = node.root()
    except AttributeError:
        return []
    return [tree[x] for x in tree.treepositions() if x[:len(pos)] > pos[:len(x)]]

def _immediately_after(node):
    if False:
        while True:
            i = 10
    '\n    Returns the set of all nodes that are immediately after the given\n    node.\n\n    Tree node A immediately follows node B if the first terminal\n    symbol (word) produced by A immediately follows the last\n    terminal symbol produced by B.\n    '
    try:
        pos = node.treeposition()
        tree = node.root()
        current = node.parent()
    except AttributeError:
        return []
    idx = len(pos) - 1
    while 0 <= idx and pos[idx] == len(current) - 1:
        idx -= 1
        current = current.parent()
    if idx < 0:
        return []
    pos = list(pos[:idx + 1])
    pos[-1] += 1
    after = tree[pos]
    return [after] + _leftmost_descendants(after)

def _tgrep_node_literal_value(node):
    if False:
        print('Hello World!')
    '\n    Gets the string value of a given parse tree node, for comparison\n    using the tgrep node literal predicates.\n    '
    return node.label() if _istree(node) else str(node)

def _tgrep_macro_use_action(_s, _l, tokens):
    if False:
        while True:
            i = 10
    '\n    Builds a lambda function which looks up the macro name used.\n    '
    assert len(tokens) == 1
    assert tokens[0][0] == '@'
    macro_name = tokens[0][1:]

    def macro_use(n, m=None, l=None):
        if False:
            while True:
                i = 10
        if m is None or macro_name not in m:
            raise TgrepException(f'macro {macro_name} not defined')
        return m[macro_name](n, m, l)
    return macro_use

def _tgrep_node_action(_s, _l, tokens):
    if False:
        i = 10
        return i + 15
    '\n    Builds a lambda function representing a predicate on a tree node\n    depending on the name of its node.\n    '
    if tokens[0] == "'":
        tokens = tokens[1:]
    if len(tokens) > 1:
        assert list(set(tokens[1::2])) == ['|']
        tokens = [_tgrep_node_action(None, None, [node]) for node in tokens[::2]]
        return (lambda t: lambda n, m=None, l=None: any((f(n, m, l) for f in t)))(tokens)
    elif hasattr(tokens[0], '__call__'):
        return tokens[0]
    elif tokens[0] == '*' or tokens[0] == '__':
        return lambda n, m=None, l=None: True
    elif tokens[0].startswith('"'):
        assert tokens[0].endswith('"')
        node_lit = tokens[0][1:-1].replace('\\"', '"').replace('\\\\', '\\')
        return (lambda s: lambda n, m=None, l=None: _tgrep_node_literal_value(n) == s)(node_lit)
    elif tokens[0].startswith('/'):
        assert tokens[0].endswith('/')
        node_lit = tokens[0][1:-1]
        return (lambda r: lambda n, m=None, l=None: r.search(_tgrep_node_literal_value(n)))(re.compile(node_lit))
    elif tokens[0].startswith('i@'):
        node_func = _tgrep_node_action(_s, _l, [tokens[0][2:].lower()])
        return (lambda f: lambda n, m=None, l=None: f(_tgrep_node_literal_value(n).lower()))(node_func)
    else:
        return (lambda s: lambda n, m=None, l=None: _tgrep_node_literal_value(n) == s)(tokens[0])

def _tgrep_parens_action(_s, _l, tokens):
    if False:
        i = 10
        return i + 15
    '\n    Builds a lambda function representing a predicate on a tree node\n    from a parenthetical notation.\n    '
    assert len(tokens) == 3
    assert tokens[0] == '('
    assert tokens[2] == ')'
    return tokens[1]

def _tgrep_nltk_tree_pos_action(_s, _l, tokens):
    if False:
        return 10
    '\n    Builds a lambda function representing a predicate on a tree node\n    which returns true if the node is located at a specific tree\n    position.\n    '
    node_tree_position = tuple((int(x) for x in tokens if x.isdigit()))
    return (lambda i: lambda n, m=None, l=None: hasattr(n, 'treeposition') and n.treeposition() == i)(node_tree_position)

def _tgrep_relation_action(_s, _l, tokens):
    if False:
        i = 10
        return i + 15
    '\n    Builds a lambda function representing a predicate on a tree node\n    depending on its relation to other nodes in the tree.\n    '
    negated = False
    if tokens[0] == '!':
        negated = True
        tokens = tokens[1:]
    if tokens[0] == '[':
        assert len(tokens) == 3
        assert tokens[2] == ']'
        retval = tokens[1]
    else:
        assert len(tokens) == 2
        (operator, predicate) = tokens
        if operator == '<':
            retval = lambda n, m=None, l=None: _istree(n) and any((predicate(x, m, l) for x in n))
        elif operator == '>':
            retval = lambda n, m=None, l=None: hasattr(n, 'parent') and bool(n.parent()) and predicate(n.parent(), m, l)
        elif operator == '<,' or operator == '<1':
            retval = lambda n, m=None, l=None: _istree(n) and bool(list(n)) and predicate(n[0], m, l)
        elif operator == '>,' or operator == '>1':
            retval = lambda n, m=None, l=None: hasattr(n, 'parent') and bool(n.parent()) and (n is n.parent()[0]) and predicate(n.parent(), m, l)
        elif operator[0] == '<' and operator[1:].isdigit():
            idx = int(operator[1:])
            retval = (lambda i: lambda n, m=None, l=None: _istree(n) and bool(list(n)) and (0 <= i < len(n)) and predicate(n[i], m, l))(idx - 1)
        elif operator[0] == '>' and operator[1:].isdigit():
            idx = int(operator[1:])
            retval = (lambda i: lambda n, m=None, l=None: hasattr(n, 'parent') and bool(n.parent()) and (0 <= i < len(n.parent())) and (n is n.parent()[i]) and predicate(n.parent(), m, l))(idx - 1)
        elif operator == "<'" or operator == '<-' or operator == '<-1':
            retval = lambda n, m=None, l=None: _istree(n) and bool(list(n)) and predicate(n[-1], m, l)
        elif operator == ">'" or operator == '>-' or operator == '>-1':
            retval = lambda n, m=None, l=None: hasattr(n, 'parent') and bool(n.parent()) and (n is n.parent()[-1]) and predicate(n.parent(), m, l)
        elif operator[:2] == '<-' and operator[2:].isdigit():
            idx = -int(operator[2:])
            retval = (lambda i: lambda n, m=None, l=None: _istree(n) and bool(list(n)) and (0 <= i + len(n) < len(n)) and predicate(n[i + len(n)], m, l))(idx)
        elif operator[:2] == '>-' and operator[2:].isdigit():
            idx = -int(operator[2:])
            retval = (lambda i: lambda n, m=None, l=None: hasattr(n, 'parent') and bool(n.parent()) and (0 <= i + len(n.parent()) < len(n.parent())) and (n is n.parent()[i + len(n.parent())]) and predicate(n.parent(), m, l))(idx)
        elif operator == '<:':
            retval = lambda n, m=None, l=None: _istree(n) and len(n) == 1 and predicate(n[0], m, l)
        elif operator == '>:':
            retval = lambda n, m=None, l=None: hasattr(n, 'parent') and bool(n.parent()) and (len(n.parent()) == 1) and predicate(n.parent(), m, l)
        elif operator == '<<':
            retval = lambda n, m=None, l=None: _istree(n) and any((predicate(x, m, l) for x in _descendants(n)))
        elif operator == '>>':
            retval = lambda n, m=None, l=None: any((predicate(x, m, l) for x in ancestors(n)))
        elif operator == '<<,' or operator == '<<1':
            retval = lambda n, m=None, l=None: _istree(n) and any((predicate(x, m, l) for x in _leftmost_descendants(n)))
        elif operator == '>>,':
            retval = lambda n, m=None, l=None: any((predicate(x, m, l) and n in _leftmost_descendants(x) for x in ancestors(n)))
        elif operator == "<<'":
            retval = lambda n, m=None, l=None: _istree(n) and any((predicate(x, m, l) for x in _rightmost_descendants(n)))
        elif operator == ">>'":
            retval = lambda n, m=None, l=None: any((predicate(x, m, l) and n in _rightmost_descendants(x) for x in ancestors(n)))
        elif operator == '<<:':
            retval = lambda n, m=None, l=None: _istree(n) and any((predicate(x, m, l) for x in _unique_descendants(n)))
        elif operator == '>>:':
            retval = lambda n, m=None, l=None: any((predicate(x, m, l) for x in unique_ancestors(n)))
        elif operator == '.':
            retval = lambda n, m=None, l=None: any((predicate(x, m, l) for x in _immediately_after(n)))
        elif operator == ',':
            retval = lambda n, m=None, l=None: any((predicate(x, m, l) for x in _immediately_before(n)))
        elif operator == '..':
            retval = lambda n, m=None, l=None: any((predicate(x, m, l) for x in _after(n)))
        elif operator == ',,':
            retval = lambda n, m=None, l=None: any((predicate(x, m, l) for x in _before(n)))
        elif operator == '$' or operator == '%':
            retval = lambda n, m=None, l=None: hasattr(n, 'parent') and bool(n.parent()) and any((predicate(x, m, l) for x in n.parent() if x is not n))
        elif operator == '$.' or operator == '%.':
            retval = lambda n, m=None, l=None: hasattr(n, 'right_sibling') and bool(n.right_sibling()) and predicate(n.right_sibling(), m, l)
        elif operator == '$,' or operator == '%,':
            retval = lambda n, m=None, l=None: hasattr(n, 'left_sibling') and bool(n.left_sibling()) and predicate(n.left_sibling(), m, l)
        elif operator == '$..' or operator == '%..':
            retval = lambda n, m=None, l=None: hasattr(n, 'parent') and hasattr(n, 'parent_index') and bool(n.parent()) and any((predicate(x, m, l) for x in n.parent()[n.parent_index() + 1:]))
        elif operator == '$,,' or operator == '%,,':
            retval = lambda n, m=None, l=None: hasattr(n, 'parent') and hasattr(n, 'parent_index') and bool(n.parent()) and any((predicate(x, m, l) for x in n.parent()[:n.parent_index()]))
        else:
            raise TgrepException(f'cannot interpret tgrep operator "{operator}"')
    if negated:
        return (lambda r: lambda n, m=None, l=None: not r(n, m, l))(retval)
    else:
        return retval

def _tgrep_conjunction_action(_s, _l, tokens, join_char='&'):
    if False:
        while True:
            i = 10
    '\n    Builds a lambda function representing a predicate on a tree node\n    from the conjunction of several other such lambda functions.\n\n    This is prototypically called for expressions like\n    (`tgrep_rel_conjunction`)::\n\n        < NP & < AP < VP\n\n    where tokens is a list of predicates representing the relations\n    (`< NP`, `< AP`, and `< VP`), possibly with the character `&`\n    included (as in the example here).\n\n    This is also called for expressions like (`tgrep_node_expr2`)::\n\n        NP < NN\n        S=s < /NP/=n : s < /VP/=v : n .. v\n\n    tokens[0] is a tgrep_expr predicate; tokens[1:] are an (optional)\n    list of segmented patterns (`tgrep_expr_labeled`, processed by\n    `_tgrep_segmented_pattern_action`).\n    '
    tokens = [x for x in tokens if x != join_char]
    if len(tokens) == 1:
        return tokens[0]
    else:
        return (lambda ts: lambda n, m=None, l=None: all((predicate(n, m, l) for predicate in ts)))(tokens)

def _tgrep_segmented_pattern_action(_s, _l, tokens):
    if False:
        for i in range(10):
            print('nop')
    "\n    Builds a lambda function representing a segmented pattern.\n\n    Called for expressions like (`tgrep_expr_labeled`)::\n\n        =s .. =v < =n\n\n    This is a segmented pattern, a tgrep2 expression which begins with\n    a node label.\n\n    The problem is that for segemented_pattern_action (': =v < =s'),\n    the first element (in this case, =v) is specifically selected by\n    virtue of matching a particular node in the tree; to retrieve\n    the node, we need the label, not a lambda function.  For node\n    labels inside a tgrep_node_expr, we need a lambda function which\n    returns true if the node visited is the same as =v.\n\n    We solve this by creating two copies of a node_label_use in the\n    grammar; the label use inside a tgrep_expr_labeled has a separate\n    parse action to the pred use inside a node_expr.  See\n    `_tgrep_node_label_use_action` and\n    `_tgrep_node_label_pred_use_action`.\n    "
    node_label = tokens[0]
    reln_preds = tokens[1:]

    def pattern_segment_pred(n, m=None, l=None):
        if False:
            while True:
                i = 10
        'This predicate function ignores its node argument.'
        if l is None or node_label not in l:
            raise TgrepException(f'node_label ={node_label} not bound in pattern')
        node = l[node_label]
        return all((pred(node, m, l) for pred in reln_preds))
    return pattern_segment_pred

def _tgrep_node_label_use_action(_s, _l, tokens):
    if False:
        i = 10
        return i + 15
    '\n    Returns the node label used to begin a tgrep_expr_labeled.  See\n    `_tgrep_segmented_pattern_action`.\n\n    Called for expressions like (`tgrep_node_label_use`)::\n\n        =s\n\n    when they appear as the first element of a `tgrep_expr_labeled`\n    expression (see `_tgrep_segmented_pattern_action`).\n\n    It returns the node label.\n    '
    assert len(tokens) == 1
    assert tokens[0].startswith('=')
    return tokens[0][1:]

def _tgrep_node_label_pred_use_action(_s, _l, tokens):
    if False:
        i = 10
        return i + 15
    "\n    Builds a lambda function representing a predicate on a tree node\n    which describes the use of a previously bound node label.\n\n    Called for expressions like (`tgrep_node_label_use_pred`)::\n\n        =s\n\n    when they appear inside a tgrep_node_expr (for example, inside a\n    relation).  The predicate returns true if and only if its node\n    argument is identical the the node looked up in the node label\n    dictionary using the node's label.\n    "
    assert len(tokens) == 1
    assert tokens[0].startswith('=')
    node_label = tokens[0][1:]

    def node_label_use_pred(n, m=None, l=None):
        if False:
            i = 10
            return i + 15
        if l is None or node_label not in l:
            raise TgrepException(f'node_label ={node_label} not bound in pattern')
        node = l[node_label]
        return n is node
    return node_label_use_pred

def _tgrep_bind_node_label_action(_s, _l, tokens):
    if False:
        print('Hello World!')
    "\n    Builds a lambda function representing a predicate on a tree node\n    which can optionally bind a matching node into the tgrep2 string's\n    label_dict.\n\n    Called for expressions like (`tgrep_node_expr2`)::\n\n        /NP/\n        @NP=n\n    "
    if len(tokens) == 1:
        return tokens[0]
    else:
        assert len(tokens) == 3
        assert tokens[1] == '='
        node_pred = tokens[0]
        node_label = tokens[2]

        def node_label_bind_pred(n, m=None, l=None):
            if False:
                return 10
            if node_pred(n, m, l):
                if l is None:
                    raise TgrepException('cannot bind node_label {}: label_dict is None'.format(node_label))
                l[node_label] = n
                return True
            else:
                return False
        return node_label_bind_pred

def _tgrep_rel_disjunction_action(_s, _l, tokens):
    if False:
        i = 10
        return i + 15
    '\n    Builds a lambda function representing a predicate on a tree node\n    from the disjunction of several other such lambda functions.\n    '
    tokens = [x for x in tokens if x != '|']
    if len(tokens) == 1:
        return tokens[0]
    elif len(tokens) == 2:
        return (lambda a, b: lambda n, m=None, l=None: a(n, m, l) or b(n, m, l))(tokens[0], tokens[1])

def _macro_defn_action(_s, _l, tokens):
    if False:
        print('Hello World!')
    '\n    Builds a dictionary structure which defines the given macro.\n    '
    assert len(tokens) == 3
    assert tokens[0] == '@'
    return {tokens[1]: tokens[2]}

def _tgrep_exprs_action(_s, _l, tokens):
    if False:
        return 10
    '\n    This is the top-lebel node in a tgrep2 search string; the\n    predicate function it returns binds together all the state of a\n    tgrep2 search string.\n\n    Builds a lambda function representing a predicate on a tree node\n    from the disjunction of several tgrep expressions.  Also handles\n    macro definitions and macro name binding, and node label\n    definitions and node label binding.\n    '
    if len(tokens) == 1:
        return lambda n, m=None, l=None: tokens[0](n, None, {})
    tokens = [x for x in tokens if x != ';']
    macro_dict = {}
    macro_defs = [tok for tok in tokens if isinstance(tok, dict)]
    for macro_def in macro_defs:
        macro_dict.update(macro_def)
    tgrep_exprs = [tok for tok in tokens if not isinstance(tok, dict)]

    def top_level_pred(n, m=macro_dict, l=None):
        if False:
            return 10
        label_dict = {}
        return any((predicate(n, m, label_dict) for predicate in tgrep_exprs))
    return top_level_pred

def _build_tgrep_parser(set_parse_actions=True):
    if False:
        while True:
            i = 10
    '\n    Builds a pyparsing-based parser object for tokenizing and\n    interpreting tgrep search strings.\n    '
    tgrep_op = pyparsing.Optional('!') + pyparsing.Regex("[$%,.<>][%,.<>0-9-':]*")
    tgrep_qstring = pyparsing.QuotedString(quoteChar='"', escChar='\\', unquoteResults=False)
    tgrep_node_regex = pyparsing.QuotedString(quoteChar='/', escChar='\\', unquoteResults=False)
    tgrep_qstring_icase = pyparsing.Regex('i@\\"(?:[^"\\n\\r\\\\]|(?:\\\\.))*\\"')
    tgrep_node_regex_icase = pyparsing.Regex('i@\\/(?:[^/\\n\\r\\\\]|(?:\\\\.))*\\/')
    tgrep_node_literal = pyparsing.Regex("[^][ \r\t\n;:.,&|<>()$!@%'^=]+")
    tgrep_expr = pyparsing.Forward()
    tgrep_relations = pyparsing.Forward()
    tgrep_parens = pyparsing.Literal('(') + tgrep_expr + ')'
    tgrep_nltk_tree_pos = pyparsing.Literal('N(') + pyparsing.Optional(pyparsing.Word(pyparsing.nums) + ',' + pyparsing.Optional(pyparsing.delimitedList(pyparsing.Word(pyparsing.nums), delim=',') + pyparsing.Optional(','))) + ')'
    tgrep_node_label = pyparsing.Regex('[A-Za-z0-9]+')
    tgrep_node_label_use = pyparsing.Combine('=' + tgrep_node_label)
    tgrep_node_label_use_pred = tgrep_node_label_use.copy()
    macro_name = pyparsing.Regex("[^];:.,&|<>()[$!@%'^=\r\t\n ]+")
    macro_name.setWhitespaceChars('')
    macro_use = pyparsing.Combine('@' + macro_name)
    tgrep_node_expr = tgrep_node_label_use_pred | macro_use | tgrep_nltk_tree_pos | tgrep_qstring_icase | tgrep_node_regex_icase | tgrep_qstring | tgrep_node_regex | '*' | tgrep_node_literal
    tgrep_node_expr2 = tgrep_node_expr + pyparsing.Literal('=').setWhitespaceChars('') + tgrep_node_label.copy().setWhitespaceChars('') | tgrep_node_expr
    tgrep_node = tgrep_parens | pyparsing.Optional("'") + tgrep_node_expr2 + pyparsing.ZeroOrMore('|' + tgrep_node_expr)
    tgrep_brackets = pyparsing.Optional('!') + '[' + tgrep_relations + ']'
    tgrep_relation = tgrep_brackets | tgrep_op + tgrep_node
    tgrep_rel_conjunction = pyparsing.Forward()
    tgrep_rel_conjunction << tgrep_relation + pyparsing.ZeroOrMore(pyparsing.Optional('&') + tgrep_rel_conjunction)
    tgrep_relations << tgrep_rel_conjunction + pyparsing.ZeroOrMore('|' + tgrep_relations)
    tgrep_expr << tgrep_node + pyparsing.Optional(tgrep_relations)
    tgrep_expr_labeled = tgrep_node_label_use + pyparsing.Optional(tgrep_relations)
    tgrep_expr2 = tgrep_expr + pyparsing.ZeroOrMore(':' + tgrep_expr_labeled)
    macro_defn = pyparsing.Literal('@') + pyparsing.White().suppress() + macro_name + tgrep_expr2
    tgrep_exprs = pyparsing.Optional(macro_defn + pyparsing.ZeroOrMore(';' + macro_defn) + ';') + tgrep_expr2 + pyparsing.ZeroOrMore(';' + (macro_defn | tgrep_expr2)) + pyparsing.ZeroOrMore(';').suppress()
    if set_parse_actions:
        tgrep_node_label_use.setParseAction(_tgrep_node_label_use_action)
        tgrep_node_label_use_pred.setParseAction(_tgrep_node_label_pred_use_action)
        macro_use.setParseAction(_tgrep_macro_use_action)
        tgrep_node.setParseAction(_tgrep_node_action)
        tgrep_node_expr2.setParseAction(_tgrep_bind_node_label_action)
        tgrep_parens.setParseAction(_tgrep_parens_action)
        tgrep_nltk_tree_pos.setParseAction(_tgrep_nltk_tree_pos_action)
        tgrep_relation.setParseAction(_tgrep_relation_action)
        tgrep_rel_conjunction.setParseAction(_tgrep_conjunction_action)
        tgrep_relations.setParseAction(_tgrep_rel_disjunction_action)
        macro_defn.setParseAction(_macro_defn_action)
        tgrep_expr.setParseAction(_tgrep_conjunction_action)
        tgrep_expr_labeled.setParseAction(_tgrep_segmented_pattern_action)
        tgrep_expr2.setParseAction(functools.partial(_tgrep_conjunction_action, join_char=':'))
        tgrep_exprs.setParseAction(_tgrep_exprs_action)
    return tgrep_exprs.ignore('#' + pyparsing.restOfLine)

def tgrep_tokenize(tgrep_string):
    if False:
        return 10
    '\n    Tokenizes a TGrep search string into separate tokens.\n    '
    parser = _build_tgrep_parser(False)
    if isinstance(tgrep_string, bytes):
        tgrep_string = tgrep_string.decode()
    return list(parser.parseString(tgrep_string))

def tgrep_compile(tgrep_string):
    if False:
        return 10
    '\n    Parses (and tokenizes, if necessary) a TGrep search string into a\n    lambda function.\n    '
    parser = _build_tgrep_parser(True)
    if isinstance(tgrep_string, bytes):
        tgrep_string = tgrep_string.decode()
    return list(parser.parseString(tgrep_string, parseAll=True))[0]

def treepositions_no_leaves(tree):
    if False:
        return 10
    '\n    Returns all the tree positions in the given tree which are not\n    leaf nodes.\n    '
    treepositions = tree.treepositions()
    prefixes = set()
    for pos in treepositions:
        for length in range(len(pos)):
            prefixes.add(pos[:length])
    return [pos for pos in treepositions if pos in prefixes]

def tgrep_positions(pattern, trees, search_leaves=True):
    if False:
        while True:
            i = 10
    '\n    Return the tree positions in the trees which match the given pattern.\n\n    :param pattern: a tgrep search pattern\n    :type pattern: str or output of tgrep_compile()\n    :param trees: a sequence of NLTK trees (usually ParentedTrees)\n    :type trees: iter(ParentedTree) or iter(Tree)\n    :param search_leaves: whether to return matching leaf nodes\n    :type search_leaves: bool\n    :rtype: iter(tree positions)\n    '
    if isinstance(pattern, (bytes, str)):
        pattern = tgrep_compile(pattern)
    for tree in trees:
        try:
            if search_leaves:
                positions = tree.treepositions()
            else:
                positions = treepositions_no_leaves(tree)
            yield [position for position in positions if pattern(tree[position])]
        except AttributeError:
            yield []

def tgrep_nodes(pattern, trees, search_leaves=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return the tree nodes in the trees which match the given pattern.\n\n    :param pattern: a tgrep search pattern\n    :type pattern: str or output of tgrep_compile()\n    :param trees: a sequence of NLTK trees (usually ParentedTrees)\n    :type trees: iter(ParentedTree) or iter(Tree)\n    :param search_leaves: whether to return matching leaf nodes\n    :type search_leaves: bool\n    :rtype: iter(tree nodes)\n    '
    if isinstance(pattern, (bytes, str)):
        pattern = tgrep_compile(pattern)
    for tree in trees:
        try:
            if search_leaves:
                positions = tree.treepositions()
            else:
                positions = treepositions_no_leaves(tree)
            yield [tree[position] for position in positions if pattern(tree[position])]
        except AttributeError:
            yield []