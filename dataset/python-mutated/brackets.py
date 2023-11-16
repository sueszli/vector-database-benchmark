"""Builds on top of nodes.py to track brackets."""
from dataclasses import dataclass, field
from typing import Dict, Final, Iterable, List, Optional, Sequence, Set, Tuple, Union
from black.nodes import BRACKET, CLOSING_BRACKETS, COMPARATORS, LOGIC_OPERATORS, MATH_OPERATORS, OPENING_BRACKETS, UNPACKING_PARENTS, VARARGS_PARENTS, is_vararg, syms
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
LN = Union[Leaf, Node]
Depth = int
LeafID = int
NodeType = int
Priority = int
COMPREHENSION_PRIORITY: Final = 20
COMMA_PRIORITY: Final = 18
TERNARY_PRIORITY: Final = 16
LOGIC_PRIORITY: Final = 14
STRING_PRIORITY: Final = 12
COMPARATOR_PRIORITY: Final = 10
MATH_PRIORITIES: Final = {token.VBAR: 9, token.CIRCUMFLEX: 8, token.AMPER: 7, token.LEFTSHIFT: 6, token.RIGHTSHIFT: 6, token.PLUS: 5, token.MINUS: 5, token.STAR: 4, token.SLASH: 4, token.DOUBLESLASH: 4, token.PERCENT: 4, token.AT: 4, token.TILDE: 3, token.DOUBLESTAR: 2}
DOT_PRIORITY: Final = 1

class BracketMatchError(Exception):
    """Raised when an opening bracket is unable to be matched to a closing bracket."""

@dataclass
class BracketTracker:
    """Keeps track of brackets on a line."""
    depth: int = 0
    bracket_match: Dict[Tuple[Depth, NodeType], Leaf] = field(default_factory=dict)
    delimiters: Dict[LeafID, Priority] = field(default_factory=dict)
    previous: Optional[Leaf] = None
    _for_loop_depths: List[int] = field(default_factory=list)
    _lambda_argument_depths: List[int] = field(default_factory=list)
    invisible: List[Leaf] = field(default_factory=list)

    def mark(self, leaf: Leaf) -> None:
        if False:
            return 10
        'Mark `leaf` with bracket-related metadata. Keep track of delimiters.\n\n        All leaves receive an int `bracket_depth` field that stores how deep\n        within brackets a given leaf is. 0 means there are no enclosing brackets\n        that started on this line.\n\n        If a leaf is itself a closing bracket and there is a matching opening\n        bracket earlier, it receives an `opening_bracket` field with which it forms a\n        pair. This is a one-directional link to avoid reference cycles. Closing\n        bracket without opening happens on lines continued from previous\n        breaks, e.g. `) -> "ReturnType":` as part of a funcdef where we place\n        the return type annotation on its own line of the previous closing RPAR.\n\n        If a leaf is a delimiter (a token on which Black can split the line if\n        needed) and it\'s on depth 0, its `id()` is stored in the tracker\'s\n        `delimiters` field.\n        '
        if leaf.type == token.COMMENT:
            return
        if self.depth == 0 and leaf.type in CLOSING_BRACKETS and ((self.depth, leaf.type) not in self.bracket_match):
            return
        self.maybe_decrement_after_for_loop_variable(leaf)
        self.maybe_decrement_after_lambda_arguments(leaf)
        if leaf.type in CLOSING_BRACKETS:
            self.depth -= 1
            try:
                opening_bracket = self.bracket_match.pop((self.depth, leaf.type))
            except KeyError as e:
                raise BracketMatchError(f'Unable to match a closing bracket to the following opening bracket: {leaf}') from e
            leaf.opening_bracket = opening_bracket
            if not leaf.value:
                self.invisible.append(leaf)
        leaf.bracket_depth = self.depth
        if self.depth == 0:
            delim = is_split_before_delimiter(leaf, self.previous)
            if delim and self.previous is not None:
                self.delimiters[id(self.previous)] = delim
            else:
                delim = is_split_after_delimiter(leaf, self.previous)
                if delim:
                    self.delimiters[id(leaf)] = delim
        if leaf.type in OPENING_BRACKETS:
            self.bracket_match[self.depth, BRACKET[leaf.type]] = leaf
            self.depth += 1
            if not leaf.value:
                self.invisible.append(leaf)
        self.previous = leaf
        self.maybe_increment_lambda_arguments(leaf)
        self.maybe_increment_for_loop_variable(leaf)

    def any_open_for_or_lambda(self) -> bool:
        if False:
            print('Hello World!')
        'Return True if there is an open for or lambda expression on the line.\n\n        See maybe_increment_for_loop_variable and maybe_increment_lambda_arguments\n        for details.'
        return bool(self._for_loop_depths or self._lambda_argument_depths)

    def any_open_brackets(self) -> bool:
        if False:
            print('Hello World!')
        'Return True if there is an yet unmatched open bracket on the line.'
        return bool(self.bracket_match)

    def max_delimiter_priority(self, exclude: Iterable[LeafID]=()) -> Priority:
        if False:
            i = 10
            return i + 15
        'Return the highest priority of a delimiter found on the line.\n\n        Values are consistent with what `is_split_*_delimiter()` return.\n        Raises ValueError on no delimiters.\n        '
        return max((v for (k, v) in self.delimiters.items() if k not in exclude))

    def delimiter_count_with_priority(self, priority: Priority=0) -> int:
        if False:
            i = 10
            return i + 15
        'Return the number of delimiters with the given `priority`.\n\n        If no `priority` is passed, defaults to max priority on the line.\n        '
        if not self.delimiters:
            return 0
        priority = priority or self.max_delimiter_priority()
        return sum((1 for p in self.delimiters.values() if p == priority))

    def maybe_increment_for_loop_variable(self, leaf: Leaf) -> bool:
        if False:
            i = 10
            return i + 15
        'In a for loop, or comprehension, the variables are often unpacks.\n\n        To avoid splitting on the comma in this situation, increase the depth of\n        tokens between `for` and `in`.\n        '
        if leaf.type == token.NAME and leaf.value == 'for':
            self.depth += 1
            self._for_loop_depths.append(self.depth)
            return True
        return False

    def maybe_decrement_after_for_loop_variable(self, leaf: Leaf) -> bool:
        if False:
            print('Hello World!')
        'See `maybe_increment_for_loop_variable` above for explanation.'
        if self._for_loop_depths and self._for_loop_depths[-1] == self.depth and (leaf.type == token.NAME) and (leaf.value == 'in'):
            self.depth -= 1
            self._for_loop_depths.pop()
            return True
        return False

    def maybe_increment_lambda_arguments(self, leaf: Leaf) -> bool:
        if False:
            print('Hello World!')
        'In a lambda expression, there might be more than one argument.\n\n        To avoid splitting on the comma in this situation, increase the depth of\n        tokens between `lambda` and `:`.\n        '
        if leaf.type == token.NAME and leaf.value == 'lambda':
            self.depth += 1
            self._lambda_argument_depths.append(self.depth)
            return True
        return False

    def maybe_decrement_after_lambda_arguments(self, leaf: Leaf) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'See `maybe_increment_lambda_arguments` above for explanation.'
        if self._lambda_argument_depths and self._lambda_argument_depths[-1] == self.depth and (leaf.type == token.COLON):
            self.depth -= 1
            self._lambda_argument_depths.pop()
            return True
        return False

    def get_open_lsqb(self) -> Optional[Leaf]:
        if False:
            print('Hello World!')
        'Return the most recent opening square bracket (if any).'
        return self.bracket_match.get((self.depth - 1, token.RSQB))

def is_split_after_delimiter(leaf: Leaf, previous: Optional[Leaf]=None) -> Priority:
    if False:
        for i in range(10):
            print('nop')
    'Return the priority of the `leaf` delimiter, given a line break after it.\n\n    The delimiter priorities returned here are from those delimiters that would\n    cause a line break after themselves.\n\n    Higher numbers are higher priority.\n    '
    if leaf.type == token.COMMA:
        return COMMA_PRIORITY
    return 0

def is_split_before_delimiter(leaf: Leaf, previous: Optional[Leaf]=None) -> Priority:
    if False:
        i = 10
        return i + 15
    'Return the priority of the `leaf` delimiter, given a line break before it.\n\n    The delimiter priorities returned here are from those delimiters that would\n    cause a line break before themselves.\n\n    Higher numbers are higher priority.\n    '
    if is_vararg(leaf, within=VARARGS_PARENTS | UNPACKING_PARENTS):
        return 0
    if leaf.type == token.DOT and leaf.parent and (leaf.parent.type not in {syms.import_from, syms.dotted_name}) and (previous is None or previous.type in CLOSING_BRACKETS):
        return DOT_PRIORITY
    if leaf.type in MATH_OPERATORS and leaf.parent and (leaf.parent.type not in {syms.factor, syms.star_expr}):
        return MATH_PRIORITIES[leaf.type]
    if leaf.type in COMPARATORS:
        return COMPARATOR_PRIORITY
    if leaf.type == token.STRING and previous is not None and (previous.type == token.STRING):
        return STRING_PRIORITY
    if leaf.type not in {token.NAME, token.ASYNC}:
        return 0
    if leaf.value == 'for' and leaf.parent and (leaf.parent.type in {syms.comp_for, syms.old_comp_for}) or leaf.type == token.ASYNC:
        if not isinstance(leaf.prev_sibling, Leaf) or leaf.prev_sibling.value != 'async':
            return COMPREHENSION_PRIORITY
    if leaf.value == 'if' and leaf.parent and (leaf.parent.type in {syms.comp_if, syms.old_comp_if}):
        return COMPREHENSION_PRIORITY
    if leaf.value in {'if', 'else'} and leaf.parent and (leaf.parent.type == syms.test):
        return TERNARY_PRIORITY
    if leaf.value == 'is':
        return COMPARATOR_PRIORITY
    if leaf.value == 'in' and leaf.parent and (leaf.parent.type in {syms.comp_op, syms.comparison}) and (not (previous is not None and previous.type == token.NAME and (previous.value == 'not'))):
        return COMPARATOR_PRIORITY
    if leaf.value == 'not' and leaf.parent and (leaf.parent.type == syms.comp_op) and (not (previous is not None and previous.type == token.NAME and (previous.value == 'is'))):
        return COMPARATOR_PRIORITY
    if leaf.value in LOGIC_OPERATORS and leaf.parent:
        return LOGIC_PRIORITY
    return 0

def max_delimiter_priority_in_atom(node: LN) -> Priority:
    if False:
        print('Hello World!')
    "Return maximum delimiter priority inside `node`.\n\n    This is specific to atoms with contents contained in a pair of parentheses.\n    If `node` isn't an atom or there are no enclosing parentheses, returns 0.\n    "
    if node.type != syms.atom:
        return 0
    first = node.children[0]
    last = node.children[-1]
    if not (first.type == token.LPAR and last.type == token.RPAR):
        return 0
    bt = BracketTracker()
    for c in node.children[1:-1]:
        if isinstance(c, Leaf):
            bt.mark(c)
        else:
            for leaf in c.leaves():
                bt.mark(leaf)
    try:
        return bt.max_delimiter_priority()
    except ValueError:
        return 0

def get_leaves_inside_matching_brackets(leaves: Sequence[Leaf]) -> Set[LeafID]:
    if False:
        while True:
            i = 10
    'Return leaves that are inside matching brackets.\n\n    The input `leaves` can have non-matching brackets at the head or tail parts.\n    Matching brackets are included.\n    '
    try:
        start_index = next((i for (i, l) in enumerate(leaves) if l.type in OPENING_BRACKETS))
    except StopIteration:
        return set()
    bracket_stack = []
    ids = set()
    for i in range(start_index, len(leaves)):
        leaf = leaves[i]
        if leaf.type in OPENING_BRACKETS:
            bracket_stack.append((BRACKET[leaf.type], i))
        if leaf.type in CLOSING_BRACKETS:
            if bracket_stack and leaf.type == bracket_stack[-1][0]:
                (_, start) = bracket_stack.pop()
                for j in range(start, i + 1):
                    ids.add(id(leaves[j]))
            else:
                break
    return ids