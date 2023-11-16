import itertools
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple, TypeVar, Union, cast
from black.brackets import COMMA_PRIORITY, DOT_PRIORITY, BracketTracker
from black.mode import Mode, Preview
from black.nodes import BRACKETS, CLOSING_BRACKETS, OPENING_BRACKETS, STANDALONE_COMMENT, TEST_DESCENDANTS, child_towards, is_docstring, is_funcdef, is_import, is_multiline_string, is_one_sequence_between, is_type_comment, is_type_ignore_comment, is_with_or_async_with_stmt, replace_child, syms, whitespace
from black.strings import str_width
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
T = TypeVar('T')
Index = int
LeafID = int
LN = Union[Leaf, Node]

@dataclass
class Line:
    """Holds leaves and comments. Can be printed with `str(line)`."""
    mode: Mode = field(repr=False)
    depth: int = 0
    leaves: List[Leaf] = field(default_factory=list)
    comments: Dict[LeafID, List[Leaf]] = field(default_factory=dict)
    bracket_tracker: BracketTracker = field(default_factory=BracketTracker)
    inside_brackets: bool = False
    should_split_rhs: bool = False
    magic_trailing_comma: Optional[Leaf] = None

    def append(self, leaf: Leaf, preformatted: bool=False, track_bracket: bool=False) -> None:
        if False:
            return 10
        'Add a new `leaf` to the end of the line.\n\n        Unless `preformatted` is True, the `leaf` will receive a new consistent\n        whitespace prefix and metadata applied by :class:`BracketTracker`.\n        Trailing commas are maybe removed, unpacked for loop variables are\n        demoted from being delimiters.\n\n        Inline comments are put aside.\n        '
        has_value = leaf.type in BRACKETS or bool(leaf.value.strip())
        if not has_value:
            return
        if token.COLON == leaf.type and self.is_class_paren_empty:
            del self.leaves[-2:]
        if self.leaves and (not preformatted):
            leaf.prefix += whitespace(leaf, complex_subscript=self.is_complex_subscript(leaf), mode=self.mode)
        if self.inside_brackets or not preformatted or track_bracket:
            self.bracket_tracker.mark(leaf)
            if self.mode.magic_trailing_comma:
                if self.has_magic_trailing_comma(leaf):
                    self.magic_trailing_comma = leaf
            elif self.has_magic_trailing_comma(leaf, ensure_removable=True):
                self.remove_trailing_comma()
        if not self.append_comment(leaf):
            self.leaves.append(leaf)

    def append_safe(self, leaf: Leaf, preformatted: bool=False) -> None:
        if False:
            return 10
        'Like :func:`append()` but disallow invalid standalone comment structure.\n\n        Raises ValueError when any `leaf` is appended after a standalone comment\n        or when a standalone comment is not the first leaf on the line.\n        '
        if self.bracket_tracker.depth == 0 or self.bracket_tracker.any_open_for_or_lambda():
            if self.is_comment:
                raise ValueError('cannot append to standalone comments')
            if self.leaves and leaf.type == STANDALONE_COMMENT:
                raise ValueError('cannot append standalone comments to a populated line')
        self.append(leaf, preformatted=preformatted)

    @property
    def is_comment(self) -> bool:
        if False:
            print('Hello World!')
        'Is this line a standalone comment?'
        return len(self.leaves) == 1 and self.leaves[0].type == STANDALONE_COMMENT

    @property
    def is_decorator(self) -> bool:
        if False:
            print('Hello World!')
        'Is this line a decorator?'
        return bool(self) and self.leaves[0].type == token.AT

    @property
    def is_import(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Is this an import line?'
        return bool(self) and is_import(self.leaves[0])

    @property
    def is_with_or_async_with_stmt(self) -> bool:
        if False:
            while True:
                i = 10
        'Is this a with_stmt line?'
        return bool(self) and is_with_or_async_with_stmt(self.leaves[0])

    @property
    def is_class(self) -> bool:
        if False:
            return 10
        'Is this line a class definition?'
        return bool(self) and self.leaves[0].type == token.NAME and (self.leaves[0].value == 'class')

    @property
    def is_stub_class(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Is this line a class definition with a body consisting only of "..."?'
        return self.is_class and self.leaves[-3:] == [Leaf(token.DOT, '.') for _ in range(3)]

    @property
    def is_def(self) -> bool:
        if False:
            while True:
                i = 10
        'Is this a function definition? (Also returns True for async defs.)'
        try:
            first_leaf = self.leaves[0]
        except IndexError:
            return False
        try:
            second_leaf: Optional[Leaf] = self.leaves[1]
        except IndexError:
            second_leaf = None
        return first_leaf.type == token.NAME and first_leaf.value == 'def' or (first_leaf.type == token.ASYNC and second_leaf is not None and (second_leaf.type == token.NAME) and (second_leaf.value == 'def'))

    @property
    def is_stub_def(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Is this line a function definition with a body consisting only of "..."?'
        return self.is_def and self.leaves[-4:] == [Leaf(token.COLON, ':')] + [Leaf(token.DOT, '.') for _ in range(3)]

    @property
    def is_class_paren_empty(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Is this a class with no base classes but using parentheses?\n\n        Those are unnecessary and should be removed.\n        '
        return bool(self) and len(self.leaves) == 4 and self.is_class and (self.leaves[2].type == token.LPAR) and (self.leaves[2].value == '(') and (self.leaves[3].type == token.RPAR) and (self.leaves[3].value == ')')

    @property
    def is_triple_quoted_string(self) -> bool:
        if False:
            print('Hello World!')
        'Is the line a triple quoted string?'
        if not self or self.leaves[0].type != token.STRING:
            return False
        value = self.leaves[0].value
        if value.startswith(('"""', "'''")):
            return True
        if Preview.accept_raw_docstrings in self.mode and value.startswith(("r'''", 'r"""', "R'''", 'R"""')):
            return True
        return False

    @property
    def opens_block(self) -> bool:
        if False:
            print('Hello World!')
        'Does this line open a new level of indentation.'
        if len(self.leaves) == 0:
            return False
        return self.leaves[-1].type == token.COLON

    def is_fmt_pass_converted(self, *, first_leaf_matches: Optional[Callable[[Leaf], bool]]=None) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Is this line converted from fmt off/skip code?\n\n        If first_leaf_matches is not None, it only returns True if the first\n        leaf of converted code matches.\n        '
        if len(self.leaves) != 1:
            return False
        leaf = self.leaves[0]
        if leaf.type != STANDALONE_COMMENT or leaf.fmt_pass_converted_first_leaf is None:
            return False
        return first_leaf_matches is None or first_leaf_matches(leaf.fmt_pass_converted_first_leaf)

    def contains_standalone_comments(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'If so, needs to be split before emitting.'
        for leaf in self.leaves:
            if leaf.type == STANDALONE_COMMENT:
                return True
        return False

    def contains_implicit_multiline_string_with_comments(self) -> bool:
        if False:
            return 10
        'Chck if we have an implicit multiline string with comments on the line'
        for (leaf_type, leaf_group_iterator) in itertools.groupby(self.leaves, lambda leaf: leaf.type):
            if leaf_type != token.STRING:
                continue
            leaf_list = list(leaf_group_iterator)
            if len(leaf_list) == 1:
                continue
            for leaf in leaf_list:
                if self.comments_after(leaf):
                    return True
        return False

    def contains_uncollapsable_type_comments(self) -> bool:
        if False:
            return 10
        ignored_ids = set()
        try:
            last_leaf = self.leaves[-1]
            ignored_ids.add(id(last_leaf))
            if last_leaf.type == token.COMMA or (last_leaf.type == token.RPAR and (not last_leaf.value)):
                last_leaf = self.leaves[-2]
                ignored_ids.add(id(last_leaf))
        except IndexError:
            return False
        comment_seen = False
        for (leaf_id, comments) in self.comments.items():
            for comment in comments:
                if is_type_comment(comment):
                    if comment_seen or (not is_type_ignore_comment(comment) and leaf_id not in ignored_ids):
                        return True
                comment_seen = True
        return False

    def contains_unsplittable_type_ignore(self) -> bool:
        if False:
            i = 10
            return i + 15
        if not self.leaves:
            return False
        first_line = next((leaf.lineno for leaf in self.leaves if leaf.lineno != 0), 0)
        last_line = next((leaf.lineno for leaf in reversed(self.leaves) if leaf.lineno != 0), 0)
        if first_line == last_line:
            for node in self.leaves[-2:]:
                for comment in self.comments.get(id(node), []):
                    if is_type_ignore_comment(comment):
                        return True
        return False

    def contains_multiline_strings(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return any((is_multiline_string(leaf) for leaf in self.leaves))

    def has_magic_trailing_comma(self, closing: Leaf, ensure_removable: bool=False) -> bool:
        if False:
            return 10
        "Return True if we have a magic trailing comma, that is when:\n        - there's a trailing comma here\n        - it's not a one-tuple\n        - it's not a single-element subscript\n        Additionally, if ensure_removable:\n        - it's not from square bracket indexing\n        (specifically, single-element square bracket indexing)\n        "
        if not (closing.type in CLOSING_BRACKETS and self.leaves and (self.leaves[-1].type == token.COMMA)):
            return False
        if closing.type == token.RBRACE:
            return True
        if closing.type == token.RSQB:
            if closing.parent is not None and closing.parent.type == syms.trailer and (closing.opening_bracket is not None) and is_one_sequence_between(closing.opening_bracket, closing, self.leaves, brackets=(token.LSQB, token.RSQB)):
                return False
            return True
        if self.is_import:
            return True
        if closing.opening_bracket is not None and (not is_one_sequence_between(closing.opening_bracket, closing, self.leaves)):
            return True
        return False

    def append_comment(self, comment: Leaf) -> bool:
        if False:
            print('Hello World!')
        'Add an inline or standalone comment to the line.'
        if comment.type == STANDALONE_COMMENT and self.bracket_tracker.any_open_brackets():
            comment.prefix = ''
            return False
        if comment.type != token.COMMENT:
            return False
        if not self.leaves:
            comment.type = STANDALONE_COMMENT
            comment.prefix = ''
            return False
        last_leaf = self.leaves[-1]
        if last_leaf.type == token.RPAR and (not last_leaf.value) and last_leaf.parent and (len(list(last_leaf.parent.leaves())) <= 3) and (not is_type_comment(comment)):
            if len(self.leaves) < 2:
                comment.type = STANDALONE_COMMENT
                comment.prefix = ''
                return False
            last_leaf = self.leaves[-2]
        self.comments.setdefault(id(last_leaf), []).append(comment)
        return True

    def comments_after(self, leaf: Leaf) -> List[Leaf]:
        if False:
            for i in range(10):
                print('nop')
        'Generate comments that should appear directly after `leaf`.'
        return self.comments.get(id(leaf), [])

    def remove_trailing_comma(self) -> None:
        if False:
            while True:
                i = 10
        'Remove the trailing comma and moves the comments attached to it.'
        trailing_comma = self.leaves.pop()
        trailing_comma_comments = self.comments.pop(id(trailing_comma), [])
        self.comments.setdefault(id(self.leaves[-1]), []).extend(trailing_comma_comments)

    def is_complex_subscript(self, leaf: Leaf) -> bool:
        if False:
            i = 10
            return i + 15
        'Return True iff `leaf` is part of a slice with non-trivial exprs.'
        open_lsqb = self.bracket_tracker.get_open_lsqb()
        if open_lsqb is None:
            return False
        subscript_start = open_lsqb.next_sibling
        if isinstance(subscript_start, Node):
            if subscript_start.type == syms.listmaker:
                return False
            if subscript_start.type == syms.subscriptlist:
                subscript_start = child_towards(subscript_start, leaf)
        return subscript_start is not None and any((n.type in TEST_DESCENDANTS for n in subscript_start.pre_order()))

    def enumerate_with_length(self, reversed: bool=False) -> Iterator[Tuple[Index, Leaf, int]]:
        if False:
            for i in range(10):
                print('nop')
        'Return an enumeration of leaves with their length.\n\n        Stops prematurely on multiline strings and standalone comments.\n        '
        op = cast(Callable[[Sequence[Leaf]], Iterator[Tuple[Index, Leaf]]], enumerate_reversed if reversed else enumerate)
        for (index, leaf) in op(self.leaves):
            length = len(leaf.prefix) + len(leaf.value)
            if '\n' in leaf.value:
                return
            for comment in self.comments_after(leaf):
                length += len(comment.value)
            yield (index, leaf, length)

    def clone(self) -> 'Line':
        if False:
            while True:
                i = 10
        return Line(mode=self.mode, depth=self.depth, inside_brackets=self.inside_brackets, should_split_rhs=self.should_split_rhs, magic_trailing_comma=self.magic_trailing_comma)

    def __str__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Render the line.'
        if not self:
            return '\n'
        indent = '    ' * self.depth
        leaves = iter(self.leaves)
        first = next(leaves)
        res = f'{first.prefix}{indent}{first.value}'
        for leaf in leaves:
            res += str(leaf)
        for comment in itertools.chain.from_iterable(self.comments.values()):
            res += str(comment)
        return res + '\n'

    def __bool__(self) -> bool:
        if False:
            while True:
                i = 10
        'Return True if the line has leaves or comments.'
        return bool(self.leaves or self.comments)

@dataclass
class RHSResult:
    """Intermediate split result from a right hand split."""
    head: Line
    body: Line
    tail: Line
    opening_bracket: Leaf
    closing_bracket: Leaf

@dataclass
class LinesBlock:
    """Class that holds information about a block of formatted lines.

    This is introduced so that the EmptyLineTracker can look behind the standalone
    comments and adjust their empty lines for class or def lines.
    """
    mode: Mode
    previous_block: Optional['LinesBlock']
    original_line: Line
    before: int = 0
    content_lines: List[str] = field(default_factory=list)
    after: int = 0

    def all_lines(self) -> List[str]:
        if False:
            print('Hello World!')
        empty_line = str(Line(mode=self.mode))
        return [empty_line * self.before] + self.content_lines + [empty_line * self.after]

@dataclass
class EmptyLineTracker:
    """Provides a stateful method that returns the number of potential extra
    empty lines needed before and after the currently processed line.

    Note: this tracker works on lines that haven't been split yet.  It assumes
    the prefix of the first leaf consists of optional newlines.  Those newlines
    are consumed by `maybe_empty_lines()` and included in the computation.
    """
    mode: Mode
    previous_line: Optional[Line] = None
    previous_block: Optional[LinesBlock] = None
    previous_defs: List[Line] = field(default_factory=list)
    semantic_leading_comment: Optional[LinesBlock] = None

    def maybe_empty_lines(self, current_line: Line) -> LinesBlock:
        if False:
            while True:
                i = 10
        'Return the number of extra empty lines before and after the `current_line`.\n\n        This is for separating `def`, `async def` and `class` with extra empty\n        lines (two on module-level).\n        '
        (before, after) = self._maybe_empty_lines(current_line)
        previous_after = self.previous_block.after if self.previous_block else 0
        before = 0 if self.previous_line is None else before - previous_after
        if Preview.module_docstring_newlines in current_line.mode and self.previous_block and (self.previous_block.previous_block is None) and (len(self.previous_block.original_line.leaves) == 1) and self.previous_block.original_line.is_triple_quoted_string and (not (current_line.is_class or current_line.is_def)):
            before = 1
        block = LinesBlock(mode=self.mode, previous_block=self.previous_block, original_line=current_line, before=before, after=after)
        if current_line.is_comment:
            if self.previous_line is None or (not self.previous_line.is_decorator and (not self.previous_line.is_comment or before) and (self.semantic_leading_comment is None or before)):
                self.semantic_leading_comment = block
        elif not current_line.is_decorator or before:
            self.semantic_leading_comment = None
        self.previous_line = current_line
        self.previous_block = block
        return block

    def _maybe_empty_lines(self, current_line: Line) -> Tuple[int, int]:
        if False:
            return 10
        max_allowed = 1
        if current_line.depth == 0:
            max_allowed = 1 if self.mode.is_pyi else 2
        if current_line.leaves:
            first_leaf = current_line.leaves[0]
            before = first_leaf.prefix.count('\n')
            before = min(before, max_allowed)
            first_leaf.prefix = ''
        else:
            before = 0
        user_had_newline = bool(before)
        depth = current_line.depth
        previous_def = None
        while self.previous_defs and self.previous_defs[-1].depth >= depth:
            previous_def = self.previous_defs.pop()
        if previous_def is not None:
            assert self.previous_line is not None
            if self.mode.is_pyi:
                if depth and (not current_line.is_def) and self.previous_line.is_def:
                    before = 1 if user_had_newline else 0
                elif Preview.blank_line_after_nested_stub_class in self.mode and previous_def.is_class and (not previous_def.is_stub_class):
                    before = 1
                elif depth:
                    before = 0
                else:
                    before = 1
            elif depth:
                before = 1
            elif not depth and previous_def.depth and (current_line.leaves[-1].type == token.COLON) and (current_line.leaves[0].value not in ('with', 'try', 'for', 'while', 'if', 'match')):
                before = 1
            else:
                before = 2
        if current_line.is_decorator or current_line.is_def or current_line.is_class:
            return self._maybe_empty_lines_for_class_or_def(current_line, before, user_had_newline)
        if self.previous_line and self.previous_line.is_import and (not current_line.is_import) and (not current_line.is_fmt_pass_converted(first_leaf_matches=is_import)) and (depth == self.previous_line.depth):
            return (before or 1, 0)
        if self.previous_line and self.previous_line.is_class and current_line.is_triple_quoted_string:
            if Preview.no_blank_line_before_class_docstring in current_line.mode:
                return (0, 1)
            return (before, 1)
        is_empty_first_line_ok = Preview.allow_empty_first_line_before_new_block_or_comment in current_line.mode and (current_line.leaves[0].type == STANDALONE_COMMENT or current_line.opens_block or (is_docstring(current_line.leaves[0]) and self.previous_line and self.previous_line.leaves[0] and self.previous_line.leaves[0].parent and (not is_funcdef(self.previous_line.leaves[0].parent))))
        if self.previous_line and self.previous_line.opens_block and (not is_empty_first_line_ok):
            return (0, 0)
        return (before, 0)

    def _maybe_empty_lines_for_class_or_def(self, current_line: Line, before: int, user_had_newline: bool) -> Tuple[int, int]:
        if False:
            while True:
                i = 10
        if not current_line.is_decorator:
            self.previous_defs.append(current_line)
        if self.previous_line is None:
            return (0, 0)
        if self.previous_line.is_decorator:
            if self.mode.is_pyi and current_line.is_stub_class:
                return (0, 1)
            return (0, 0)
        if self.previous_line.depth < current_line.depth and (self.previous_line.is_class or self.previous_line.is_def):
            return (0, 0)
        comment_to_add_newlines: Optional[LinesBlock] = None
        if self.previous_line.is_comment and self.previous_line.depth == current_line.depth and (before == 0):
            slc = self.semantic_leading_comment
            if slc is not None and slc.previous_block is not None and (not slc.previous_block.original_line.is_class) and (not slc.previous_block.original_line.opens_block) and (slc.before <= 1):
                comment_to_add_newlines = slc
            else:
                return (0, 0)
        if self.mode.is_pyi:
            if current_line.is_class or self.previous_line.is_class:
                if self.previous_line.depth < current_line.depth:
                    newlines = 0
                elif self.previous_line.depth > current_line.depth:
                    newlines = 1
                elif current_line.is_stub_class and self.previous_line.is_stub_class:
                    newlines = 0
                else:
                    newlines = 1
            elif Preview.blank_line_between_nested_and_def_stub_file in current_line.mode and self.previous_line.depth > current_line.depth:
                newlines = 1
            elif (current_line.is_def or current_line.is_decorator) and (not self.previous_line.is_def):
                if current_line.depth:
                    newlines = min(1, before)
                else:
                    newlines = 1
            elif self.previous_line.depth > current_line.depth:
                newlines = 1
            else:
                newlines = 0
        else:
            newlines = 1 if current_line.depth else 2
            if Preview.dummy_implementations in self.mode and self.previous_line.is_stub_def and (not user_had_newline):
                newlines = 0
        if comment_to_add_newlines is not None:
            previous_block = comment_to_add_newlines.previous_block
            if previous_block is not None:
                comment_to_add_newlines.before = max(comment_to_add_newlines.before, newlines) - previous_block.after
                newlines = 0
        return (newlines, 0)

def enumerate_reversed(sequence: Sequence[T]) -> Iterator[Tuple[Index, T]]:
    if False:
        while True:
            i = 10
    'Like `reversed(enumerate(sequence))` if that were possible.'
    index = len(sequence) - 1
    for element in reversed(sequence):
        yield (index, element)
        index -= 1

def append_leaves(new_line: Line, old_line: Line, leaves: List[Leaf], preformatted: bool=False) -> None:
    if False:
        return 10
    '\n    Append leaves (taken from @old_line) to @new_line, making sure to fix the\n    underlying Node structure where appropriate.\n\n    All of the leaves in @leaves are duplicated. The duplicates are then\n    appended to @new_line and used to replace their originals in the underlying\n    Node structure. Any comments attached to the old leaves are reattached to\n    the new leaves.\n\n    Pre-conditions:\n        set(@leaves) is a subset of set(@old_line.leaves).\n    '
    for old_leaf in leaves:
        new_leaf = Leaf(old_leaf.type, old_leaf.value)
        replace_child(old_leaf, new_leaf)
        new_line.append(new_leaf, preformatted=preformatted)
        for comment_leaf in old_line.comments_after(old_leaf):
            new_line.append(comment_leaf, preformatted=True)

def is_line_short_enough(line: Line, *, mode: Mode, line_str: str='') -> bool:
    if False:
        return 10
    'For non-multiline strings, return True if `line` is no longer than `line_length`.\n    For multiline strings, looks at the context around `line` to determine\n    if it should be inlined or split up.\n    Uses the provided `line_str` rendering, if any, otherwise computes a new one.\n    '
    if not line_str:
        line_str = line_to_string(line)
    width = str_width if mode.preview else len
    if Preview.multiline_string_handling not in mode:
        return width(line_str) <= mode.line_length and '\n' not in line_str and (not line.contains_standalone_comments())
    if line.contains_standalone_comments():
        return False
    if '\n' not in line_str:
        return width(line_str) <= mode.line_length
    (first, *_, last) = line_str.split('\n')
    if width(first) > mode.line_length or width(last) > mode.line_length:
        return False
    commas: List[int] = []
    multiline_string: Optional[Leaf] = None
    multiline_string_contexts: List[LN] = []
    max_level_to_update: Union[int, float] = math.inf
    for (i, leaf) in enumerate(line.leaves):
        if max_level_to_update == math.inf:
            had_comma: Optional[int] = None
            if leaf.bracket_depth + 1 > len(commas):
                commas.append(0)
            elif leaf.bracket_depth + 1 < len(commas):
                had_comma = commas.pop()
            if had_comma is not None and multiline_string is not None and (multiline_string.bracket_depth == leaf.bracket_depth + 1):
                max_level_to_update = leaf.bracket_depth
                if had_comma > 0:
                    return False
        if leaf.bracket_depth <= max_level_to_update and leaf.type == token.COMMA:
            ignore_ctxs: List[Optional[LN]] = [None]
            ignore_ctxs += multiline_string_contexts
            if not (leaf.prev_sibling in ignore_ctxs and i == len(line.leaves) - 1):
                commas[leaf.bracket_depth] += 1
        if max_level_to_update != math.inf:
            max_level_to_update = min(max_level_to_update, leaf.bracket_depth)
        if is_multiline_string(leaf):
            if len(multiline_string_contexts) > 0:
                return False
            multiline_string = leaf
            ctx: LN = leaf
            while str(ctx) in line_str:
                multiline_string_contexts.append(ctx)
                if ctx.parent is None:
                    break
                ctx = ctx.parent
    if len(multiline_string_contexts) == 0:
        return True
    return all((val == 0 for val in commas))

def can_be_split(line: Line) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Return False if the line cannot be split *for sure*.\n\n    This is not an exhaustive search but a cheap heuristic that we can use to\n    avoid some unfortunate formattings (mostly around wrapping unsplittable code\n    in unnecessary parentheses).\n    '
    leaves = line.leaves
    if len(leaves) < 2:
        return False
    if leaves[0].type == token.STRING and leaves[1].type == token.DOT:
        call_count = 0
        dot_count = 0
        next = leaves[-1]
        for leaf in leaves[-2::-1]:
            if leaf.type in OPENING_BRACKETS:
                if next.type not in CLOSING_BRACKETS:
                    return False
                call_count += 1
            elif leaf.type == token.DOT:
                dot_count += 1
            elif leaf.type == token.NAME:
                if not (next.type == token.DOT or next.type in OPENING_BRACKETS):
                    return False
            elif leaf.type not in CLOSING_BRACKETS:
                return False
            if dot_count > 1 and call_count > 1:
                return False
    return True

def can_omit_invisible_parens(rhs: RHSResult, line_length: int) -> bool:
    if False:
        print('Hello World!')
    'Does `rhs.body` have a shape safe to reformat without optional parens around it?\n\n    Returns True for only a subset of potentially nice looking formattings but\n    the point is to not return false positives that end up producing lines that\n    are too long.\n    '
    line = rhs.body
    closing_bracket: Optional[Leaf] = None
    for leaf in reversed(line.leaves):
        if closing_bracket and leaf is closing_bracket.opening_bracket:
            closing_bracket = None
        if leaf.type == STANDALONE_COMMENT and (not closing_bracket):
            return False
        if not closing_bracket and leaf.type in CLOSING_BRACKETS and (leaf.opening_bracket in line.leaves) and leaf.value:
            closing_bracket = leaf
    bt = line.bracket_tracker
    if not bt.delimiters:
        return True
    max_priority = bt.max_delimiter_priority()
    delimiter_count = bt.delimiter_count_with_priority(max_priority)
    if delimiter_count > 1:
        return False
    if delimiter_count == 1:
        if Preview.wrap_multiple_context_managers_in_parens in line.mode and max_priority == COMMA_PRIORITY and rhs.head.is_with_or_async_with_stmt:
            return False
    if max_priority == DOT_PRIORITY:
        return True
    assert len(line.leaves) >= 2, 'Stranded delimiter'
    first = line.leaves[0]
    second = line.leaves[1]
    if first.type in OPENING_BRACKETS and second.type not in CLOSING_BRACKETS:
        if _can_omit_opening_paren(line, first=first, line_length=line_length):
            return True
    penultimate = line.leaves[-2]
    last = line.leaves[-1]
    if last.type == token.RPAR or last.type == token.RBRACE or (last.type == token.RSQB and last.parent and (last.parent.type != syms.trailer)):
        if penultimate.type in OPENING_BRACKETS:
            return False
        if is_multiline_string(first):
            return True
        if _can_omit_closing_paren(line, last=last, line_length=line_length):
            return True
    return False

def _can_omit_opening_paren(line: Line, *, first: Leaf, line_length: int) -> bool:
    if False:
        return 10
    'See `can_omit_invisible_parens`.'
    remainder = False
    length = 4 * line.depth
    _index = -1
    for (_index, leaf, leaf_length) in line.enumerate_with_length():
        if leaf.type in CLOSING_BRACKETS and leaf.opening_bracket is first:
            remainder = True
        if remainder:
            length += leaf_length
            if length > line_length:
                break
            if leaf.type in OPENING_BRACKETS:
                remainder = False
    else:
        if len(line.leaves) == _index + 1:
            return True
    return False

def _can_omit_closing_paren(line: Line, *, last: Leaf, line_length: int) -> bool:
    if False:
        i = 10
        return i + 15
    'See `can_omit_invisible_parens`.'
    length = 4 * line.depth
    seen_other_brackets = False
    for (_index, leaf, leaf_length) in line.enumerate_with_length():
        length += leaf_length
        if leaf is last.opening_bracket:
            if seen_other_brackets or length <= line_length:
                return True
        elif leaf.type in OPENING_BRACKETS:
            seen_other_brackets = True
    return False

def line_to_string(line: Line) -> str:
    if False:
        while True:
            i = 10
    'Returns the string representation of @line.\n\n    WARNING: This is known to be computationally expensive.\n    '
    return str(line).strip('\n')