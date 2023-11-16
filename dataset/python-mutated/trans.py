"""
String transformers that can split and merge strings.
"""
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Collection, Dict, Final, Iterable, Iterator, List, Literal, Optional, Sequence, Set, Tuple, TypeVar, Union
from mypy_extensions import trait
from black.comments import contains_pragma_comment
from black.lines import Line, append_leaves
from black.mode import Feature, Mode
from black.nodes import CLOSING_BRACKETS, OPENING_BRACKETS, STANDALONE_COMMENT, is_empty_lpar, is_empty_par, is_empty_rpar, is_part_of_annotation, parent_type, replace_child, syms
from black.rusty import Err, Ok, Result
from black.strings import assert_is_leaf_string, count_chars_in_width, get_string_prefix, has_triple_quotes, normalize_string_quotes, str_width
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node

class CannotTransform(Exception):
    """Base class for errors raised by Transformers."""
T = TypeVar('T')
LN = Union[Leaf, Node]
Transformer = Callable[[Line, Collection[Feature], Mode], Iterator[Line]]
Index = int
NodeType = int
ParserState = int
StringID = int
TResult = Result[T, CannotTransform]
TMatchResult = TResult[List[Index]]
SPLIT_SAFE_CHARS = frozenset(['、', '。', '，'])

def TErr(err_msg: str) -> Err[CannotTransform]:
    if False:
        for i in range(10):
            print('nop')
    '(T)ransform Err\n\n    Convenience function used when working with the TResult type.\n    '
    cant_transform = CannotTransform(err_msg)
    return Err(cant_transform)

def hug_power_op(line: Line, features: Collection[Feature], mode: Mode) -> Iterator[Line]:
    if False:
        return 10
    'A transformer which normalizes spacing around power operators.'
    for leaf in line.leaves:
        if leaf.type == token.DOUBLESTAR:
            break
    else:
        raise CannotTransform('No doublestar token was found in the line.')

    def is_simple_lookup(index: int, step: Literal[1, -1]) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if step == -1:
            disallowed = {token.RPAR, token.RSQB}
        else:
            disallowed = {token.LPAR, token.LSQB}
        while 0 <= index < len(line.leaves):
            current = line.leaves[index]
            if current.type in disallowed:
                return False
            if current.type not in {token.NAME, token.DOT} or current.value == 'for':
                return True
            index += step
        return True

    def is_simple_operand(index: int, kind: Literal['base', 'exponent']) -> bool:
        if False:
            i = 10
            return i + 15
        start = line.leaves[index]
        if start.type in {token.NAME, token.NUMBER}:
            return is_simple_lookup(index, step=1 if kind == 'exponent' else -1)
        if start.type in {token.PLUS, token.MINUS, token.TILDE}:
            if line.leaves[index + 1].type in {token.NAME, token.NUMBER}:
                return is_simple_lookup(index + 1, step=1)
        return False
    new_line = line.clone()
    should_hug = False
    for (idx, leaf) in enumerate(line.leaves):
        new_leaf = leaf.clone()
        if should_hug:
            new_leaf.prefix = ''
            should_hug = False
        should_hug = 0 < idx < len(line.leaves) - 1 and leaf.type == token.DOUBLESTAR and is_simple_operand(idx - 1, kind='base') and (line.leaves[idx - 1].value != 'lambda') and is_simple_operand(idx + 1, kind='exponent')
        if should_hug:
            new_leaf.prefix = ''
        new_line.append(new_leaf, preformatted=True)
        for comment_leaf in line.comments_after(leaf):
            new_line.append(comment_leaf, preformatted=True)
    yield new_line

class StringTransformer(ABC):
    """
    An implementation of the Transformer protocol that relies on its
    subclasses overriding the template methods `do_match(...)` and
    `do_transform(...)`.

    This Transformer works exclusively on strings (for example, by merging
    or splitting them).

    The following sections can be found among the docstrings of each concrete
    StringTransformer subclass.

    Requirements:
        Which requirements must be met of the given Line for this
        StringTransformer to be applied?

    Transformations:
        If the given Line meets all of the above requirements, which string
        transformations can you expect to be applied to it by this
        StringTransformer?

    Collaborations:
        What contractual agreements does this StringTransformer have with other
        StringTransfomers? Such collaborations should be eliminated/minimized
        as much as possible.
    """
    __name__: Final = 'StringTransformer'

    def __init__(self, line_length: int, normalize_strings: bool) -> None:
        if False:
            print('Hello World!')
        self.line_length = line_length
        self.normalize_strings = normalize_strings

    @abstractmethod
    def do_match(self, line: Line) -> TMatchResult:
        if False:
            while True:
                i = 10
        "\n        Returns:\n            * Ok(string_indices) such that for each index, `line.leaves[index]`\n              is our target string if a match was able to be made. For\n              transformers that don't result in more lines (e.g. StringMerger,\n              StringParenStripper), multiple matches and transforms are done at\n              once to reduce the complexity.\n              OR\n            * Err(CannotTransform), if no match could be made.\n        "

    @abstractmethod
    def do_transform(self, line: Line, string_indices: List[int]) -> Iterator[TResult[Line]]:
        if False:
            print('Hello World!')
        "\n        Yields:\n            * Ok(new_line) where new_line is the new transformed line.\n              OR\n            * Err(CannotTransform) if the transformation failed for some reason. The\n              `do_match(...)` template method should usually be used to reject\n              the form of the given Line, but in some cases it is difficult to\n              know whether or not a Line meets the StringTransformer's\n              requirements until the transformation is already midway.\n\n        Side Effects:\n            This method should NOT mutate @line directly, but it MAY mutate the\n            Line's underlying Node structure. (WARNING: If the underlying Node\n            structure IS altered, then this method should NOT be allowed to\n            yield an CannotTransform after that point.)\n        "

    def __call__(self, line: Line, _features: Collection[Feature], _mode: Mode) -> Iterator[Line]:
        if False:
            i = 10
            return i + 15
        '\n        StringTransformer instances have a call signature that mirrors that of\n        the Transformer type.\n\n        Raises:\n            CannotTransform(...) if the concrete StringTransformer class is unable\n            to transform @line.\n        '
        if not any((leaf.type == token.STRING for leaf in line.leaves)):
            raise CannotTransform('There are no strings in this line.')
        match_result = self.do_match(line)
        if isinstance(match_result, Err):
            cant_transform = match_result.err()
            raise CannotTransform(f'The string transformer {self.__class__.__name__} does not recognize this line as one that it can transform.') from cant_transform
        string_indices = match_result.ok()
        for line_result in self.do_transform(line, string_indices):
            if isinstance(line_result, Err):
                cant_transform = line_result.err()
                raise CannotTransform('StringTransformer failed while attempting to transform string.') from cant_transform
            line = line_result.ok()
            yield line

@dataclass
class CustomSplit:
    """A custom (i.e. manual) string split.

    A single CustomSplit instance represents a single substring.

    Examples:
        Consider the following string:
        ```
        "Hi there friend."
        " This is a custom"
        f" string {split}."
        ```

        This string will correspond to the following three CustomSplit instances:
        ```
        CustomSplit(False, 16)
        CustomSplit(False, 17)
        CustomSplit(True, 16)
        ```
    """
    has_prefix: bool
    break_idx: int

@trait
class CustomSplitMapMixin:
    """
    This mixin class is used to map merged strings to a sequence of
    CustomSplits, which will then be used to re-split the strings iff none of
    the resultant substrings go over the configured max line length.
    """
    _Key: ClassVar = Tuple[StringID, str]
    _CUSTOM_SPLIT_MAP: ClassVar[Dict[_Key, Tuple[CustomSplit, ...]]] = defaultdict(tuple)

    @staticmethod
    def _get_key(string: str) -> 'CustomSplitMapMixin._Key':
        if False:
            print('Hello World!')
        '\n        Returns:\n            A unique identifier that is used internally to map @string to a\n            group of custom splits.\n        '
        return (id(string), string)

    def add_custom_splits(self, string: str, custom_splits: Iterable[CustomSplit]) -> None:
        if False:
            return 10
        'Custom Split Map Setter Method\n\n        Side Effects:\n            Adds a mapping from @string to the custom splits @custom_splits.\n        '
        key = self._get_key(string)
        self._CUSTOM_SPLIT_MAP[key] = tuple(custom_splits)

    def pop_custom_splits(self, string: str) -> List[CustomSplit]:
        if False:
            while True:
                i = 10
        'Custom Split Map Getter Method\n\n        Returns:\n            * A list of the custom splits that are mapped to @string, if any\n              exist.\n              OR\n            * [], otherwise.\n\n        Side Effects:\n            Deletes the mapping between @string and its associated custom\n            splits (which are returned to the caller).\n        '
        key = self._get_key(string)
        custom_splits = self._CUSTOM_SPLIT_MAP[key]
        del self._CUSTOM_SPLIT_MAP[key]
        return list(custom_splits)

    def has_custom_splits(self, string: str) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Returns:\n            True iff @string is associated with a set of custom splits.\n        '
        key = self._get_key(string)
        return key in self._CUSTOM_SPLIT_MAP

class StringMerger(StringTransformer, CustomSplitMapMixin):
    """StringTransformer that merges strings together.

    Requirements:
        (A) The line contains adjacent strings such that ALL of the validation checks
        listed in StringMerger._validate_msg(...)'s docstring pass.
        OR
        (B) The line contains a string which uses line continuation backslashes.

    Transformations:
        Depending on which of the two requirements above where met, either:

        (A) The string group associated with the target string is merged.
        OR
        (B) All line-continuation backslashes are removed from the target string.

    Collaborations:
        StringMerger provides custom split information to StringSplitter.
    """

    def do_match(self, line: Line) -> TMatchResult:
        if False:
            while True:
                i = 10
        LL = line.leaves
        is_valid_index = is_valid_index_factory(LL)
        string_indices = []
        idx = 0
        while is_valid_index(idx):
            leaf = LL[idx]
            if leaf.type == token.STRING and is_valid_index(idx + 1) and (LL[idx + 1].type == token.STRING):
                contains_comment = False
                i = idx
                while is_valid_index(i):
                    if LL[i].type != token.STRING:
                        break
                    if line.comments_after(LL[i]):
                        contains_comment = True
                        break
                    i += 1
                if not is_part_of_annotation(leaf) and (not contains_comment):
                    string_indices.append(idx)
                idx += 2
                while is_valid_index(idx) and LL[idx].type == token.STRING:
                    idx += 1
            elif leaf.type == token.STRING and '\\\n' in leaf.value:
                string_indices.append(idx)
                idx += 1
                while is_valid_index(idx) and LL[idx].type == token.STRING:
                    idx += 1
            else:
                idx += 1
        if string_indices:
            return Ok(string_indices)
        else:
            return TErr('This line has no strings that need merging.')

    def do_transform(self, line: Line, string_indices: List[int]) -> Iterator[TResult[Line]]:
        if False:
            i = 10
            return i + 15
        new_line = line
        rblc_result = self._remove_backslash_line_continuation_chars(new_line, string_indices)
        if isinstance(rblc_result, Ok):
            new_line = rblc_result.ok()
        msg_result = self._merge_string_group(new_line, string_indices)
        if isinstance(msg_result, Ok):
            new_line = msg_result.ok()
        if isinstance(rblc_result, Err) and isinstance(msg_result, Err):
            msg_cant_transform = msg_result.err()
            rblc_cant_transform = rblc_result.err()
            cant_transform = CannotTransform('StringMerger failed to merge any strings in this line.')
            msg_cant_transform.__cause__ = rblc_cant_transform
            cant_transform.__cause__ = msg_cant_transform
            yield Err(cant_transform)
        else:
            yield Ok(new_line)

    @staticmethod
    def _remove_backslash_line_continuation_chars(line: Line, string_indices: List[int]) -> TResult[Line]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Merge strings that were split across multiple lines using\n        line-continuation backslashes.\n\n        Returns:\n            Ok(new_line), if @line contains backslash line-continuation\n            characters.\n                OR\n            Err(CannotTransform), otherwise.\n        '
        LL = line.leaves
        indices_to_transform = []
        for string_idx in string_indices:
            string_leaf = LL[string_idx]
            if string_leaf.type == token.STRING and '\\\n' in string_leaf.value and (not has_triple_quotes(string_leaf.value)):
                indices_to_transform.append(string_idx)
        if not indices_to_transform:
            return TErr('Found no string leaves that contain backslash line continuation characters.')
        new_line = line.clone()
        new_line.comments = line.comments.copy()
        append_leaves(new_line, line, LL)
        for string_idx in indices_to_transform:
            new_string_leaf = new_line.leaves[string_idx]
            new_string_leaf.value = new_string_leaf.value.replace('\\\n', '')
        return Ok(new_line)

    def _merge_string_group(self, line: Line, string_indices: List[int]) -> TResult[Line]:
        if False:
            while True:
                i = 10
        "\n        Merges string groups (i.e. set of adjacent strings).\n\n        Each index from `string_indices` designates one string group's first\n        leaf in `line.leaves`.\n\n        Returns:\n            Ok(new_line), if ALL of the validation checks found in\n            _validate_msg(...) pass.\n                OR\n            Err(CannotTransform), otherwise.\n        "
        LL = line.leaves
        is_valid_index = is_valid_index_factory(LL)
        merged_string_idx_dict: Dict[int, Tuple[int, Leaf]] = {}
        for string_idx in string_indices:
            vresult = self._validate_msg(line, string_idx)
            if isinstance(vresult, Err):
                continue
            merged_string_idx_dict[string_idx] = self._merge_one_string_group(LL, string_idx, is_valid_index)
        if not merged_string_idx_dict:
            return TErr('No string group is merged')
        new_line = line.clone()
        previous_merged_string_idx = -1
        previous_merged_num_of_strings = -1
        for (i, leaf) in enumerate(LL):
            if i in merged_string_idx_dict:
                previous_merged_string_idx = i
                (previous_merged_num_of_strings, string_leaf) = merged_string_idx_dict[i]
                new_line.append(string_leaf)
            if previous_merged_string_idx <= i < previous_merged_string_idx + previous_merged_num_of_strings:
                for comment_leaf in line.comments_after(LL[i]):
                    new_line.append(comment_leaf, preformatted=True)
                continue
            append_leaves(new_line, line, [leaf])
        return Ok(new_line)

    def _merge_one_string_group(self, LL: List[Leaf], string_idx: int, is_valid_index: Callable[[int], bool]) -> Tuple[int, Leaf]:
        if False:
            while True:
                i = 10
        '\n        Merges one string group where the first string in the group is\n        `LL[string_idx]`.\n\n        Returns:\n            A tuple of `(num_of_strings, leaf)` where `num_of_strings` is the\n            number of strings merged and `leaf` is the newly merged string\n            to be replaced in the new line.\n        '
        atom_node = LL[string_idx].parent
        BREAK_MARK = '@@@@@ BLACK BREAKPOINT MARKER @@@@@'
        QUOTE = LL[string_idx].value[-1]

        def make_naked(string: str, string_prefix: str) -> str:
            if False:
                i = 10
                return i + 15
            'Strip @string (i.e. make it a "naked" string)\n\n            Pre-conditions:\n                * assert_is_leaf_string(@string)\n\n            Returns:\n                A string that is identical to @string except that\n                @string_prefix has been stripped, the surrounding QUOTE\n                characters have been removed, and any remaining QUOTE\n                characters have been escaped.\n            '
            assert_is_leaf_string(string)
            if 'f' in string_prefix:
                f_expressions = (string[span[0] + 1:span[1] - 1] for span in iter_fexpr_spans(string))
                debug_expressions_contain_visible_quotes = any((re.search('.*[\\\'\\"].*(?<![!:=])={1}(?!=)(?![^\\s:])', expression) for expression in f_expressions))
                if not debug_expressions_contain_visible_quotes:
                    string = _toggle_fexpr_quotes(string, QUOTE)
            RE_EVEN_BACKSLASHES = '(?:(?<!\\\\)(?:\\\\\\\\)*)'
            naked_string = string[len(string_prefix) + 1:-1]
            naked_string = re.sub('(' + RE_EVEN_BACKSLASHES + ')' + QUOTE, '\\1\\\\' + QUOTE, naked_string)
            return naked_string
        custom_splits = []
        prefix_tracker = []
        next_str_idx = string_idx
        prefix = ''
        while not prefix and is_valid_index(next_str_idx) and (LL[next_str_idx].type == token.STRING):
            prefix = get_string_prefix(LL[next_str_idx].value).lower()
            next_str_idx += 1
        S = ''
        NS = ''
        num_of_strings = 0
        next_str_idx = string_idx
        while is_valid_index(next_str_idx) and LL[next_str_idx].type == token.STRING:
            num_of_strings += 1
            SS = LL[next_str_idx].value
            next_prefix = get_string_prefix(SS).lower()
            if 'f' in prefix and 'f' not in next_prefix:
                SS = re.sub('(\\{|\\})', '\\1\\1', SS)
            NSS = make_naked(SS, next_prefix)
            has_prefix = bool(next_prefix)
            prefix_tracker.append(has_prefix)
            S = prefix + QUOTE + NS + NSS + BREAK_MARK + QUOTE
            NS = make_naked(S, prefix)
            next_str_idx += 1
        non_string_idx = next_str_idx
        S_leaf = Leaf(token.STRING, S)
        if self.normalize_strings:
            S_leaf.value = normalize_string_quotes(S_leaf.value)
        temp_string = S_leaf.value[len(prefix) + 1:-1]
        for has_prefix in prefix_tracker:
            mark_idx = temp_string.find(BREAK_MARK)
            assert mark_idx >= 0, 'Logic error while filling the custom string breakpoint cache.'
            temp_string = temp_string[mark_idx + len(BREAK_MARK):]
            breakpoint_idx = mark_idx + (len(prefix) if has_prefix else 0) + 1
            custom_splits.append(CustomSplit(has_prefix, breakpoint_idx))
        string_leaf = Leaf(token.STRING, S_leaf.value.replace(BREAK_MARK, ''))
        if atom_node is not None:
            if non_string_idx - string_idx < len(atom_node.children):
                first_child_idx = LL[string_idx].remove()
                for idx in range(string_idx + 1, non_string_idx):
                    LL[idx].remove()
                if first_child_idx is not None:
                    atom_node.insert_child(first_child_idx, string_leaf)
            else:
                replace_child(atom_node, string_leaf)
        self.add_custom_splits(string_leaf.value, custom_splits)
        return (num_of_strings, string_leaf)

    @staticmethod
    def _validate_msg(line: Line, string_idx: int) -> TResult[None]:
        if False:
            for i in range(10):
                print('nop')
        'Validate (M)erge (S)tring (G)roup\n\n        Transform-time string validation logic for _merge_string_group(...).\n\n        Returns:\n            * Ok(None), if ALL validation checks (listed below) pass.\n                OR\n            * Err(CannotTransform), if any of the following are true:\n                - The target string group does not contain ANY stand-alone comments.\n                - The target string is not in a string group (i.e. it has no\n                  adjacent strings).\n                - The string group has more than one inline comment.\n                - The string group has an inline comment that appears to be a pragma.\n                - The set of all string prefixes in the string group is of\n                  length greater than one and is not equal to {"", "f"}.\n                - The string group consists of raw strings.\n                - The string group is stringified type annotations. We don\'t want to\n                  process stringified type annotations since pyright doesn\'t support\n                  them spanning multiple string values. (NOTE: mypy, pytype, pyre do\n                  support them, so we can change if pyright also gains support in the\n                  future. See https://github.com/microsoft/pyright/issues/4359.)\n        '
        for inc in [1, -1]:
            i = string_idx
            found_sa_comment = False
            is_valid_index = is_valid_index_factory(line.leaves)
            while is_valid_index(i) and line.leaves[i].type in [token.STRING, STANDALONE_COMMENT]:
                if line.leaves[i].type == STANDALONE_COMMENT:
                    found_sa_comment = True
                elif found_sa_comment:
                    return TErr('StringMerger does NOT merge string groups which contain stand-alone comments.')
                i += inc
        num_of_inline_string_comments = 0
        set_of_prefixes = set()
        num_of_strings = 0
        for leaf in line.leaves[string_idx:]:
            if leaf.type != token.STRING:
                if leaf.type == token.COMMA and id(leaf) in line.comments:
                    num_of_inline_string_comments += 1
                break
            if has_triple_quotes(leaf.value):
                return TErr('StringMerger does NOT merge multiline strings.')
            num_of_strings += 1
            prefix = get_string_prefix(leaf.value).lower()
            if 'r' in prefix:
                return TErr('StringMerger does NOT merge raw strings.')
            set_of_prefixes.add(prefix)
            if id(leaf) in line.comments:
                num_of_inline_string_comments += 1
                if contains_pragma_comment(line.comments[id(leaf)]):
                    return TErr('Cannot merge strings which have pragma comments.')
        if num_of_strings < 2:
            return TErr(f'Not enough strings to merge (num_of_strings={num_of_strings}).')
        if num_of_inline_string_comments > 1:
            return TErr(f'Too many inline string comments ({num_of_inline_string_comments}).')
        if len(set_of_prefixes) > 1 and set_of_prefixes != {'', 'f'}:
            return TErr(f'Too many different prefixes ({set_of_prefixes}).')
        return Ok(None)

class StringParenStripper(StringTransformer):
    """StringTransformer that strips surrounding parentheses from strings.

    Requirements:
        The line contains a string which is surrounded by parentheses and:
            - The target string is NOT the only argument to a function call.
            - The target string is NOT a "pointless" string.
            - If the target string contains a PERCENT, the brackets are not
              preceded or followed by an operator with higher precedence than
              PERCENT.

    Transformations:
        The parentheses mentioned in the 'Requirements' section are stripped.

    Collaborations:
        StringParenStripper has its own inherent usefulness, but it is also
        relied on to clean up the parentheses created by StringParenWrapper (in
        the event that they are no longer needed).
    """

    def do_match(self, line: Line) -> TMatchResult:
        if False:
            for i in range(10):
                print('nop')
        LL = line.leaves
        is_valid_index = is_valid_index_factory(LL)
        string_indices = []
        idx = -1
        while True:
            idx += 1
            if idx >= len(LL):
                break
            leaf = LL[idx]
            if leaf.type != token.STRING:
                continue
            if leaf.parent and leaf.parent.parent and (leaf.parent.parent.type == syms.simple_stmt):
                continue
            if not is_valid_index(idx - 1) or LL[idx - 1].type != token.LPAR or is_empty_lpar(LL[idx - 1]):
                continue
            if is_valid_index(idx - 2) and (LL[idx - 2].type == token.NAME or LL[idx - 2].type in CLOSING_BRACKETS):
                continue
            string_idx = idx
            string_parser = StringParser()
            next_idx = string_parser.parse(LL, string_idx)
            if is_valid_index(idx - 2):
                before_lpar = LL[idx - 2]
                if token.PERCENT in {leaf.type for leaf in LL[idx - 1:next_idx]} and (before_lpar.type in {token.STAR, token.AT, token.SLASH, token.DOUBLESLASH, token.PERCENT, token.TILDE, token.DOUBLESTAR, token.AWAIT, token.LSQB, token.LPAR} or (before_lpar.parent and before_lpar.parent.type == syms.factor and (before_lpar.type in {token.PLUS, token.MINUS}))):
                    continue
            if is_valid_index(next_idx) and LL[next_idx].type == token.RPAR and (not is_empty_rpar(LL[next_idx])):
                if is_valid_index(next_idx + 1) and LL[next_idx + 1].type in {token.DOUBLESTAR, token.LSQB, token.LPAR, token.DOT}:
                    continue
                string_indices.append(string_idx)
                idx = string_idx
                while idx < len(LL) - 1 and LL[idx + 1].type == token.STRING:
                    idx += 1
        if string_indices:
            return Ok(string_indices)
        return TErr('This line has no strings wrapped in parens.')

    def do_transform(self, line: Line, string_indices: List[int]) -> Iterator[TResult[Line]]:
        if False:
            return 10
        LL = line.leaves
        string_and_rpar_indices: List[int] = []
        for string_idx in string_indices:
            string_parser = StringParser()
            rpar_idx = string_parser.parse(LL, string_idx)
            should_transform = True
            for leaf in (LL[string_idx - 1], LL[rpar_idx]):
                if line.comments_after(leaf):
                    should_transform = False
                    break
            if should_transform:
                string_and_rpar_indices.extend((string_idx, rpar_idx))
        if string_and_rpar_indices:
            yield Ok(self._transform_to_new_line(line, string_and_rpar_indices))
        else:
            yield Err(CannotTransform('All string groups have comments attached to them.'))

    def _transform_to_new_line(self, line: Line, string_and_rpar_indices: List[int]) -> Line:
        if False:
            for i in range(10):
                print('nop')
        LL = line.leaves
        new_line = line.clone()
        new_line.comments = line.comments.copy()
        previous_idx = -1
        for idx in sorted(string_and_rpar_indices):
            leaf = LL[idx]
            lpar_or_rpar_idx = idx - 1 if leaf.type == token.STRING else idx
            append_leaves(new_line, line, LL[previous_idx + 1:lpar_or_rpar_idx])
            if leaf.type == token.STRING:
                string_leaf = Leaf(token.STRING, LL[idx].value)
                LL[lpar_or_rpar_idx].remove()
                replace_child(LL[idx], string_leaf)
                new_line.append(string_leaf)
                old_comments = new_line.comments.pop(id(LL[idx]), [])
                new_line.comments.setdefault(id(string_leaf), []).extend(old_comments)
            else:
                LL[lpar_or_rpar_idx].remove()
            previous_idx = idx
        append_leaves(new_line, line, LL[idx + 1:])
        return new_line

class BaseStringSplitter(StringTransformer):
    """
    Abstract class for StringTransformers which transform a Line's strings by splitting
    them or placing them on their own lines where necessary to avoid going over
    the configured line length.

    Requirements:
        * The target string value is responsible for the line going over the
          line length limit. It follows that after all of black's other line
          split methods have been exhausted, this line (or one of the resulting
          lines after all line splits are performed) would still be over the
          line_length limit unless we split this string.
          AND

        * The target string is NOT a "pointless" string (i.e. a string that has
          no parent or siblings).
          AND

        * The target string is not followed by an inline comment that appears
          to be a pragma.
          AND

        * The target string is not a multiline (i.e. triple-quote) string.
    """
    STRING_OPERATORS: Final = [token.EQEQUAL, token.GREATER, token.GREATEREQUAL, token.LESS, token.LESSEQUAL, token.NOTEQUAL, token.PERCENT, token.PLUS, token.STAR]

    @abstractmethod
    def do_splitter_match(self, line: Line) -> TMatchResult:
        if False:
            return 10
        '\n        BaseStringSplitter asks its clients to override this method instead of\n        `StringTransformer.do_match(...)`.\n\n        Follows the same protocol as `StringTransformer.do_match(...)`.\n\n        Refer to `help(StringTransformer.do_match)` for more information.\n        '

    def do_match(self, line: Line) -> TMatchResult:
        if False:
            print('Hello World!')
        match_result = self.do_splitter_match(line)
        if isinstance(match_result, Err):
            return match_result
        string_indices = match_result.ok()
        assert len(string_indices) == 1, f'{self.__class__.__name__} should only find one match at a time, found {len(string_indices)}'
        string_idx = string_indices[0]
        vresult = self._validate(line, string_idx)
        if isinstance(vresult, Err):
            return vresult
        return match_result

    def _validate(self, line: Line, string_idx: int) -> TResult[None]:
        if False:
            print('Hello World!')
        "\n        Checks that @line meets all of the requirements listed in this classes'\n        docstring. Refer to `help(BaseStringSplitter)` for a detailed\n        description of those requirements.\n\n        Returns:\n            * Ok(None), if ALL of the requirements are met.\n              OR\n            * Err(CannotTransform), if ANY of the requirements are NOT met.\n        "
        LL = line.leaves
        string_leaf = LL[string_idx]
        max_string_length = self._get_max_string_length(line, string_idx)
        if len(string_leaf.value) <= max_string_length:
            return TErr('The string itself is not what is causing this line to be too long.')
        if not string_leaf.parent or [L.type for L in string_leaf.parent.children] == [token.STRING, token.NEWLINE]:
            return TErr(f'This string ({string_leaf.value}) appears to be pointless (i.e. has no parent).')
        if id(line.leaves[string_idx]) in line.comments and contains_pragma_comment(line.comments[id(line.leaves[string_idx])]):
            return TErr("Line appears to end with an inline pragma comment. Splitting the line could modify the pragma's behavior.")
        if has_triple_quotes(string_leaf.value):
            return TErr('We cannot split multiline strings.')
        return Ok(None)

    def _get_max_string_length(self, line: Line, string_idx: int) -> int:
        if False:
            i = 10
            return i + 15
        '\n        Calculates the max string length used when attempting to determine\n        whether or not the target string is responsible for causing the line to\n        go over the line length limit.\n\n        WARNING: This method is tightly coupled to both StringSplitter and\n        (especially) StringParenWrapper. There is probably a better way to\n        accomplish what is being done here.\n\n        Returns:\n            max_string_length: such that `line.leaves[string_idx].value >\n            max_string_length` implies that the target string IS responsible\n            for causing this line to exceed the line length limit.\n        '
        LL = line.leaves
        is_valid_index = is_valid_index_factory(LL)
        offset = line.depth * 4
        if is_valid_index(string_idx - 1):
            p_idx = string_idx - 1
            if LL[string_idx - 1].type == token.LPAR and LL[string_idx - 1].value == '' and (string_idx >= 2):
                p_idx -= 1
            P = LL[p_idx]
            if P.type in self.STRING_OPERATORS:
                offset += len(str(P)) + 1
            if P.type == token.COMMA:
                offset += 3
            if P.type in [token.COLON, token.EQUAL, token.PLUSEQUAL, token.NAME]:
                offset += 1
                for leaf in reversed(LL[:p_idx + 1]):
                    offset += len(str(leaf))
                    if leaf.type in CLOSING_BRACKETS:
                        break
        if is_valid_index(string_idx + 1):
            N = LL[string_idx + 1]
            if N.type == token.RPAR and N.value == '' and (len(LL) > string_idx + 2):
                N = LL[string_idx + 2]
            if N.type == token.COMMA:
                offset += 1
            if is_valid_index(string_idx + 2):
                NN = LL[string_idx + 2]
                if N.type == token.DOT and NN.type == token.NAME:
                    offset += 1
                    if is_valid_index(string_idx + 3) and LL[string_idx + 3].type == token.LPAR:
                        offset += 1
                    offset += len(NN.value)
        has_comments = False
        for comment_leaf in line.comments_after(LL[string_idx]):
            if not has_comments:
                has_comments = True
                offset += 2
            offset += len(comment_leaf.value)
        max_string_length = count_chars_in_width(str(line), self.line_length - offset)
        return max_string_length

    @staticmethod
    def _prefer_paren_wrap_match(LL: List[Leaf]) -> Optional[int]:
        if False:
            return 10
        '\n        Returns:\n            string_idx such that @LL[string_idx] is equal to our target (i.e.\n            matched) string, if this line matches the "prefer paren wrap" statement\n            requirements listed in the \'Requirements\' section of the StringParenWrapper\n            class\'s docstring.\n                OR\n            None, otherwise.\n        '
        if LL[0].type != token.STRING:
            return None
        matching_nodes = [syms.listmaker, syms.dictsetmaker, syms.testlist_gexp]
        if parent_type(LL[0]) in matching_nodes or parent_type(LL[0].parent) in matching_nodes:
            prev_sibling = LL[0].prev_sibling
            next_sibling = LL[0].next_sibling
            if not prev_sibling and (not next_sibling) and (parent_type(LL[0]) == syms.atom):
                parent = LL[0].parent
                assert parent is not None
                prev_sibling = parent.prev_sibling
                next_sibling = parent.next_sibling
            if (not prev_sibling or prev_sibling.type == token.COMMA) and (not next_sibling or next_sibling.type == token.COMMA):
                return 0
        return None

def iter_fexpr_spans(s: str) -> Iterator[Tuple[int, int]]:
    if False:
        while True:
            i = 10
    '\n    Yields spans corresponding to expressions in a given f-string.\n    Spans are half-open ranges (left inclusive, right exclusive).\n    Assumes the input string is a valid f-string, but will not crash if the input\n    string is invalid.\n    '
    stack: List[int] = []
    i = 0
    while i < len(s):
        if s[i] == '{':
            if not stack and i + 1 < len(s) and (s[i + 1] == '{'):
                i += 2
                continue
            stack.append(i)
            i += 1
            continue
        if s[i] == '}':
            if not stack:
                i += 1
                continue
            j = stack.pop()
            if not stack:
                yield (j, i + 1)
            i += 1
            continue
        if stack:
            delim = None
            if s[i:i + 3] in ("'''", '"""'):
                delim = s[i:i + 3]
            elif s[i] in ("'", '"'):
                delim = s[i]
            if delim:
                i += len(delim)
                while i < len(s) and s[i:i + len(delim)] != delim:
                    i += 1
                i += len(delim)
                continue
        i += 1

def fstring_contains_expr(s: str) -> bool:
    if False:
        while True:
            i = 10
    return any(iter_fexpr_spans(s))

def _toggle_fexpr_quotes(fstring: str, old_quote: str) -> str:
    if False:
        i = 10
        return i + 15
    "\n    Toggles quotes used in f-string expressions that are `old_quote`.\n\n    f-string expressions can't contain backslashes, so we need to toggle the\n    quotes if the f-string itself will end up using the same quote. We can\n    simply toggle without escaping because, quotes can't be reused in f-string\n    expressions. They will fail to parse.\n\n    NOTE: If PEP 701 is accepted, above statement will no longer be true.\n    Though if quotes can be reused, we can simply reuse them without updates or\n    escaping, once Black figures out how to parse the new grammar.\n    "
    new_quote = "'" if old_quote == '"' else '"'
    parts = []
    previous_index = 0
    for (start, end) in iter_fexpr_spans(fstring):
        parts.append(fstring[previous_index:start])
        parts.append(fstring[start:end].replace(old_quote, new_quote))
        previous_index = end
    parts.append(fstring[previous_index:])
    return ''.join(parts)

class StringSplitter(BaseStringSplitter, CustomSplitMapMixin):
    """
    StringTransformer that splits "atom" strings (i.e. strings which exist on
    lines by themselves).

    Requirements:
        * The line consists ONLY of a single string (possibly prefixed by a
          string operator [e.g. '+' or '==']), MAYBE a string trailer, and MAYBE
          a trailing comma.
          AND
        * All of the requirements listed in BaseStringSplitter's docstring.

    Transformations:
        The string mentioned in the 'Requirements' section is split into as
        many substrings as necessary to adhere to the configured line length.

        In the final set of substrings, no substring should be smaller than
        MIN_SUBSTR_SIZE characters.

        The string will ONLY be split on spaces (i.e. each new substring should
        start with a space). Note that the string will NOT be split on a space
        which is escaped with a backslash.

        If the string is an f-string, it will NOT be split in the middle of an
        f-expression (e.g. in f"FooBar: {foo() if x else bar()}", {foo() if x
        else bar()} is an f-expression).

        If the string that is being split has an associated set of custom split
        records and those custom splits will NOT result in any line going over
        the configured line length, those custom splits are used. Otherwise the
        string is split as late as possible (from left-to-right) while still
        adhering to the transformation rules listed above.

    Collaborations:
        StringSplitter relies on StringMerger to construct the appropriate
        CustomSplit objects and add them to the custom split map.
    """
    MIN_SUBSTR_SIZE: Final = 6

    def do_splitter_match(self, line: Line) -> TMatchResult:
        if False:
            return 10
        LL = line.leaves
        if self._prefer_paren_wrap_match(LL) is not None:
            return TErr('Line needs to be wrapped in parens first.')
        is_valid_index = is_valid_index_factory(LL)
        idx = 0
        if is_valid_index(idx) and is_valid_index(idx + 1) and ([LL[idx].type, LL[idx + 1].type] == [token.NAME, token.NAME]) and (str(LL[idx]) + str(LL[idx + 1]) == 'not in'):
            idx += 2
        elif is_valid_index(idx) and (LL[idx].type in self.STRING_OPERATORS or (LL[idx].type == token.NAME and str(LL[idx]) == 'in')):
            idx += 1
        if is_valid_index(idx) and is_empty_lpar(LL[idx]):
            idx += 1
        if not is_valid_index(idx) or LL[idx].type != token.STRING:
            return TErr('Line does not start with a string.')
        string_idx = idx
        string_parser = StringParser()
        idx = string_parser.parse(LL, string_idx)
        if is_valid_index(idx) and is_empty_rpar(LL[idx]):
            idx += 1
        if is_valid_index(idx) and LL[idx].type == token.COMMA:
            idx += 1
        if is_valid_index(idx):
            return TErr('This line does not end with a string.')
        return Ok([string_idx])

    def do_transform(self, line: Line, string_indices: List[int]) -> Iterator[TResult[Line]]:
        if False:
            i = 10
            return i + 15
        LL = line.leaves
        assert len(string_indices) == 1, f'{self.__class__.__name__} should only find one match at a time, found {len(string_indices)}'
        string_idx = string_indices[0]
        QUOTE = LL[string_idx].value[-1]
        is_valid_index = is_valid_index_factory(LL)
        insert_str_child = insert_str_child_factory(LL[string_idx])
        prefix = get_string_prefix(LL[string_idx].value).lower()
        drop_pointless_f_prefix = 'f' in prefix and fstring_contains_expr(LL[string_idx].value)
        first_string_line = True
        string_op_leaves = self._get_string_operator_leaves(LL)
        string_op_leaves_length = sum((len(str(prefix_leaf)) for prefix_leaf in string_op_leaves)) + 1 if string_op_leaves else 0

        def maybe_append_string_operators(new_line: Line) -> None:
            if False:
                while True:
                    i = 10
            '\n            Side Effects:\n                If @line starts with a string operator and this is the first\n                line we are constructing, this function appends the string\n                operator to @new_line and replaces the old string operator leaf\n                in the node structure. Otherwise this function does nothing.\n            '
            maybe_prefix_leaves = string_op_leaves if first_string_line else []
            for (i, prefix_leaf) in enumerate(maybe_prefix_leaves):
                replace_child(LL[i], prefix_leaf)
                new_line.append(prefix_leaf)
        ends_with_comma = is_valid_index(string_idx + 1) and LL[string_idx + 1].type == token.COMMA

        def max_last_string_column() -> int:
            if False:
                print('Hello World!')
            '\n            Returns:\n                The max allowed width of the string value used for the last\n                line we will construct.  Note that this value means the width\n                rather than the number of characters (e.g., many East Asian\n                characters expand to two columns).\n            '
            result = self.line_length
            result -= line.depth * 4
            result -= 1 if ends_with_comma else 0
            result -= string_op_leaves_length
            return result
        max_break_width = self.line_length
        max_break_width -= 1
        max_break_width -= line.depth * 4
        if max_break_width < 0:
            yield TErr(f'Unable to split {LL[string_idx].value} at such high of a line depth: {line.depth}')
            return
        custom_splits = self.pop_custom_splits(LL[string_idx].value)
        use_custom_breakpoints = bool(custom_splits and all((csplit.break_idx <= max_break_width for csplit in custom_splits)))
        rest_value = LL[string_idx].value

        def more_splits_should_be_made() -> bool:
            if False:
                while True:
                    i = 10
            '\n            Returns:\n                True iff `rest_value` (the remaining string value from the last\n                split), should be split again.\n            '
            if use_custom_breakpoints:
                return len(custom_splits) > 1
            else:
                return str_width(rest_value) > max_last_string_column()
        string_line_results: List[Ok[Line]] = []
        while more_splits_should_be_made():
            if use_custom_breakpoints:
                csplit = custom_splits.pop(0)
                break_idx = csplit.break_idx
            else:
                max_bidx = count_chars_in_width(rest_value, max_break_width) - string_op_leaves_length
                maybe_break_idx = self._get_break_idx(rest_value, max_bidx)
                if maybe_break_idx is None:
                    if custom_splits:
                        rest_value = LL[string_idx].value
                        string_line_results = []
                        first_string_line = True
                        use_custom_breakpoints = True
                        continue
                    break
                break_idx = maybe_break_idx
            next_value = rest_value[:break_idx] + QUOTE
            if use_custom_breakpoints and (not csplit.has_prefix) and (next_value == prefix + QUOTE or next_value != self._normalize_f_string(next_value, prefix)):
                break_idx += 1
                next_value = rest_value[:break_idx] + QUOTE
            if drop_pointless_f_prefix:
                next_value = self._normalize_f_string(next_value, prefix)
            next_leaf = Leaf(token.STRING, next_value)
            insert_str_child(next_leaf)
            self._maybe_normalize_string_quotes(next_leaf)
            next_line = line.clone()
            maybe_append_string_operators(next_line)
            next_line.append(next_leaf)
            string_line_results.append(Ok(next_line))
            rest_value = prefix + QUOTE + rest_value[break_idx:]
            first_string_line = False
        yield from string_line_results
        if drop_pointless_f_prefix:
            rest_value = self._normalize_f_string(rest_value, prefix)
        rest_leaf = Leaf(token.STRING, rest_value)
        insert_str_child(rest_leaf)
        self._maybe_normalize_string_quotes(rest_leaf)
        last_line = line.clone()
        maybe_append_string_operators(last_line)
        if is_valid_index(string_idx + 1):
            temp_value = rest_value
            for leaf in LL[string_idx + 1:]:
                temp_value += str(leaf)
                if leaf.type == token.LPAR:
                    break
            if str_width(temp_value) <= max_last_string_column() or LL[string_idx + 1].type == token.COMMA:
                last_line.append(rest_leaf)
                append_leaves(last_line, line, LL[string_idx + 1:])
                yield Ok(last_line)
            else:
                last_line.append(rest_leaf)
                yield Ok(last_line)
                non_string_line = line.clone()
                append_leaves(non_string_line, line, LL[string_idx + 1:])
                yield Ok(non_string_line)
        else:
            last_line.append(rest_leaf)
            last_line.comments = line.comments.copy()
            yield Ok(last_line)

    def _iter_nameescape_slices(self, string: str) -> Iterator[Tuple[Index, Index]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Yields:\n            All ranges of @string which, if @string were to be split there,\n            would result in the splitting of an \\N{...} expression (which is NOT\n            allowed).\n        '
        previous_was_unescaped_backslash = False
        it = iter(enumerate(string))
        for (idx, c) in it:
            if c == '\\':
                previous_was_unescaped_backslash = not previous_was_unescaped_backslash
                continue
            if not previous_was_unescaped_backslash or c != 'N':
                previous_was_unescaped_backslash = False
                continue
            previous_was_unescaped_backslash = False
            begin = idx - 1
            for (idx, c) in it:
                if c == '}':
                    end = idx
                    break
            else:
                raise RuntimeError(f'{self.__class__.__name__} LOGIC ERROR!')
            yield (begin, end)

    def _iter_fexpr_slices(self, string: str) -> Iterator[Tuple[Index, Index]]:
        if False:
            i = 10
            return i + 15
        '\n        Yields:\n            All ranges of @string which, if @string were to be split there,\n            would result in the splitting of an f-expression (which is NOT\n            allowed).\n        '
        if 'f' not in get_string_prefix(string).lower():
            return
        yield from iter_fexpr_spans(string)

    def _get_illegal_split_indices(self, string: str) -> Set[Index]:
        if False:
            for i in range(10):
                print('nop')
        illegal_indices: Set[Index] = set()
        iterators = [self._iter_fexpr_slices(string), self._iter_nameescape_slices(string)]
        for it in iterators:
            for (begin, end) in it:
                illegal_indices.update(range(begin, end + 1))
        return illegal_indices

    def _get_break_idx(self, string: str, max_break_idx: int) -> Optional[int]:
        if False:
            for i in range(10):
                print('nop')
        "\n        This method contains the algorithm that StringSplitter uses to\n        determine which character to split each string at.\n\n        Args:\n            @string: The substring that we are attempting to split.\n            @max_break_idx: The ideal break index. We will return this value if it\n            meets all the necessary conditions. In the likely event that it\n            doesn't we will try to find the closest index BELOW @max_break_idx\n            that does. If that fails, we will expand our search by also\n            considering all valid indices ABOVE @max_break_idx.\n\n        Pre-Conditions:\n            * assert_is_leaf_string(@string)\n            * 0 <= @max_break_idx < len(@string)\n\n        Returns:\n            break_idx, if an index is able to be found that meets all of the\n            conditions listed in the 'Transformations' section of this classes'\n            docstring.\n                OR\n            None, otherwise.\n        "
        is_valid_index = is_valid_index_factory(string)
        assert is_valid_index(max_break_idx)
        assert_is_leaf_string(string)
        _illegal_split_indices = self._get_illegal_split_indices(string)

        def breaks_unsplittable_expression(i: Index) -> bool:
            if False:
                i = 10
                return i + 15
            '\n            Returns:\n                True iff returning @i would result in the splitting of an\n                unsplittable expression (which is NOT allowed).\n            '
            return i in _illegal_split_indices

        def passes_all_checks(i: Index) -> bool:
            if False:
                print('Hello World!')
            "\n            Returns:\n                True iff ALL of the conditions listed in the 'Transformations'\n                section of this classes' docstring would be be met by returning @i.\n            "
            is_space = string[i] == ' '
            is_split_safe = is_valid_index(i - 1) and string[i - 1] in SPLIT_SAFE_CHARS
            is_not_escaped = True
            j = i - 1
            while is_valid_index(j) and string[j] == '\\':
                is_not_escaped = not is_not_escaped
                j -= 1
            is_big_enough = len(string[i:]) >= self.MIN_SUBSTR_SIZE and len(string[:i]) >= self.MIN_SUBSTR_SIZE
            return (is_space or is_split_safe) and is_not_escaped and is_big_enough and (not breaks_unsplittable_expression(i))
        break_idx = max_break_idx
        while is_valid_index(break_idx - 1) and (not passes_all_checks(break_idx)):
            break_idx -= 1
        if not passes_all_checks(break_idx):
            break_idx = max_break_idx + 1
            while is_valid_index(break_idx + 1) and (not passes_all_checks(break_idx)):
                break_idx += 1
            if not is_valid_index(break_idx) or not passes_all_checks(break_idx):
                return None
        return break_idx

    def _maybe_normalize_string_quotes(self, leaf: Leaf) -> None:
        if False:
            while True:
                i = 10
        if self.normalize_strings:
            leaf.value = normalize_string_quotes(leaf.value)

    def _normalize_f_string(self, string: str, prefix: str) -> str:
        if False:
            while True:
                i = 10
        "\n        Pre-Conditions:\n            * assert_is_leaf_string(@string)\n\n        Returns:\n            * If @string is an f-string that contains no f-expressions, we\n            return a string identical to @string except that the 'f' prefix\n            has been stripped and all double braces (i.e. '{{' or '}}') have\n            been normalized (i.e. turned into '{' or '}').\n                OR\n            * Otherwise, we return @string.\n        "
        assert_is_leaf_string(string)
        if 'f' in prefix and (not fstring_contains_expr(string)):
            new_prefix = prefix.replace('f', '')
            temp = string[len(prefix):]
            temp = re.sub('\\{\\{', '{', temp)
            temp = re.sub('\\}\\}', '}', temp)
            new_string = temp
            return f'{new_prefix}{new_string}'
        else:
            return string

    def _get_string_operator_leaves(self, leaves: Iterable[Leaf]) -> List[Leaf]:
        if False:
            i = 10
            return i + 15
        LL = list(leaves)
        string_op_leaves = []
        i = 0
        while LL[i].type in self.STRING_OPERATORS + [token.NAME]:
            prefix_leaf = Leaf(LL[i].type, str(LL[i]).strip())
            string_op_leaves.append(prefix_leaf)
            i += 1
        return string_op_leaves

class StringParenWrapper(BaseStringSplitter, CustomSplitMapMixin):
    """
    StringTransformer that wraps strings in parens and then splits at the LPAR.

    Requirements:
        All of the requirements listed in BaseStringSplitter's docstring in
        addition to the requirements listed below:

        * The line is a return/yield statement, which returns/yields a string.
          OR
        * The line is part of a ternary expression (e.g. `x = y if cond else
          z`) such that the line starts with `else <string>`, where <string> is
          some string.
          OR
        * The line is an assert statement, which ends with a string.
          OR
        * The line is an assignment statement (e.g. `x = <string>` or `x +=
          <string>`) such that the variable is being assigned the value of some
          string.
          OR
        * The line is a dictionary key assignment where some valid key is being
          assigned the value of some string.
          OR
        * The line is an lambda expression and the value is a string.
          OR
        * The line starts with an "atom" string that prefers to be wrapped in
          parens. It's preferred to be wrapped when it's is an immediate child of
          a list/set/tuple literal, AND the string is surrounded by commas (or is
          the first/last child).

    Transformations:
        The chosen string is wrapped in parentheses and then split at the LPAR.

        We then have one line which ends with an LPAR and another line that
        starts with the chosen string. The latter line is then split again at
        the RPAR. This results in the RPAR (and possibly a trailing comma)
        being placed on its own line.

        NOTE: If any leaves exist to the right of the chosen string (except
        for a trailing comma, which would be placed after the RPAR), those
        leaves are placed inside the parentheses.  In effect, the chosen
        string is not necessarily being "wrapped" by parentheses. We can,
        however, count on the LPAR being placed directly before the chosen
        string.

        In other words, StringParenWrapper creates "atom" strings. These
        can then be split again by StringSplitter, if necessary.

    Collaborations:
        In the event that a string line split by StringParenWrapper is
        changed such that it no longer needs to be given its own line,
        StringParenWrapper relies on StringParenStripper to clean up the
        parentheses it created.

        For "atom" strings that prefers to be wrapped in parens, it requires
        StringSplitter to hold the split until the string is wrapped in parens.
    """

    def do_splitter_match(self, line: Line) -> TMatchResult:
        if False:
            return 10
        LL = line.leaves
        if line.leaves[-1].type in OPENING_BRACKETS:
            return TErr('Cannot wrap parens around a line that ends in an opening bracket.')
        string_idx = self._return_match(LL) or self._else_match(LL) or self._assert_match(LL) or self._assign_match(LL) or self._dict_or_lambda_match(LL) or self._prefer_paren_wrap_match(LL)
        if string_idx is not None:
            string_value = line.leaves[string_idx].value
            if not any((char == ' ' or char in SPLIT_SAFE_CHARS for char in string_value)):
                max_string_width = self.line_length - (line.depth + 1) * 4
                if str_width(string_value) > max_string_width:
                    if not self.has_custom_splits(string_value):
                        return TErr("We do not wrap long strings in parentheses when the resultant line would still be over the specified line length and can't be split further by StringSplitter.")
            return Ok([string_idx])
        return TErr('This line does not contain any non-atomic strings.')

    @staticmethod
    def _return_match(LL: List[Leaf]) -> Optional[int]:
        if False:
            i = 10
            return i + 15
        "\n        Returns:\n            string_idx such that @LL[string_idx] is equal to our target (i.e.\n            matched) string, if this line matches the return/yield statement\n            requirements listed in the 'Requirements' section of this classes'\n            docstring.\n                OR\n            None, otherwise.\n        "
        if parent_type(LL[0]) in [syms.return_stmt, syms.yield_expr] and LL[0].value in ['return', 'yield']:
            is_valid_index = is_valid_index_factory(LL)
            idx = 2 if is_valid_index(1) and is_empty_par(LL[1]) else 1
            if is_valid_index(idx) and LL[idx].type == token.STRING:
                return idx
        return None

    @staticmethod
    def _else_match(LL: List[Leaf]) -> Optional[int]:
        if False:
            print('Hello World!')
        "\n        Returns:\n            string_idx such that @LL[string_idx] is equal to our target (i.e.\n            matched) string, if this line matches the ternary expression\n            requirements listed in the 'Requirements' section of this classes'\n            docstring.\n                OR\n            None, otherwise.\n        "
        if parent_type(LL[0]) == syms.test and LL[0].type == token.NAME and (LL[0].value == 'else'):
            is_valid_index = is_valid_index_factory(LL)
            idx = 2 if is_valid_index(1) and is_empty_par(LL[1]) else 1
            if is_valid_index(idx) and LL[idx].type == token.STRING:
                return idx
        return None

    @staticmethod
    def _assert_match(LL: List[Leaf]) -> Optional[int]:
        if False:
            i = 10
            return i + 15
        "\n        Returns:\n            string_idx such that @LL[string_idx] is equal to our target (i.e.\n            matched) string, if this line matches the assert statement\n            requirements listed in the 'Requirements' section of this classes'\n            docstring.\n                OR\n            None, otherwise.\n        "
        if parent_type(LL[0]) == syms.assert_stmt and LL[0].value == 'assert':
            is_valid_index = is_valid_index_factory(LL)
            for (i, leaf) in enumerate(LL):
                if leaf.type == token.COMMA:
                    idx = i + 2 if is_empty_par(LL[i + 1]) else i + 1
                    if is_valid_index(idx) and LL[idx].type == token.STRING:
                        string_idx = idx
                        string_parser = StringParser()
                        idx = string_parser.parse(LL, string_idx)
                        if not is_valid_index(idx):
                            return string_idx
        return None

    @staticmethod
    def _assign_match(LL: List[Leaf]) -> Optional[int]:
        if False:
            for i in range(10):
                print('nop')
        "\n        Returns:\n            string_idx such that @LL[string_idx] is equal to our target (i.e.\n            matched) string, if this line matches the assignment statement\n            requirements listed in the 'Requirements' section of this classes'\n            docstring.\n                OR\n            None, otherwise.\n        "
        if parent_type(LL[0]) in [syms.expr_stmt, syms.argument, syms.power] and LL[0].type == token.NAME:
            is_valid_index = is_valid_index_factory(LL)
            for (i, leaf) in enumerate(LL):
                if leaf.type in [token.EQUAL, token.PLUSEQUAL]:
                    idx = i + 2 if is_empty_par(LL[i + 1]) else i + 1
                    if is_valid_index(idx) and LL[idx].type == token.STRING:
                        string_idx = idx
                        string_parser = StringParser()
                        idx = string_parser.parse(LL, string_idx)
                        if parent_type(LL[0]) == syms.argument and is_valid_index(idx) and (LL[idx].type == token.COMMA):
                            idx += 1
                        if not is_valid_index(idx):
                            return string_idx
        return None

    @staticmethod
    def _dict_or_lambda_match(LL: List[Leaf]) -> Optional[int]:
        if False:
            for i in range(10):
                print('nop')
        "\n        Returns:\n            string_idx such that @LL[string_idx] is equal to our target (i.e.\n            matched) string, if this line matches the dictionary key assignment\n            statement or lambda expression requirements listed in the\n            'Requirements' section of this classes' docstring.\n                OR\n            None, otherwise.\n        "
        parent_types = [parent_type(LL[0]), parent_type(LL[0].parent)]
        if syms.dictsetmaker in parent_types or syms.lambdef in parent_types:
            is_valid_index = is_valid_index_factory(LL)
            for (i, leaf) in enumerate(LL):
                if leaf.type == token.COLON and i < len(LL) - 1:
                    idx = i + 2 if is_empty_par(LL[i + 1]) else i + 1
                    if is_valid_index(idx) and LL[idx].type == token.STRING:
                        string_idx = idx
                        string_parser = StringParser()
                        idx = string_parser.parse(LL, string_idx)
                        if is_valid_index(idx) and LL[idx].type == token.COMMA:
                            idx += 1
                        if not is_valid_index(idx):
                            return string_idx
        return None

    def do_transform(self, line: Line, string_indices: List[int]) -> Iterator[TResult[Line]]:
        if False:
            return 10
        LL = line.leaves
        assert len(string_indices) == 1, f'{self.__class__.__name__} should only find one match at a time, found {len(string_indices)}'
        string_idx = string_indices[0]
        is_valid_index = is_valid_index_factory(LL)
        insert_str_child = insert_str_child_factory(LL[string_idx])
        comma_idx = -1
        ends_with_comma = False
        if LL[comma_idx].type == token.COMMA:
            ends_with_comma = True
        leaves_to_steal_comments_from = [LL[string_idx]]
        if ends_with_comma:
            leaves_to_steal_comments_from.append(LL[comma_idx])
        first_line = line.clone()
        left_leaves = LL[:string_idx]
        old_parens_exist = False
        if left_leaves and left_leaves[-1].type == token.LPAR:
            old_parens_exist = True
            leaves_to_steal_comments_from.append(left_leaves[-1])
            left_leaves.pop()
        append_leaves(first_line, line, left_leaves)
        lpar_leaf = Leaf(token.LPAR, '(')
        if old_parens_exist:
            replace_child(LL[string_idx - 1], lpar_leaf)
        else:
            insert_str_child(lpar_leaf)
        first_line.append(lpar_leaf)
        for leaf in leaves_to_steal_comments_from:
            for comment_leaf in line.comments_after(leaf):
                first_line.append(comment_leaf, preformatted=True)
        yield Ok(first_line)
        string_value = LL[string_idx].value
        string_line = Line(mode=line.mode, depth=line.depth + 1, inside_brackets=True, should_split_rhs=line.should_split_rhs, magic_trailing_comma=line.magic_trailing_comma)
        string_leaf = Leaf(token.STRING, string_value)
        insert_str_child(string_leaf)
        string_line.append(string_leaf)
        old_rpar_leaf = None
        if is_valid_index(string_idx + 1):
            right_leaves = LL[string_idx + 1:]
            if ends_with_comma:
                right_leaves.pop()
            if old_parens_exist:
                assert right_leaves and right_leaves[-1].type == token.RPAR, f'Apparently, old parentheses do NOT exist?! (left_leaves={left_leaves}, right_leaves={right_leaves})'
                old_rpar_leaf = right_leaves.pop()
            elif right_leaves and right_leaves[-1].type == token.RPAR:
                opening_bracket = right_leaves[-1].opening_bracket
                if opening_bracket is not None and opening_bracket in left_leaves:
                    index = left_leaves.index(opening_bracket)
                    if index > 0 and index < len(left_leaves) - 1 and (left_leaves[index - 1].type == token.COLON) and (left_leaves[index + 1].value == 'lambda'):
                        right_leaves.pop()
            append_leaves(string_line, line, right_leaves)
        yield Ok(string_line)
        last_line = line.clone()
        last_line.bracket_tracker = first_line.bracket_tracker
        new_rpar_leaf = Leaf(token.RPAR, ')')
        if old_rpar_leaf is not None:
            replace_child(old_rpar_leaf, new_rpar_leaf)
        else:
            insert_str_child(new_rpar_leaf)
        last_line.append(new_rpar_leaf)
        if ends_with_comma:
            comma_leaf = Leaf(token.COMMA, ',')
            replace_child(LL[comma_idx], comma_leaf)
            last_line.append(comma_leaf)
        yield Ok(last_line)

class StringParser:
    """
    A state machine that aids in parsing a string's "trailer", which can be
    either non-existent, an old-style formatting sequence (e.g. `% varX` or `%
    (varX, varY)`), or a method-call / attribute access (e.g. `.format(varX,
    varY)`).

    NOTE: A new StringParser object MUST be instantiated for each string
    trailer we need to parse.

    Examples:
        We shall assume that `line` equals the `Line` object that corresponds
        to the following line of python code:
        ```
        x = "Some {}.".format("String") + some_other_string
        ```

        Furthermore, we will assume that `string_idx` is some index such that:
        ```
        assert line.leaves[string_idx].value == "Some {}."
        ```

        The following code snippet then holds:
        ```
        string_parser = StringParser()
        idx = string_parser.parse(line.leaves, string_idx)
        assert line.leaves[idx].type == token.PLUS
        ```
    """
    DEFAULT_TOKEN: Final = 20210605
    START: Final = 1
    DOT: Final = 2
    NAME: Final = 3
    PERCENT: Final = 4
    SINGLE_FMT_ARG: Final = 5
    LPAR: Final = 6
    RPAR: Final = 7
    DONE: Final = 8
    _goto: Final[Dict[Tuple[ParserState, NodeType], ParserState]] = {(START, token.DOT): DOT, (START, token.PERCENT): PERCENT, (START, DEFAULT_TOKEN): DONE, (DOT, token.NAME): NAME, (NAME, token.LPAR): LPAR, (NAME, DEFAULT_TOKEN): DONE, (PERCENT, token.LPAR): LPAR, (PERCENT, DEFAULT_TOKEN): SINGLE_FMT_ARG, (SINGLE_FMT_ARG, DEFAULT_TOKEN): DONE, (RPAR, DEFAULT_TOKEN): DONE}

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        self._state = self.START
        self._unmatched_lpars = 0

    def parse(self, leaves: List[Leaf], string_idx: int) -> int:
        if False:
            print('Hello World!')
        '\n        Pre-conditions:\n            * @leaves[@string_idx].type == token.STRING\n\n        Returns:\n            The index directly after the last leaf which is apart of the string\n            trailer, if a "trailer" exists.\n            OR\n            @string_idx + 1, if no string "trailer" exists.\n        '
        assert leaves[string_idx].type == token.STRING
        idx = string_idx + 1
        while idx < len(leaves) and self._next_state(leaves[idx]):
            idx += 1
        return idx

    def _next_state(self, leaf: Leaf) -> bool:
        if False:
            print('Hello World!')
        "\n        Pre-conditions:\n            * On the first call to this function, @leaf MUST be the leaf that\n              was directly after the string leaf in question (e.g. if our target\n              string is `line.leaves[i]` then the first call to this method must\n              be `line.leaves[i + 1]`).\n            * On the next call to this function, the leaf parameter passed in\n              MUST be the leaf directly following @leaf.\n\n        Returns:\n            True iff @leaf is apart of the string's trailer.\n        "
        if is_empty_par(leaf):
            return True
        next_token = leaf.type
        if next_token == token.LPAR:
            self._unmatched_lpars += 1
        current_state = self._state
        if current_state == self.LPAR:
            if next_token == token.RPAR:
                self._unmatched_lpars -= 1
                if self._unmatched_lpars == 0:
                    self._state = self.RPAR
        else:
            if (current_state, next_token) in self._goto:
                self._state = self._goto[current_state, next_token]
            elif (current_state, self.DEFAULT_TOKEN) in self._goto:
                self._state = self._goto[current_state, self.DEFAULT_TOKEN]
            else:
                raise RuntimeError(f'{self.__class__.__name__} LOGIC ERROR!')
            if self._state == self.DONE:
                return False
        return True

def insert_str_child_factory(string_leaf: Leaf) -> Callable[[LN], None]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Factory for a convenience function that is used to orphan @string_leaf\n    and then insert multiple new leaves into the same part of the node\n    structure that @string_leaf had originally occupied.\n\n    Examples:\n        Let `string_leaf = Leaf(token.STRING, \'"foo"\')` and `N =\n        string_leaf.parent`. Assume the node `N` has the following\n        original structure:\n\n        Node(\n            expr_stmt, [\n                Leaf(NAME, \'x\'),\n                Leaf(EQUAL, \'=\'),\n                Leaf(STRING, \'"foo"\'),\n            ]\n        )\n\n        We then run the code snippet shown below.\n        ```\n        insert_str_child = insert_str_child_factory(string_leaf)\n\n        lpar = Leaf(token.LPAR, \'(\')\n        insert_str_child(lpar)\n\n        bar = Leaf(token.STRING, \'"bar"\')\n        insert_str_child(bar)\n\n        rpar = Leaf(token.RPAR, \')\')\n        insert_str_child(rpar)\n        ```\n\n        After which point, it follows that `string_leaf.parent is None` and\n        the node `N` now has the following structure:\n\n        Node(\n            expr_stmt, [\n                Leaf(NAME, \'x\'),\n                Leaf(EQUAL, \'=\'),\n                Leaf(LPAR, \'(\'),\n                Leaf(STRING, \'"bar"\'),\n                Leaf(RPAR, \')\'),\n            ]\n        )\n    '
    string_parent = string_leaf.parent
    string_child_idx = string_leaf.remove()

    def insert_str_child(child: LN) -> None:
        if False:
            print('Hello World!')
        nonlocal string_child_idx
        assert string_parent is not None
        assert string_child_idx is not None
        string_parent.insert_child(string_child_idx, child)
        string_child_idx += 1
    return insert_str_child

def is_valid_index_factory(seq: Sequence[Any]) -> Callable[[int], bool]:
    if False:
        return 10
    '\n    Examples:\n        ```\n        my_list = [1, 2, 3]\n\n        is_valid_index = is_valid_index_factory(my_list)\n\n        assert is_valid_index(0)\n        assert is_valid_index(2)\n\n        assert not is_valid_index(3)\n        assert not is_valid_index(-1)\n        ```\n    '

    def is_valid_index(idx: int) -> bool:
        if False:
            return 10
        '\n        Returns:\n            True iff @idx is positive AND seq[@idx] does NOT raise an\n            IndexError.\n        '
        return 0 <= idx < len(seq)
    return is_valid_index