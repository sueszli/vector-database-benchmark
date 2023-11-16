"""Very low-level nginx config parser based on pyparsing."""
import copy
import logging
import operator
import typing
from typing import Any
from typing import IO
from typing import Iterable
from typing import Iterator
from typing import List
from typing import overload
from typing import SupportsIndex
from typing import Tuple
from typing import Union
from pyparsing import Combine
from pyparsing import Forward
from pyparsing import Group
from pyparsing import Literal
from pyparsing import Optional
from pyparsing import ParseResults
from pyparsing import QuotedString
from pyparsing import Regex
from pyparsing import restOfLine
from pyparsing import stringEnd
from pyparsing import White
from pyparsing import ZeroOrMore
logger = logging.getLogger(__name__)

class RawNginxParser:
    """A class that parses nginx configuration with pyparsing."""
    space = Optional(White()).leaveWhitespace()
    required_space = White().leaveWhitespace()
    left_bracket = Literal('{').suppress()
    right_bracket = space + Literal('}').suppress()
    semicolon = Literal(';').suppress()
    dquoted = QuotedString('"', multiline=True, unquoteResults=False, escChar='\\')
    squoted = QuotedString("'", multiline=True, unquoteResults=False, escChar='\\')
    quoted = dquoted | squoted
    head_tokenchars = Regex('(\\$\\{)|[^{};\\s\'\\"]')
    tail_tokenchars = Regex('(\\$\\{)|[^{;\\s]')
    tokenchars = Combine(head_tokenchars + ZeroOrMore(tail_tokenchars))
    paren_quote_extend = Combine(quoted + Literal(')') + ZeroOrMore(tail_tokenchars))
    token = paren_quote_extend | tokenchars | quoted
    whitespace_token_group = space + token + ZeroOrMore(required_space + token) + space
    assignment = whitespace_token_group + semicolon
    comment = space + Literal('#') + restOfLine
    block = Forward()
    contents = Group(comment) | Group(block) | Group(assignment)
    block_begin = Group(whitespace_token_group)
    block_innards = Group(ZeroOrMore(contents) + space).leaveWhitespace()
    block << block_begin + left_bracket + block_innards + right_bracket
    script = ZeroOrMore(contents) + space + stringEnd
    script.parseWithTabs().leaveWhitespace()

    def __init__(self, source: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.source = source

    def parse(self) -> ParseResults:
        if False:
            for i in range(10):
                print('nop')
        'Returns the parsed tree.'
        return self.script.parseString(self.source)

    def as_list(self) -> List[Any]:
        if False:
            for i in range(10):
                print('nop')
        'Returns the parsed tree as a list.'
        return self.parse().asList()

class RawNginxDumper:
    """A class that dumps nginx configuration from the provided tree."""

    def __init__(self, blocks: List[Any]) -> None:
        if False:
            print('Hello World!')
        self.blocks = blocks

    def __iter__(self, blocks: typing.Optional[List[Any]]=None) -> Iterator[str]:
        if False:
            return 10
        'Iterates the dumped nginx content.'
        blocks = blocks or self.blocks
        for b0 in blocks:
            if isinstance(b0, str):
                yield b0
                continue
            item = copy.deepcopy(b0)
            if spacey(item[0]):
                yield item.pop(0)
                if not item:
                    continue
            if isinstance(item[0], list):
                yield (''.join(item.pop(0)) + '{')
                for parameter in item.pop(0):
                    for line in self.__iter__([parameter]):
                        yield line
                yield '}'
            else:
                semicolon = ';'
                if isinstance(item[0], str) and item[0].strip() == '#':
                    semicolon = ''
                yield (''.join(item) + semicolon)

    def __str__(self) -> str:
        if False:
            i = 10
            return i + 15
        'Return the parsed block as a string.'
        return ''.join(self)

def spacey(x: Any) -> bool:
    if False:
        return 10
    'Is x an empty string or whitespace?'
    return isinstance(x, str) and x.isspace() or x == ''

class UnspacedList(List[Any]):
    """Wrap a list [of lists], making any whitespace entries magically invisible"""

    def __init__(self, list_source: Iterable[Any]) -> None:
        if False:
            while True:
                i = 10
        self.spaced = copy.deepcopy(list(list_source))
        self.dirty = False
        super().__init__(list_source)
        for (i, entry) in reversed(list(enumerate(self))):
            if isinstance(entry, list):
                sublist = UnspacedList(entry)
                super().__setitem__(i, sublist)
                self.spaced[i] = sublist.spaced
            elif spacey(entry):
                if '#' not in self[:i]:
                    super().__delitem__(i)

    @overload
    def _coerce(self, inbound: None) -> Tuple[None, None]:
        if False:
            print('Hello World!')
        ...

    @overload
    def _coerce(self, inbound: str) -> Tuple[str, str]:
        if False:
            while True:
                i = 10
        ...

    @overload
    def _coerce(self, inbound: List[Any]) -> Tuple['UnspacedList', List[Any]]:
        if False:
            return 10
        ...

    def _coerce(self, inbound: Any) -> Tuple[Any, Any]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Coerce some inbound object to be appropriately usable in this object\n\n        :param inbound: string or None or list or UnspacedList\n        :returns: (coerced UnspacedList or string or None, spaced equivalent)\n        :rtype: tuple\n\n        '
        if not isinstance(inbound, list):
            return (inbound, inbound)
        else:
            if not hasattr(inbound, 'spaced'):
                inbound = UnspacedList(inbound)
            return (inbound, inbound.spaced)

    def insert(self, i: 'SupportsIndex', x: Any) -> None:
        if False:
            i = 10
            return i + 15
        'Insert object before index.'
        idx = operator.index(i)
        (item, spaced_item) = self._coerce(x)
        slicepos = self._spaced_position(idx) if idx < len(self) else len(self.spaced)
        self.spaced.insert(slicepos, spaced_item)
        if not spacey(item):
            super().insert(idx, item)
        self.dirty = True

    def append(self, x: Any) -> None:
        if False:
            print('Hello World!')
        'Append object to the end of the list.'
        (item, spaced_item) = self._coerce(x)
        self.spaced.append(spaced_item)
        if not spacey(item):
            super().append(item)
        self.dirty = True

    def extend(self, x: Any) -> None:
        if False:
            i = 10
            return i + 15
        'Extend list by appending elements from the iterable.'
        (item, spaced_item) = self._coerce(x)
        self.spaced.extend(spaced_item)
        super().extend(item)
        self.dirty = True

    def __add__(self, other: List[Any]) -> 'UnspacedList':
        if False:
            i = 10
            return i + 15
        new_list = copy.deepcopy(self)
        new_list.extend(other)
        new_list.dirty = True
        return new_list

    def pop(self, *args: Any, **kwargs: Any) -> None:
        if False:
            return 10
        'Function pop() is not implemented for UnspacedList'
        raise NotImplementedError('UnspacedList.pop() not yet implemented')

    def remove(self, *args: Any, **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        'Function remove() is not implemented for UnspacedList'
        raise NotImplementedError('UnspacedList.remove() not yet implemented')

    def reverse(self) -> None:
        if False:
            i = 10
            return i + 15
        'Function reverse() is not implemented for UnspacedList'
        raise NotImplementedError('UnspacedList.reverse() not yet implemented')

    def sort(self, *_args: Any, **_kwargs: Any) -> None:
        if False:
            return 10
        'Function sort() is not implemented for UnspacedList'
        raise NotImplementedError('UnspacedList.sort() not yet implemented')

    def __setslice__(self, *args: Any, **kwargs: Any) -> None:
        if False:
            return 10
        raise NotImplementedError('Slice operations on UnspacedLists not yet implemented')

    def __setitem__(self, i: Union['SupportsIndex', slice], value: Any) -> None:
        if False:
            while True:
                i = 10
        if isinstance(i, slice):
            raise NotImplementedError('Slice operations on UnspacedLists not yet implemented')
        (item, spaced_item) = self._coerce(value)
        self.spaced.__setitem__(self._spaced_position(i), spaced_item)
        if not spacey(item):
            super().__setitem__(i, item)
        self.dirty = True

    def __delitem__(self, i: Union['SupportsIndex', slice]) -> None:
        if False:
            i = 10
            return i + 15
        if isinstance(i, slice):
            raise NotImplementedError('Slice operations on UnspacedLists not yet implemented')
        self.spaced.__delitem__(self._spaced_position(i))
        super().__delitem__(i)
        self.dirty = True

    def __deepcopy__(self, memo: Any) -> 'UnspacedList':
        if False:
            print('Hello World!')
        new_spaced = copy.deepcopy(self.spaced, memo=memo)
        new_list = UnspacedList(new_spaced)
        new_list.dirty = self.dirty
        return new_list

    def is_dirty(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Recurse through the parse tree to figure out if any sublists are dirty'
        if self.dirty:
            return True
        return any((isinstance(x, UnspacedList) and x.is_dirty() for x in self))

    def _spaced_position(self, idx: 'SupportsIndex') -> int:
        if False:
            return 10
        'Convert from indexes in the unspaced list to positions in the spaced one'
        int_idx = operator.index(idx)
        pos = spaces = 0
        if int_idx < 0:
            int_idx = len(self) + int_idx
        if not 0 <= int_idx < len(self):
            raise IndexError('list index out of range')
        int_idx0 = int_idx
        while int_idx != -1:
            if spacey(self.spaced[pos]):
                spaces += 1
            else:
                int_idx -= 1
            pos += 1
        return int_idx0 + spaces

def loads(source: str) -> UnspacedList:
    if False:
        for i in range(10):
            print('nop')
    'Parses from a string.\n\n    :param str source: The string to parse\n    :returns: The parsed tree\n    :rtype: list\n\n    '
    return UnspacedList(RawNginxParser(source).as_list())

def load(file_: IO[Any]) -> UnspacedList:
    if False:
        print('Hello World!')
    'Parses from a file.\n\n    :param file file_: The file to parse\n    :returns: The parsed tree\n    :rtype: list\n\n    '
    return loads(file_.read())

def dumps(blocks: UnspacedList) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Dump to a Unicode string.\n\n    :param UnspacedList blocks: The parsed tree\n    :rtype: six.text_type\n\n    '
    return str(RawNginxDumper(blocks.spaced))

def dump(blocks: UnspacedList, file_: IO[Any]) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Dump to a file.\n\n    :param UnspacedList blocks: The parsed tree\n    :param IO[Any] file_: The file stream to dump to. It must be opened with\n                          Unicode encoding.\n    :rtype: None\n\n    '
    file_.write(dumps(blocks))