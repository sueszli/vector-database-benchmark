""" This file contains parsing routines and object classes to help derive meaning from
raw lists of tokens from pyparsing. """
import abc
import logging
from typing import Any
from typing import Callable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from certbot import errors
logger = logging.getLogger(__name__)
COMMENT = ' managed by Certbot'
COMMENT_BLOCK = ['#', COMMENT]

class Parsable:
    """ Abstract base class for "Parsable" objects whose underlying representation
    is a tree of lists.

    :param .Parsable parent: This object's parsed parent in the tree
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, parent: Optional['Parsable']=None):
        if False:
            return 10
        self._data: List[Any] = []
        self._tabs = None
        self.parent = parent

    @classmethod
    def parsing_hooks(cls) -> Tuple[Type['Block'], Type['Sentence'], Type['Statements']]:
        if False:
            while True:
                i = 10
        'Returns object types that this class should be able to `parse` recusrively.\n        The order of the objects indicates the order in which the parser should\n        try to parse each subitem.\n        :returns: A list of Parsable classes.\n        :rtype list:\n        '
        return (Block, Sentence, Statements)

    @staticmethod
    @abc.abstractmethod
    def should_parse(lists: Any) -> bool:
        if False:
            i = 10
            return i + 15
        ' Returns whether the contents of `lists` can be parsed into this object.\n\n        :returns: Whether `lists` can be parsed as this object.\n        :rtype bool:\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def parse(self, raw_list: List[Any], add_spaces: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        ' Loads information into this object from underlying raw_list structure.\n        Each Parsable object might make different assumptions about the structure of\n        raw_list.\n\n        :param list raw_list: A list or sublist of tokens from pyparsing, containing whitespace\n            as separate tokens.\n        :param bool add_spaces: If set, the method can and should manipulate and insert spacing\n            between non-whitespace tokens and lists to delimit them.\n        :raises .errors.MisconfigurationError: when the assumptions about the structure of\n            raw_list are not met.\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def iterate(self, expanded: bool=False, match: Optional[Callable[['Parsable'], bool]]=None) -> Iterator[Any]:
        if False:
            for i in range(10):
                print('nop')
        ' Iterates across this object. If this object is a leaf object, only yields\n        itself. If it contains references other parsing objects, and `expanded` is set,\n        this function should first yield itself, then recursively iterate across all of them.\n        :param bool expanded: Whether to recursively iterate on possible children.\n        :param callable match: If provided, an object is only iterated if this callable\n            returns True when called on that object.\n\n        :returns: Iterator over desired objects.\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def get_tabs(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        ' Guess at the tabbing style of this parsed object, based on whitespace.\n\n        If this object is a leaf, it deducts the tabbing based on its own contents.\n        Other objects may guess by calling `get_tabs` recursively on child objects.\n\n        :returns: Guess at tabbing for this object. Should only return whitespace strings\n            that does not contain newlines.\n        :rtype str:\n        '
        raise NotImplementedError()

    @abc.abstractmethod
    def set_tabs(self, tabs: str='    ') -> None:
        if False:
            return 10
        'This tries to set and alter the tabbing of the current object to a desired\n        whitespace string. Primarily meant for objects that were constructed, so they\n        can conform to surrounding whitespace.\n\n        :param str tabs: A whitespace string (not containing newlines).\n        '
        raise NotImplementedError()

    def dump(self, include_spaces: bool=False) -> List[Any]:
        if False:
            i = 10
            return i + 15
        ' Dumps back to pyparsing-like list tree. The opposite of `parse`.\n\n        Note: if this object has not been modified, `dump` with `include_spaces=True`\n        should always return the original input of `parse`.\n\n        :param bool include_spaces: If set to False, magically hides whitespace tokens from\n            dumped output.\n\n        :returns: Pyparsing-like list tree.\n        :rtype list:\n        '
        return [elem.dump(include_spaces) for elem in self._data]

class Statements(Parsable):
    """ A group or list of "Statements". A Statement is either a Block or a Sentence.

    The underlying representation is simply a list of these Statement objects, with
    an extra `_trailing_whitespace` string to keep track of the whitespace that does not
    precede any more statements.
    """

    def __init__(self, parent: Optional[Parsable]=None):
        if False:
            i = 10
            return i + 15
        super().__init__(parent)
        self._trailing_whitespace = None

    @staticmethod
    def should_parse(lists: Any) -> bool:
        if False:
            return 10
        return isinstance(lists, list)

    def set_tabs(self, tabs: str='    ') -> None:
        if False:
            print('Hello World!')
        ' Sets the tabbing for this set of statements. Does this by calling `set_tabs`\n        on each of the child statements.\n\n        Then, if a parent is present, sets trailing whitespace to parent tabbing. This\n        is so that the trailing } of any Block that contains Statements lines up\n        with parent tabbing.\n        '
        for statement in self._data:
            statement.set_tabs(tabs)
        if self.parent is not None:
            self._trailing_whitespace = '\n' + self.parent.get_tabs()

    def parse(self, raw_list: List[Any], add_spaces: bool=False) -> None:
        if False:
            print('Hello World!')
        ' Parses a list of statements.\n        Expects all elements in `raw_list` to be parseable by `type(self).parsing_hooks`,\n        with an optional whitespace string at the last index of `raw_list`.\n        '
        if not isinstance(raw_list, list):
            raise errors.MisconfigurationError('Statements parsing expects a list!')
        if raw_list and isinstance(raw_list[-1], str) and raw_list[-1].isspace():
            self._trailing_whitespace = raw_list[-1]
            raw_list = raw_list[:-1]
        self._data = [parse_raw(elem, self, add_spaces) for elem in raw_list]

    def get_tabs(self) -> str:
        if False:
            i = 10
            return i + 15
        ' Takes a guess at the tabbing of all contained Statements by retrieving the\n        tabbing of the first Statement.'
        if self._data:
            return self._data[0].get_tabs()
        return ''

    def dump(self, include_spaces: bool=False) -> List[Any]:
        if False:
            print('Hello World!')
        ' Dumps this object by first dumping each statement, then appending its\n        trailing whitespace (if `include_spaces` is set) '
        data = super().dump(include_spaces)
        if include_spaces and self._trailing_whitespace is not None:
            return data + [self._trailing_whitespace]
        return data

    def iterate(self, expanded: bool=False, match: Optional[Callable[['Parsable'], bool]]=None) -> Iterator[Any]:
        if False:
            for i in range(10):
                print('nop')
        " Combines each statement's iterator.  "
        for elem in self._data:
            for sub_elem in elem.iterate(expanded, match):
                yield sub_elem

def _space_list(list_: Sequence[Any]) -> List[str]:
    if False:
        while True:
            i = 10
    ' Inserts whitespace between adjacent non-whitespace tokens. '
    spaced_statement: List[str] = []
    for i in reversed(range(len(list_))):
        spaced_statement.insert(0, list_[i])
        if i > 0 and (not list_[i].isspace()) and (not list_[i - 1].isspace()):
            spaced_statement.insert(0, ' ')
    return spaced_statement

class Sentence(Parsable):
    """ A list of words. Non-whitespace words are typically separated with whitespace tokens. """

    @staticmethod
    def should_parse(lists: Any) -> bool:
        if False:
            print('Hello World!')
        ' Returns True if `lists` can be parseable as a `Sentence`-- that is,\n        every element is a string type.\n\n        :param list lists: The raw unparsed list to check.\n\n        :returns: whether this lists is parseable by `Sentence`.\n        '
        return isinstance(lists, list) and len(lists) > 0 and all((isinstance(elem, str) for elem in lists))

    def parse(self, raw_list: List[Any], add_spaces: bool=False) -> None:
        if False:
            print('Hello World!')
        ' Parses a list of string types into this object.\n        If add_spaces is set, adds whitespace tokens between adjacent non-whitespace tokens.'
        if add_spaces:
            raw_list = _space_list(raw_list)
        if not isinstance(raw_list, list) or any((not isinstance(elem, str) for elem in raw_list)):
            raise errors.MisconfigurationError('Sentence parsing expects a list of string types.')
        self._data = raw_list

    def iterate(self, expanded: bool=False, match: Optional[Callable[[Parsable], bool]]=None) -> Iterator[Any]:
        if False:
            while True:
                i = 10
        ' Simply yields itself. '
        if match is None or match(self):
            yield self

    def set_tabs(self, tabs: str='    ') -> None:
        if False:
            i = 10
            return i + 15
        ' Sets the tabbing on this sentence. Inserts a newline and `tabs` at the\n        beginning of `self._data`. '
        if self._data[0].isspace():
            return
        self._data.insert(0, '\n' + tabs)

    def dump(self, include_spaces: bool=False) -> List[Any]:
        if False:
            print('Hello World!')
        ' Dumps this sentence. If include_spaces is set, includes whitespace tokens.'
        if not include_spaces:
            return self.words
        return self._data

    def get_tabs(self) -> str:
        if False:
            i = 10
            return i + 15
        ' Guesses at the tabbing of this sentence. If the first element is whitespace,\n        returns the whitespace after the rightmost newline in the string. '
        first = self._data[0]
        if not first.isspace():
            return ''
        rindex = first.rfind('\n')
        return first[rindex + 1:]

    @property
    def words(self) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        ' Iterates over words, but without spaces. Like Unspaced List. '
        return [word.strip('"\'') for word in self._data if not word.isspace()]

    def __getitem__(self, index: int) -> str:
        if False:
            print('Hello World!')
        return self.words[index]

    def __contains__(self, word: str) -> bool:
        if False:
            print('Hello World!')
        return word in self.words

class Block(Parsable):
    """ Any sort of block, denoted by a block name and curly braces, like so:
    The parsed block:
        block name {
            content 1;
            content 2;
        }
    might be represented with the list [names, contents], where
        names = ["block", " ", "name", " "]
        contents = [["
    ", "content", " ", "1"], ["
    ", "content", " ", "2"], "
"]
    """

    def __init__(self, parent: Optional[Parsable]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent)
        self.names: Optional[Sentence] = None
        self.contents: Optional[Block] = None

    @staticmethod
    def should_parse(lists: Any) -> bool:
        if False:
            i = 10
            return i + 15
        " Returns True if `lists` can be parseable as a `Block`-- that is,\n        it's got a length of 2, the first element is a `Sentence` and the second can be\n        a `Statements`.\n\n        :param list lists: The raw unparsed list to check.\n\n        :returns: whether this lists is parseable by `Block`. "
        return isinstance(lists, list) and len(lists) == 2 and Sentence.should_parse(lists[0]) and isinstance(lists[1], list)

    def set_tabs(self, tabs: str='    ') -> None:
        if False:
            return 10
        ' Sets tabs by setting equivalent tabbing on names, then adding tabbing\n        to contents.'
        self.names.set_tabs(tabs)
        self.contents.set_tabs(tabs + '    ')

    def iterate(self, expanded: bool=False, match: Optional[Callable[[Parsable], bool]]=None) -> Iterator[Any]:
        if False:
            while True:
                i = 10
        ' Iterator over self, and if expanded is set, over its contents. '
        if match is None or match(self):
            yield self
        if expanded:
            for elem in self.contents.iterate(expanded, match):
                yield elem

    def parse(self, raw_list: List[Any], add_spaces: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        " Parses a list that resembles a block.\n\n        The assumptions that this routine makes are:\n            1. the first element of `raw_list` is a valid Sentence.\n            2. the second element of `raw_list` is a valid Statement.\n        If add_spaces is set, we call it recursively on `names` and `contents`, and\n        add an extra trailing space to `names` (to separate the block's opening bracket\n        and the block name).\n        "
        if not Block.should_parse(raw_list):
            raise errors.MisconfigurationError('Block parsing expects a list of length 2. First element should be a list of string types (the block names), and second should be another list of statements (the block content).')
        self.names = Sentence(self)
        if add_spaces:
            raw_list[0].append(' ')
        self.names.parse(raw_list[0], add_spaces)
        self.contents = Statements(self)
        self.contents.parse(raw_list[1], add_spaces)
        self._data = [self.names, self.contents]

    def get_tabs(self) -> str:
        if False:
            return 10
        ' Guesses tabbing by retrieving tabbing guess of self.names. '
        return self.names.get_tabs()

def _is_comment(parsed_obj: Parsable) -> bool:
    if False:
        while True:
            i = 10
    ' Checks whether parsed_obj is a comment.\n\n    :param .Parsable parsed_obj:\n\n    :returns: whether parsed_obj represents a comment sentence.\n    :rtype bool:\n    '
    if not isinstance(parsed_obj, Sentence):
        return False
    return parsed_obj.words[0] == '#'

def _is_certbot_comment(parsed_obj: Parsable) -> bool:
    if False:
        print('Hello World!')
    ' Checks whether parsed_obj is a "managed by Certbot" comment.\n\n    :param .Parsable parsed_obj:\n\n    :returns: whether parsed_obj is a "managed by Certbot" comment.\n    :rtype bool:\n    '
    if not _is_comment(parsed_obj):
        return False
    if len(parsed_obj.words) != len(COMMENT_BLOCK):
        return False
    for (i, word) in enumerate(parsed_obj.words):
        if word != COMMENT_BLOCK[i]:
            return False
    return True

def _certbot_comment(parent: Parsable, preceding_spaces: int=4) -> Sentence:
    if False:
        print('Hello World!')
    ' A "Managed by Certbot" comment.\n    :param int preceding_spaces: Number of spaces between the end of the previous\n        statement and the comment.\n    :returns: Sentence containing the comment.\n    :rtype: .Sentence\n    '
    result = Sentence(parent)
    result.parse([' ' * preceding_spaces] + COMMENT_BLOCK)
    return result

def _choose_parser(parent: Parsable, list_: Any) -> Parsable:
    if False:
        for i in range(10):
            print('nop')
    ' Choose a parser from type(parent).parsing_hooks, depending on whichever hook\n    returns True first. '
    hooks = Parsable.parsing_hooks()
    if parent:
        hooks = type(parent).parsing_hooks()
    for type_ in hooks:
        if type_.should_parse(list_):
            return type_(parent)
    raise errors.MisconfigurationError("None of the parsing hooks succeeded, so we don't know how to parse this set of lists.")

def parse_raw(lists_: Any, parent: Optional[Parsable]=None, add_spaces: bool=False) -> Parsable:
    if False:
        return 10
    " Primary parsing factory function.\n\n    :param list lists_: raw lists from pyparsing to parse.\n    :param .Parent parent: The parent containing this object.\n    :param bool add_spaces: Whether to pass add_spaces to the parser.\n\n    :returns .Parsable: The parsed object.\n\n    :raises errors.MisconfigurationError: If no parsing hook passes, and we can't\n        determine which type to parse the raw lists into.\n    "
    parser = _choose_parser(parent, lists_)
    parser.parse(lists_, add_spaces)
    return parser