from enum import auto, Enum
from dataclasses import dataclass
from .typeinfo import TypeInfo

class TypeInfoTokenType(Enum):
    NAME = auto()
    LEFT_SQUARE = auto()
    RIGHT_SQUARE = auto()
    PIPE = auto()
    COMMA = auto()

    def __repr__(self):
        if False:
            while True:
                i = 10
        return str(self)

@dataclass
class TypeInfoToken:
    type: TypeInfoTokenType
    value: str
    position: int = -1

class TypeInfoTokenizer:
    markers = {'[': TypeInfoTokenType.LEFT_SQUARE, ']': TypeInfoTokenType.RIGHT_SQUARE, '|': TypeInfoTokenType.PIPE, ',': TypeInfoTokenType.COMMA}

    def __init__(self, source: str):
        if False:
            i = 10
            return i + 15
        self.source = source
        self.tokens: 'list[TypeInfoToken]' = []
        self.start = 0
        self.current = 0

    @property
    def at_end(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self.current >= len(self.source)

    def tokenize(self) -> 'list[TypeInfoToken]':
        if False:
            return 10
        while not self.at_end:
            self.start = self.current
            char = self.advance()
            if char in self.markers:
                self.add_token(self.markers[char])
            elif char.strip():
                self.name()
        return self.tokens

    def advance(self) -> str:
        if False:
            print('Hello World!')
        char = self.source[self.current]
        self.current += 1
        return char

    def peek(self) -> 'str|None':
        if False:
            return 10
        try:
            return self.source[self.current]
        except IndexError:
            return None

    def name(self):
        if False:
            for i in range(10):
                print('nop')
        end_at = set(self.markers) | {None}
        while self.peek() not in end_at:
            self.current += 1
        self.add_token(TypeInfoTokenType.NAME)

    def add_token(self, type: TypeInfoTokenType):
        if False:
            i = 10
            return i + 15
        value = self.source[self.start:self.current].strip()
        self.tokens.append(TypeInfoToken(type, value, self.start))

class TypeInfoParser:

    def __init__(self, source: str):
        if False:
            print('Hello World!')
        self.source = source
        self.tokens: 'list[TypeInfoToken]' = []
        self.current = 0

    @property
    def at_end(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self.peek() is None

    def parse(self) -> TypeInfo:
        if False:
            return 10
        self.tokens = TypeInfoTokenizer(self.source).tokenize()
        info = self.type()
        if not self.at_end:
            self.error(f"Extra content after '{info}'.")
        return info

    def type(self) -> TypeInfo:
        if False:
            return 10
        if not self.check(TypeInfoTokenType.NAME):
            self.error('Type name missing.')
        info = TypeInfo(self.advance().value)
        if self.match(TypeInfoTokenType.LEFT_SQUARE):
            info.nested = self.params()
        if self.match(TypeInfoTokenType.PIPE):
            nested = [info] + self.union()
            info = TypeInfo('Union', nested=nested)
        return info

    def params(self) -> 'list[TypeInfo]':
        if False:
            i = 10
            return i + 15
        params = []
        while not params or self.match(TypeInfoTokenType.COMMA):
            params.append(self.type())
        if not self.match(TypeInfoTokenType.RIGHT_SQUARE):
            self.error("Closing ']' missing.")
        return params

    def union(self) -> 'list[TypeInfo]':
        if False:
            i = 10
            return i + 15
        types = []
        while not types or self.match(TypeInfoTokenType.PIPE):
            info = self.type()
            if info.is_union:
                types.extend(info.nested)
            else:
                types.append(info)
        return types

    def match(self, *types: TypeInfoTokenType) -> bool:
        if False:
            while True:
                i = 10
        for typ in types:
            if self.check(typ):
                self.advance()
                return True
        return False

    def check(self, expected: TypeInfoTokenType) -> bool:
        if False:
            print('Hello World!')
        peeked = self.peek()
        return peeked and peeked.type == expected

    def advance(self) -> 'TypeInfoToken|None':
        if False:
            for i in range(10):
                print('nop')
        token = self.peek()
        if token:
            self.current += 1
        return token

    def peek(self) -> 'TypeInfoToken|None':
        if False:
            print('Hello World!')
        try:
            return self.tokens[self.current]
        except IndexError:
            return None

    def error(self, message: str):
        if False:
            i = 10
            return i + 15
        token = self.peek()
        position = f'index {token.position}' if token else 'end'
        raise ValueError(f'Parsing type {self.source!r} failed: Error at {position}: {message}')