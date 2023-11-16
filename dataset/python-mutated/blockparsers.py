from abc import ABC, abstractmethod
from ..lexer import Token
from ..model import Block, Container, End, For, If, Keyword, NestedBlock, Statement, TestCase, Try, While

class Parser(ABC):
    model: Container

    def __init__(self, model: Container):
        if False:
            print('Hello World!')
        self.model = model

    @abstractmethod
    def handles(self, statement: Statement) -> bool:
        if False:
            return 10
        raise NotImplementedError

    @abstractmethod
    def parse(self, statement: Statement) -> 'Parser|None':
        if False:
            print('Hello World!')
        raise NotImplementedError

class BlockParser(Parser, ABC):
    model: Block
    unhandled_tokens = Token.HEADER_TOKENS | frozenset((Token.TESTCASE_NAME, Token.KEYWORD_NAME))

    def __init__(self, model: Block):
        if False:
            while True:
                i = 10
        super().__init__(model)
        self.parsers: 'dict[str, type[NestedBlockParser]]' = {Token.FOR: ForParser, Token.IF: IfParser, Token.INLINE_IF: IfParser, Token.TRY: TryParser, Token.WHILE: WhileParser}

    def handles(self, statement: Statement) -> bool:
        if False:
            print('Hello World!')
        return statement.type not in self.unhandled_tokens

    def parse(self, statement: Statement) -> 'BlockParser|None':
        if False:
            return 10
        parser_class = self.parsers.get(statement.type)
        if parser_class:
            model_class = parser_class.__annotations__['model']
            parser = parser_class(model_class(statement))
            self.model.body.append(parser.model)
            return parser
        self.model.body.append(statement)
        return None

class TestCaseParser(BlockParser):
    model: TestCase

class KeywordParser(BlockParser):
    model: Keyword

class NestedBlockParser(BlockParser, ABC):
    model: NestedBlock

    def __init__(self, model: NestedBlock, handle_end: bool=True):
        if False:
            return 10
        super().__init__(model)
        self.handle_end = handle_end

    def handles(self, statement: Statement) -> bool:
        if False:
            print('Hello World!')
        if self.model.end:
            return False
        if statement.type == Token.END:
            return self.handle_end
        return super().handles(statement)

    def parse(self, statement: Statement) -> 'BlockParser|None':
        if False:
            while True:
                i = 10
        if isinstance(statement, End):
            self.model.end = statement
            return None
        return super().parse(statement)

class ForParser(NestedBlockParser):
    model: For

class WhileParser(NestedBlockParser):
    model: While

class IfParser(NestedBlockParser):
    model: If

    def parse(self, statement: Statement) -> 'BlockParser|None':
        if False:
            for i in range(10):
                print('nop')
        if statement.type in (Token.ELSE_IF, Token.ELSE):
            parser = IfParser(If(statement), handle_end=False)
            self.model.orelse = parser.model
            return parser
        return super().parse(statement)

class TryParser(NestedBlockParser):
    model: Try

    def parse(self, statement) -> 'BlockParser|None':
        if False:
            while True:
                i = 10
        if statement.type in (Token.EXCEPT, Token.ELSE, Token.FINALLY):
            parser = TryParser(Try(statement), handle_end=False)
            self.model.next = parser.model
            return parser
        return super().parse(statement)