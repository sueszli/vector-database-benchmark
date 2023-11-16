from pathlib import Path
from robot.utils import Source
from ..lexer import Token
from ..model import CommentSection, File, ImplicitCommentSection, InvalidSection, Keyword, KeywordSection, Section, SettingSection, Statement, TestCase, TestCaseSection, VariableSection
from .blockparsers import KeywordParser, Parser, TestCaseParser

class FileParser(Parser):
    model: File

    def __init__(self, source: 'Source|None'=None):
        if False:
            return 10
        super().__init__(File(source=self._get_path(source)))
        self.parsers: 'dict[str, type[SectionParser]]' = {Token.SETTING_HEADER: SettingSectionParser, Token.VARIABLE_HEADER: VariableSectionParser, Token.TESTCASE_HEADER: TestCaseSectionParser, Token.TASK_HEADER: TestCaseSectionParser, Token.KEYWORD_HEADER: KeywordSectionParser, Token.COMMENT_HEADER: CommentSectionParser, Token.INVALID_HEADER: InvalidSectionParser, Token.CONFIG: ImplicitCommentSectionParser, Token.COMMENT: ImplicitCommentSectionParser, Token.ERROR: ImplicitCommentSectionParser, Token.EOL: ImplicitCommentSectionParser}

    def _get_path(self, source: 'Source|None') -> 'Path|None':
        if False:
            for i in range(10):
                print('nop')
        if not source:
            return None
        if isinstance(source, str) and '\n' not in source:
            source = Path(source)
        try:
            if isinstance(source, Path) and source.is_file():
                return source
        except OSError:
            pass
        return None

    def handles(self, statement: Statement) -> bool:
        if False:
            print('Hello World!')
        return True

    def parse(self, statement: Statement) -> 'SectionParser':
        if False:
            for i in range(10):
                print('nop')
        parser_class = self.parsers[statement.type]
        model_class: 'type[Section]' = parser_class.__annotations__['model']
        parser = parser_class(model_class(statement))
        self.model.sections.append(parser.model)
        return parser

class SectionParser(Parser):
    model: Section

    def handles(self, statement: Statement) -> bool:
        if False:
            i = 10
            return i + 15
        return statement.type not in Token.HEADER_TOKENS

    def parse(self, statement: Statement) -> 'Parser|None':
        if False:
            for i in range(10):
                print('nop')
        self.model.body.append(statement)
        return None

class SettingSectionParser(SectionParser):
    model: SettingSection

class VariableSectionParser(SectionParser):
    model: VariableSection

class CommentSectionParser(SectionParser):
    model: CommentSection

class ImplicitCommentSectionParser(SectionParser):
    model: ImplicitCommentSection

class InvalidSectionParser(SectionParser):
    model: InvalidSection

class TestCaseSectionParser(SectionParser):
    model: TestCaseSection

    def parse(self, statement: Statement) -> 'Parser|None':
        if False:
            for i in range(10):
                print('nop')
        if statement.type == Token.TESTCASE_NAME:
            parser = TestCaseParser(TestCase(statement))
            self.model.body.append(parser.model)
            return parser
        return super().parse(statement)

class KeywordSectionParser(SectionParser):
    model: KeywordSection

    def parse(self, statement: Statement) -> 'Parser|None':
        if False:
            while True:
                i = 10
        if statement.type == Token.KEYWORD_NAME:
            parser = KeywordParser(Keyword(statement))
            self.model.body.append(parser.model)
            return parser
        return super().parse(statement)