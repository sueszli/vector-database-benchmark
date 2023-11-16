"""
Module responsible for reading code based on position/range.
"""
import dataclasses
from typing import ClassVar, Optional
from ..language_server import protocol as lsp

@dataclasses.dataclass(frozen=True)
class SourceCodeContext:
    MAX_LINES_BEFORE_OR_AFTER: ClassVar[int] = 2500

    @staticmethod
    def from_source_and_position(source: str, position: lsp.LspPosition, max_lines_before_or_after: int=MAX_LINES_BEFORE_OR_AFTER) -> Optional[str]:
        if False:
            print('Hello World!')
        lines = source.splitlines()
        line_number = position.line
        if line_number >= len(lines):
            return None
        full_document_contents = source.splitlines()
        lower_line_number = max(position.line - max_lines_before_or_after, 0)
        higher_line_number = min(position.line + max_lines_before_or_after + 1, len(full_document_contents))
        return '\n'.join(full_document_contents[lower_line_number:higher_line_number])

    @staticmethod
    def character_at_position(source: str, position: lsp.LspPosition) -> Optional[str]:
        if False:
            print('Hello World!')
        lines = source.splitlines()
        if position.line >= len(lines) or position.line < 0 or position.character < 0 or (position.character >= len(lines[position.line])):
            return None
        return lines[position.line][position.character]

    @staticmethod
    def text_at_range(source: str, text_range: lsp.LspRange) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        start = text_range.start
        end = text_range.end
        lines = source.splitlines()
        if start.line >= len(lines) or start.line < 0 or start.character < 0 or (start.character >= len(lines[start.line])):
            return None
        if end.line >= len(lines) or end.line < 0 or end.character < 0 or (end.character > len(lines[end.line])):
            return None
        if start.line > end.line or (start.line == end.line and start.character > end.character):
            return None
        if start.line == end.line:
            return lines[start.line][start.character:end.character]
        result = ''
        result += lines[start.line][start.character:]
        for line_num in range(start.line + 1, end.line):
            result += '\n' + lines[line_num]
        result += '\n' + lines[end.line][:end.character]
        return result