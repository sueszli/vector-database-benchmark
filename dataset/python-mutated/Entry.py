import datetime
import logging
import os
import re
from typing import TYPE_CHECKING
from jrnl.color import colorize
from jrnl.color import highlight_tags_with_background_color
from jrnl.output import wrap_with_ansi_colors
if TYPE_CHECKING:
    from .Journal import Journal

class Entry:

    def __init__(self, journal: 'Journal', date: datetime.datetime | None=None, text: str='', starred: bool=False):
        if False:
            i = 10
            return i + 15
        self.journal = journal
        self.date = date or datetime.datetime.now()
        self.text = text
        self._title = None
        self._body = None
        self._tags = None
        self.starred = starred
        self.modified = False

    @property
    def fulltext(self) -> str:
        if False:
            print('Hello World!')
        return self.title + ' ' + self.body

    def _parse_text(self):
        if False:
            while True:
                i = 10
        raw_text = self.text
        lines = raw_text.splitlines()
        if lines and lines[0].strip().endswith('*'):
            self.starred = True
            raw_text = lines[0].strip('\n *') + '\n' + '\n'.join(lines[1:])
        (self._title, self._body) = split_title(raw_text)
        if self._tags is None:
            self._tags = list(self._parse_tags())

    @property
    def title(self) -> str:
        if False:
            i = 10
            return i + 15
        if self._title is None:
            self._parse_text()
        return self._title

    @title.setter
    def title(self, x: str):
        if False:
            return 10
        self._title = x

    @property
    def body(self) -> str:
        if False:
            i = 10
            return i + 15
        if self._body is None:
            self._parse_text()
        return self._body

    @body.setter
    def body(self, x: str):
        if False:
            while True:
                i = 10
        self._body = x

    @property
    def tags(self) -> list[str]:
        if False:
            return 10
        if self._tags is None:
            self._parse_text()
        return self._tags

    @tags.setter
    def tags(self, x: list[str]):
        if False:
            while True:
                i = 10
        self._tags = x

    @staticmethod
    def tag_regex(tagsymbols: str) -> re.Pattern:
        if False:
            while True:
                i = 10
        pattern = f'(?<!\\S)([{tagsymbols}][-+*#/\\w]+)'
        return re.compile(pattern)

    def _parse_tags(self) -> set[str]:
        if False:
            while True:
                i = 10
        tagsymbols = self.journal.config['tagsymbols']
        return {tag.lower() for tag in re.findall(Entry.tag_regex(tagsymbols), self.text)}

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns string representation of the entry to be written to journal file.'
        date_str = self.date.strftime(self.journal.config['timeformat'])
        title = '[{}] {}'.format(date_str, self.title.rstrip('\n '))
        if self.starred:
            title += ' *'
        return '{title}{sep}{body}\n'.format(title=title, sep='\n' if self.body.rstrip('\n ') else '', body=self.body.rstrip('\n '))

    def pprint(self, short: bool=False) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Returns a pretty-printed version of the entry.\n        If short is true, only print the title.'
        if self.journal.config['indent_character']:
            indent = self.journal.config['indent_character'].rstrip() + ' '
        else:
            indent = ''
        date_str = colorize(self.date.strftime(self.journal.config['timeformat']), self.journal.config['colors']['date'], bold=True)
        if not short and self.journal.config['linewrap']:
            columns = self.journal.config['linewrap']
            if columns == 'auto':
                try:
                    columns = os.get_terminal_size().columns
                except OSError:
                    logging.debug("Can't determine terminal size automatically 'linewrap': '%s'", self.journal.config['linewrap'])
                    columns = 79
            title = wrap_with_ansi_colors(date_str + ' ' + highlight_tags_with_background_color(self, self.title, self.journal.config['colors']['title'], is_title=True), columns)
            body = highlight_tags_with_background_color(self, self.body.rstrip(' \n'), self.journal.config['colors']['body'])
            body = wrap_with_ansi_colors(body, columns - len(indent))
            if indent:
                body = '\n'.join((colorize(indent, self.journal.config['colors']['body']) + line for line in body.splitlines()))
            body = colorize(body, self.journal.config['colors']['body'])
        else:
            title = date_str + ' ' + highlight_tags_with_background_color(self, self.title.rstrip('\n'), self.journal.config['colors']['title'], is_title=True)
            body = highlight_tags_with_background_color(self, self.body.rstrip('\n '), self.journal.config['colors']['body'])
        has_body = len(self.body) > 20 or not all((char in (' ', '\n') for char in self.body))
        if short:
            return title
        else:
            return '{title}{sep}{body}\n'.format(title=title, sep='\n' if has_body else '', body=body if has_body else '')

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return "<Entry '{}' on {}>".format(self.title.strip(), self.date.strftime('%Y-%m-%d %H:%M'))

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return hash(self.__repr__())

    def __eq__(self, other: 'Entry'):
        if False:
            return 10
        if not isinstance(other, Entry) or self.title.strip() != other.title.strip() or self.body.rstrip() != other.body.rstrip() or (self.date != other.date) or (self.starred != other.starred):
            return False
        return True

    def __ne__(self, other: 'Entry'):
        if False:
            i = 10
            return i + 15
        return not self.__eq__(other)
SENTENCE_SPLITTER = re.compile('\n    (\n    [.!?\\u2026\\u203C\\u203D\\u2047\\u2048\\u2049\\u22EF\\uFE52\\uFE57] # Sequence starting with a sentence terminal,\n    [\\\'\\u2019\\"\\u201D]? # an optional right quote,\n    [\\]\\)]*             # optional closing bracket\n    \\s+                 # AND a sequence of required spaces.\n    )\n    |[\\uFF01\\uFF0E\\uFF1F\\uFF61\\u3002] # CJK full/half width terminals usually do not have following spaces.\n    ', re.VERBOSE)
SENTENCE_SPLITTER_ONLY_NEWLINE = re.compile('\n')

def split_title(text: str) -> tuple[str, str]:
    if False:
        return 10
    'Splits the first sentence off from a text.'
    sep = SENTENCE_SPLITTER_ONLY_NEWLINE.search(text.lstrip())
    if not sep:
        sep = SENTENCE_SPLITTER.search(text)
        if not sep:
            return (text, '')
    return (text[:sep.end()].strip(), text[sep.end():].strip())