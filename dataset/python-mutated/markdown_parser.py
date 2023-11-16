"""Markdown parser.

Contains parser for md files.

"""
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast
import tiktoken
from application.parser.file.base_parser import BaseParser

class MarkdownParser(BaseParser):
    """Markdown parser.

    Extract text from markdown files.
    Returns dictionary with keys as headers and values as the text between headers.

    """

    def __init__(self, *args: Any, remove_hyperlinks: bool=True, remove_images: bool=True, max_tokens: int=2048, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        'Init params.'
        super().__init__(*args, **kwargs)
        self._remove_hyperlinks = remove_hyperlinks
        self._remove_images = remove_images
        self._max_tokens = max_tokens

    def tups_chunk_append(self, tups: List[Tuple[Optional[str], str]], current_header: Optional[str], current_text: str):
        if False:
            while True:
                i = 10
        'Append to tups chunk.'
        num_tokens = len(tiktoken.get_encoding('cl100k_base').encode(current_text))
        if num_tokens > self._max_tokens:
            chunks = [current_text[i:i + self._max_tokens] for i in range(0, len(current_text), self._max_tokens)]
            for chunk in chunks:
                tups.append((current_header, chunk))
        else:
            tups.append((current_header, current_text))
        return tups

    def markdown_to_tups(self, markdown_text: str) -> List[Tuple[Optional[str], str]]:
        if False:
            return 10
        'Convert a markdown file to a dictionary.\n\n        The keys are the headers and the values are the text under each header.\n\n        '
        markdown_tups: List[Tuple[Optional[str], str]] = []
        lines = markdown_text.split('\n')
        current_header = None
        current_text = ''
        for line in lines:
            header_match = re.match('^#+\\s', line)
            if header_match:
                if current_header is not None:
                    if current_text == '' or None:
                        continue
                    markdown_tups = self.tups_chunk_append(markdown_tups, current_header, current_text)
                current_header = line
                current_text = ''
            else:
                current_text += line + '\n'
        markdown_tups = self.tups_chunk_append(markdown_tups, current_header, current_text)
        if current_header is not None:
            markdown_tups = [(re.sub('#', '', cast(str, key)).strip(), re.sub('<.*?>', '', value)) for (key, value) in markdown_tups]
        else:
            markdown_tups = [(key, re.sub('\n', '', value)) for (key, value) in markdown_tups]
        return markdown_tups

    def remove_images(self, content: str) -> str:
        if False:
            return 10
        'Get a dictionary of a markdown file from its path.'
        pattern = '!{1}\\[\\[(.*)\\]\\]'
        content = re.sub(pattern, '', content)
        return content

    def remove_hyperlinks(self, content: str) -> str:
        if False:
            i = 10
            return i + 15
        'Get a dictionary of a markdown file from its path.'
        pattern = '\\[(.*?)\\]\\((.*?)\\)'
        content = re.sub(pattern, '\\1', content)
        return content

    def _init_parser(self) -> Dict:
        if False:
            i = 10
            return i + 15
        'Initialize the parser with the config.'
        return {}

    def parse_tups(self, filepath: Path, errors: str='ignore') -> List[Tuple[Optional[str], str]]:
        if False:
            for i in range(10):
                print('nop')
        'Parse file into tuples.'
        with open(filepath, 'r') as f:
            content = f.read()
        if self._remove_hyperlinks:
            content = self.remove_hyperlinks(content)
        if self._remove_images:
            content = self.remove_images(content)
        markdown_tups = self.markdown_to_tups(content)
        return markdown_tups

    def parse_file(self, filepath: Path, errors: str='ignore') -> Union[str, List[str]]:
        if False:
            while True:
                i = 10
        'Parse file into string.'
        tups = self.parse_tups(filepath, errors=errors)
        results = []
        for (header, value) in tups:
            if header is None:
                results.append(value)
            else:
                results.append(f'\n\n{header}\n{value}')
        return results