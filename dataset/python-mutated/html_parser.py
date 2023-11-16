"""HTML parser.

Contains parser for html files.

"""
import re
from pathlib import Path
from typing import Dict, Union
from application.parser.file.base_parser import BaseParser

class HTMLParser(BaseParser):
    """HTML parser."""

    def _init_parser(self) -> Dict:
        if False:
            for i in range(10):
                print('nop')
        'Init parser.'
        return {}

    def parse_file(self, file: Path, errors: str='ignore') -> Union[str, list[str]]:
        if False:
            i = 10
            return i + 15
        'Parse file.\n\n            Returns:\n            Union[str, List[str]]: a string or a List of strings.\n        '
        try:
            from unstructured.partition.html import partition_html
            from unstructured.staging.base import convert_to_isd
            from unstructured.cleaners.core import clean
        except ImportError:
            raise ValueError('unstructured package is required to parse HTML files.')
        with open(file, 'r', encoding='utf-8') as fp:
            elements = partition_html(file=fp)
            isd = convert_to_isd(elements)
        for isd_el in isd:
            isd_el['text'] = isd_el['text'].encode('ascii', 'ignore').decode()
        for isd_el in isd:
            isd_el['text'] = re.sub('\\n', ' ', isd_el['text'], flags=re.MULTILINE | re.DOTALL)
            isd_el['text'] = re.sub('\\s{2,}', ' ', isd_el['text'], flags=re.MULTILINE | re.DOTALL)
        for isd_el in isd:
            clean(isd_el['text'], extra_whitespace=True, dashes=True, bullets=True, trailing_punctuation=True)
        title_indexes = [i for (i, isd_el) in enumerate(isd) if isd_el['type'] == 'Title']
        Chunks = [[]]
        final_chunks = list(list())
        for (i, isd_el) in enumerate(isd):
            if i in title_indexes:
                Chunks.append([])
            Chunks[-1].append(isd_el['text'])
        for chunk in Chunks:
            sum = 0
            sum += len(str(chunk))
            if sum < 25:
                Chunks.remove(chunk)
            else:
                final_chunks.append(' '.join([str(item) for item in chunk]))
        return final_chunks