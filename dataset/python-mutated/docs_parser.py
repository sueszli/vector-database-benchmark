"""Docs parser.

Contains parsers for docx, pdf files.

"""
from pathlib import Path
from typing import Dict
from application.parser.file.base_parser import BaseParser

class PDFParser(BaseParser):
    """PDF parser."""

    def _init_parser(self) -> Dict:
        if False:
            for i in range(10):
                print('nop')
        'Init parser.'
        return {}

    def parse_file(self, file: Path, errors: str='ignore') -> str:
        if False:
            while True:
                i = 10
        'Parse file.'
        try:
            import PyPDF2
        except ImportError:
            raise ValueError('PyPDF2 is required to read PDF files.')
        text_list = []
        with open(file, 'rb') as fp:
            pdf = PyPDF2.PdfReader(fp)
            num_pages = len(pdf.pages)
            for page in range(num_pages):
                page_text = pdf.pages[page].extract_text()
                text_list.append(page_text)
        text = '\n'.join(text_list)
        return text

class DocxParser(BaseParser):
    """Docx parser."""

    def _init_parser(self) -> Dict:
        if False:
            return 10
        'Init parser.'
        return {}

    def parse_file(self, file: Path, errors: str='ignore') -> str:
        if False:
            while True:
                i = 10
        'Parse file.'
        try:
            import docx2txt
        except ImportError:
            raise ValueError('docx2txt is required to read Microsoft Word files.')
        text = docx2txt.process(file)
        return text