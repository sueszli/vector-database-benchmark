import docx2txt
from .utils import BaseParser

class Parser(BaseParser):
    """Extract text from docx file using python-docx.
    """

    def extract(self, filename, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return docx2txt.process(filename)