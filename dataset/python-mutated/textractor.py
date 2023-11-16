"""
Textractor module
"""
import contextlib
import os
from subprocess import Popen
from urllib.request import urlopen
try:
    from bs4 import BeautifulSoup
    from tika import parser
    TIKA = True
except ImportError:
    TIKA = False
from .segmentation import Segmentation

class Textractor(Segmentation):
    """
    Extracts text from files.
    """

    def __init__(self, sentences=False, lines=False, paragraphs=False, minlength=None, join=False, tika=True):
        if False:
            i = 10
            return i + 15
        if not TIKA:
            raise ImportError('Textractor pipeline is not available - install "pipeline" extra to enable')
        super().__init__(sentences, lines, paragraphs, minlength, join)
        self.tika = self.checkjava() if tika else False

    def text(self, text):
        if False:
            for i in range(10):
                print('nop')
        if self.tika:
            text = text.replace('file://', '')
            parsed = parser.from_file(text)
            return parsed['content']
        text = f'file://{text}' if os.path.exists(text) else text
        with contextlib.closing(urlopen(text)) as connection:
            text = connection.read()
        soup = BeautifulSoup(text, features='html.parser')
        return soup.get_text()

    def checkjava(self, path=None):
        if False:
            print('Hello World!')
        '\n        Checks if a Java executable is available for Tika.\n\n        Args:\n            path: path to java executable\n\n        Returns:\n            True if Java is available, False otherwise\n        '
        if not path:
            path = os.getenv('TIKA_JAVA', 'java')
        try:
            _ = Popen(path, stdout=open(os.devnull, 'w'), stderr=open(os.devnull, 'w'))
        except:
            return False
        return True