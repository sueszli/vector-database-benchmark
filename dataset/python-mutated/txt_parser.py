from .utils import BaseParser

class Parser(BaseParser):
    """Parse ``.txt`` files"""

    def extract(self, filename, **kwargs):
        if False:
            while True:
                i = 10
        with open(filename) as stream:
            return stream.read()