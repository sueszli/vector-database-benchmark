from io import StringIO
from html.parser import HTMLParser

class MLStripper(HTMLParser):
    """
    Markup Language Stripper
    """

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = StringIO()

    def handle_data(self, d):
        if False:
            while True:
                i = 10
        self.text.write(d)

    def get_data(self):
        if False:
            i = 10
            return i + 15
        return self.text.getvalue()

def strip_tags(html):
    if False:
        while True:
            i = 10
    s = MLStripper()
    s.feed(html)
    return s.get_data()