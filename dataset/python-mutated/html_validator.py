try:
    from html.parser import HTMLParser
except ImportError:
    from HTMLParser import HTMLParser

class SimpleParser(HTMLParser):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        HTMLParser.__init__(self)
        self.elements = []

    def handle_starttag(self, tag, attrs):
        if False:
            i = 10
            return i + 15
        self.elements.append((tag, dict(attrs)))

def assertValidHTML(text):
    if False:
        for i in range(10):
            print('nop')
    h = SimpleParser()
    h.feed(text)
    return True