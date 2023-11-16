from docutils.parsers import Parser

class DummyMarkdownParser(Parser):
    supported = ('markdown',)

    def parse(self, inputstring, document):
        if False:
            return 10
        document.rawsource = inputstring

def setup(app):
    if False:
        while True:
            i = 10
    app.add_source_suffix('.md', 'markdown')
    app.add_source_parser(DummyMarkdownParser)