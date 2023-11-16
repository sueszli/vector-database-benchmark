from docutils.parsers import Parser

class TestSourceParser(Parser):
    supported = ('test',)

def setup(app):
    if False:
        print('Hello World!')
    app.add_source_suffix('.test', 'test')
    app.add_source_parser(TestSourceParser)