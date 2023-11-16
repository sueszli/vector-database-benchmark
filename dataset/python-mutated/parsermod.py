from docutils import nodes
from docutils.parsers import Parser

class Parser(Parser):
    supported = ('foo',)

    def parse(self, input, document):
        if False:
            print('Hello World!')
        section = nodes.section(ids=['id1'])
        section += nodes.title('Generated section', 'Generated section')
        document += section

    def get_transforms(self):
        if False:
            for i in range(10):
                print('nop')
        return []