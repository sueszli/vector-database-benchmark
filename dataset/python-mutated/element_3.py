from twisted.python.filepath import FilePath
from twisted.web.template import Element, XMLFile, renderer, tags

class ExampleElement(Element):
    loader = XMLFile(FilePath('template-1.xml'))

    @renderer
    def header(self, request, tag):
        if False:
            for i in range(10):
                print('nop')
        return tag(tags.p('Header.'), id='header')

    @renderer
    def footer(self, request, tag):
        if False:
            while True:
                i = 10
        return tag(tags.p('Footer.'), id='footer')