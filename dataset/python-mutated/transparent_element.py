from twisted.python.filepath import FilePath
from twisted.web.template import Element, XMLFile, renderer

class ExampleElement(Element):
    loader = XMLFile(FilePath('transparent-1.xml'))

    @renderer
    def renderer1(self, request, tag):
        if False:
            print('Hello World!')
        return tag('hello')

    @renderer
    def renderer2(self, request, tag):
        if False:
            i = 10
            return i + 15
        return tag('world')