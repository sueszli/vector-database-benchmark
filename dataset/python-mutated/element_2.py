from twisted.python.filepath import FilePath
from twisted.web.template import Element, XMLFile, renderer, tags

class ExampleElement(Element):
    loader = XMLFile(FilePath('template-1.xml'))

    @renderer
    def header(self, request, tag):
        if False:
            return 10
        return tag(tags.b('Header.'))

    @renderer
    def footer(self, request, tag):
        if False:
            i = 10
            return i + 15
        return tag(tags.b('Footer.'))