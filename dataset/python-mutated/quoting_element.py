from twisted.python.filepath import FilePath
from twisted.web.template import Element, XMLFile, renderer

class ExampleElement(Element):
    loader = XMLFile(FilePath('template-1.xml'))

    @renderer
    def header(self, request, tag):
        if False:
            i = 10
            return i + 15
        return tag('<<<Header>>>!')

    @renderer
    def footer(self, request, tag):
        if False:
            while True:
                i = 10
        return tag('>>>"Footer!"<<<', id='<"fun">')