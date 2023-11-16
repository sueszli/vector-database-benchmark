from twisted.python.filepath import FilePath
from twisted.web.template import Element, XMLFile, flattenString, renderer

class WidgetsElement(Element):
    loader = XMLFile(FilePath('iteration-1.xml'))
    widgetData = ['gadget', 'contraption', 'gizmo', 'doohickey']

    @renderer
    def widgets(self, request, tag):
        if False:
            return 10
        for widget in self.widgetData:
            yield tag.clone().fillSlots(widgetName=widget)

def printResult(result):
    if False:
        i = 10
        return i + 15
    print(result)
flattenString(None, WidgetsElement()).addCallback(printResult)