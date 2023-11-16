from element_1 import ExampleElement
from twisted.web.template import flattenString

def renderDone(output):
    if False:
        return 10
    print(output)
flattenString(None, ExampleElement()).addCallback(renderDone)