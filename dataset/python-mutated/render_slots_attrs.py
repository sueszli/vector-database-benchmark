from slots_attributes_1 import ExampleElement
from twisted.web.template import flattenString

def renderDone(output):
    if False:
        i = 10
        return i + 15
    print(output)
flattenString(None, ExampleElement()).addCallback(renderDone)