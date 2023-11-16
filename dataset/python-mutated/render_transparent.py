from transparent_element import ExampleElement
from twisted.web.template import flattenString

def renderDone(output):
    if False:
        print('Hello World!')
    print(output)
flattenString(None, ExampleElement()).addCallback(renderDone)