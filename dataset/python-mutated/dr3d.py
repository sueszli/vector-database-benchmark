from .namespaces import DR3DNS
from .draw import StyleRefElement

def Cube(**args):
    if False:
        while True:
            i = 10
    return StyleRefElement(qname=(DR3DNS, 'cube'), **args)

def Extrude(**args):
    if False:
        while True:
            i = 10
    return StyleRefElement(qname=(DR3DNS, 'extrude'), **args)

def Light(Element, **args):
    if False:
        print('Hello World!')
    return StyleRefElement(qname=(DR3DNS, 'light'), **args)

def Rotate(**args):
    if False:
        i = 10
        return i + 15
    return StyleRefElement(qname=(DR3DNS, 'rotate'), **args)

def Scene(**args):
    if False:
        return 10
    return StyleRefElement(qname=(DR3DNS, 'scene'), **args)

def Sphere(**args):
    if False:
        for i in range(10):
            print('nop')
    return StyleRefElement(qname=(DR3DNS, 'sphere'), **args)