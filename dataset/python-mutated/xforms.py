from .namespaces import XFORMSNS
from .element import Element

def Model(**args):
    if False:
        return 10
    return Element(qname=(XFORMSNS, 'model'), **args)

def Instance(**args):
    if False:
        i = 10
        return i + 15
    return Element(qname=(XFORMSNS, 'instance'), **args)

def Bind(**args):
    if False:
        while True:
            i = 10
    return Element(qname=(XFORMSNS, 'bind'), **args)