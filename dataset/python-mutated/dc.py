from .namespaces import DCNS
from .element import Element

def Creator(**args):
    if False:
        print('Hello World!')
    return Element(qname=(DCNS, 'creator'), **args)

def Date(**args):
    if False:
        for i in range(10):
            print('nop')
    return Element(qname=(DCNS, 'date'), **args)

def Description(**args):
    if False:
        while True:
            i = 10
    return Element(qname=(DCNS, 'description'), **args)

def Language(**args):
    if False:
        while True:
            i = 10
    return Element(qname=(DCNS, 'language'), **args)

def Subject(**args):
    if False:
        for i in range(10):
            print('nop')
    return Element(qname=(DCNS, 'subject'), **args)

def Title(**args):
    if False:
        for i in range(10):
            print('nop')
    return Element(qname=(DCNS, 'title'), **args)