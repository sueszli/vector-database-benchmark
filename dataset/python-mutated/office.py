from .namespaces import OFFICENS
from .element import Element
from .draw import StyleRefElement

def Annotation(**args):
    if False:
        while True:
            i = 10
    return StyleRefElement(qname=(OFFICENS, 'annotation'), **args)

def AutomaticStyles(**args):
    if False:
        while True:
            i = 10
    return Element(qname=(OFFICENS, 'automatic-styles'), **args)

def BinaryData(**args):
    if False:
        i = 10
        return i + 15
    return Element(qname=(OFFICENS, 'binary-data'), **args)

def Body(**args):
    if False:
        while True:
            i = 10
    return Element(qname=(OFFICENS, 'body'), **args)

def ChangeInfo(**args):
    if False:
        return 10
    return Element(qname=(OFFICENS, 'change-info'), **args)

def Chart(**args):
    if False:
        for i in range(10):
            print('nop')
    return Element(qname=(OFFICENS, 'chart'), **args)

def DdeSource(**args):
    if False:
        print('Hello World!')
    return Element(qname=(OFFICENS, 'dde-source'), **args)

def Document(version='1.1', **args):
    if False:
        print('Hello World!')
    return Element(qname=(OFFICENS, 'document'), version=version, **args)

def DocumentContent(version='1.1', **args):
    if False:
        return 10
    return Element(qname=(OFFICENS, 'document-content'), version=version, **args)

def DocumentMeta(version='1.1', **args):
    if False:
        while True:
            i = 10
    return Element(qname=(OFFICENS, 'document-meta'), version=version, **args)

def DocumentSettings(version='1.1', **args):
    if False:
        i = 10
        return i + 15
    return Element(qname=(OFFICENS, 'document-settings'), version=version, **args)

def DocumentStyles(version='1.1', **args):
    if False:
        for i in range(10):
            print('nop')
    return Element(qname=(OFFICENS, 'document-styles'), version=version, **args)

def Drawing(**args):
    if False:
        print('Hello World!')
    return Element(qname=(OFFICENS, 'drawing'), **args)

def EventListeners(**args):
    if False:
        print('Hello World!')
    return Element(qname=(OFFICENS, 'event-listeners'), **args)

def FontFaceDecls(**args):
    if False:
        for i in range(10):
            print('nop')
    return Element(qname=(OFFICENS, 'font-face-decls'), **args)

def Forms(**args):
    if False:
        print('Hello World!')
    return Element(qname=(OFFICENS, 'forms'), **args)

def Image(**args):
    if False:
        while True:
            i = 10
    return Element(qname=(OFFICENS, 'image'), **args)

def MasterStyles(**args):
    if False:
        while True:
            i = 10
    return Element(qname=(OFFICENS, 'master-styles'), **args)

def Meta(**args):
    if False:
        i = 10
        return i + 15
    return Element(qname=(OFFICENS, 'meta'), **args)

def Presentation(**args):
    if False:
        i = 10
        return i + 15
    return Element(qname=(OFFICENS, 'presentation'), **args)

def Script(**args):
    if False:
        print('Hello World!')
    return Element(qname=(OFFICENS, 'script'), **args)

def Scripts(**args):
    if False:
        print('Hello World!')
    return Element(qname=(OFFICENS, 'scripts'), **args)

def Settings(**args):
    if False:
        for i in range(10):
            print('nop')
    return Element(qname=(OFFICENS, 'settings'), **args)

def Spreadsheet(**args):
    if False:
        print('Hello World!')
    return Element(qname=(OFFICENS, 'spreadsheet'), **args)

def Styles(**args):
    if False:
        return 10
    return Element(qname=(OFFICENS, 'styles'), **args)

def Text(**args):
    if False:
        while True:
            i = 10
    return Element(qname=(OFFICENS, 'text'), **args)