from xml.sax import handler, make_parser

class categoryHandler(handler.ContentHandler):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.document = None
        self.in_importants = False

    def startElement(self, name, attrs):
        if False:
            print('Hello World!')
        if name == 'document':
            self.document = Document(attrs)
        if name == 'category':
            self.document.categories.append(Category(attrs))
        elif name == 'overviews':
            category = self.document.categories[-1]
            assert category.overviewItems is None, f'category {category!r} already has overviews'
            category.overviewItems = OverviewItems(attrs)
        elif name == 'item':
            item = Item(attrs)
            if self.in_importants:
                self.document.important.append(item)
            elif self.document.categories:
                category = self.document.categories[-1]
                category.overviewItems.items.append(item)
            else:
                self.document.links.append(item)
        elif name == 'important':
            self.in_importants = True

    def endElement(self, name):
        if False:
            i = 10
            return i + 15
        if name == 'important':
            self.in_importants = False

    def endDocument(self):
        if False:
            i = 10
            return i + 15
        pass

class Document:

    def __init__(self, attrs):
        if False:
            return 10
        self.__dict__.update(attrs)
        self.categories = []
        self.links = []
        self.important = []

    def __iter__(self):
        if False:
            while True:
                i = 10
        return iter(self.categories)

class Category:

    def __init__(self, attrs):
        if False:
            for i in range(10):
                print('nop')
        self.__dict__.update(attrs)
        self.overviewItems = None

class OverviewItems:

    def __init__(self, attrs):
        if False:
            return 10
        self.__dict__.update(attrs)
        self.items = []

    def __iter__(self):
        if False:
            print('Hello World!')
        return iter(self.items)

class Item:

    def __init__(self, attrs):
        if False:
            for i in range(10):
                print('nop')
        self.__dict__.update(attrs)

def GetDocument(fname='pywin32-document.xml'):
    if False:
        i = 10
        return i + 15
    parser = make_parser()
    handler = categoryHandler()
    parser.setContentHandler(handler)
    parser.parse(fname)
    return handler.document
if __name__ == '__main__':
    doc = GetDocument()
    print('Important Notes')
    for link in doc.important:
        print(' ', link.name, link.href)
    print('Doc links')
    for link in doc.links:
        print(' ', link.name, link.href)
    print('Doc categories')
    for c in doc:
        print(' ', c.id, c.label)