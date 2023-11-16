class XMLDoc:

    def __init__(self, version):
        if False:
            while True:
                i = 10
        self.version = version
        self.root_element = None

    def saveFile(self, filename):
        if False:
            print('Hello World!')
        f = file(filename, 'w')
        f.write('<?xml version="' + self.version + '"?>\n')
        self.root_element._write(f, 0)

    def saveFormatFile(self, filename, fmt):
        if False:
            return 10
        self.saveFile(filename)

    def freeDoc(self):
        if False:
            print('Hello World!')
        pass

class XMLNode:

    def __init__(self, name):
        if False:
            i = 10
            return i + 15
        self.name = name
        self.props = []
        self.children = []
        self.content = None

    def docSetRootElement(self, doc):
        if False:
            for i in range(10):
                print('nop')
        doc.root_element = self

    def newChild(self, namespace, name, content):
        if False:
            print('Hello World!')
        if namespace:
            fullname = namespace + ':' + name
        else:
            fullname = name
        child = XMLNode(fullname)
        child.content = content
        self.children.append(child)
        return child

    def setProp(self, name, value):
        if False:
            for i in range(10):
                print('nop')
        self.props.append((name, value))

    def _write(self, f, indent):
        if False:
            while True:
                i = 10
        istr = '\t' * indent
        tag = self.name
        for prop in self.props:
            (name, value) = prop
            tag += ' ' + name + '="' + value + '"'
        if self.children:
            f.write(istr + '<%s>\n' % tag)
            for child in self.children:
                child._write(f, indent + 1)
            f.write(istr + '</%s>\n' % self.name)
        else:
            f.write(istr + '<%s/>\n' % tag)

def newDoc(version):
    if False:
        print('Hello World!')
    return XMLDoc(version)

def newNode(name):
    if False:
        return 10
    return XMLNode(name)