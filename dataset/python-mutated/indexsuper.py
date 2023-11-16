import sys
from xml.dom import minidom
from xml.dom import Node
try:
    from generatedssuper import GeneratedsSuper
except ImportError as exp:

    class GeneratedsSuper(object):

        def format_string(self, input_data, input_name=''):
            if False:
                i = 10
                return i + 15
            return input_data

        def format_integer(self, input_data, input_name=''):
            if False:
                while True:
                    i = 10
            return '%d' % input_data

        def format_float(self, input_data, input_name=''):
            if False:
                print('Hello World!')
            return '%f' % input_data

        def format_double(self, input_data, input_name=''):
            if False:
                print('Hello World!')
            return '%e' % input_data

        def format_boolean(self, input_data, input_name=''):
            if False:
                while True:
                    i = 10
            return '%s' % input_data
ExternalEncoding = 'ascii'

def showIndent(outfile, level):
    if False:
        print('Hello World!')
    for idx in range(level):
        outfile.write('    ')

def quote_xml(inStr):
    if False:
        return 10
    s1 = isinstance(inStr, str) and inStr or '%s' % inStr
    s1 = s1.replace('&', '&amp;')
    s1 = s1.replace('<', '&lt;')
    s1 = s1.replace('>', '&gt;')
    return s1

def quote_attrib(inStr):
    if False:
        return 10
    s1 = isinstance(inStr, str) and inStr or '%s' % inStr
    s1 = s1.replace('&', '&amp;')
    s1 = s1.replace('<', '&lt;')
    s1 = s1.replace('>', '&gt;')
    if '"' in s1:
        if "'" in s1:
            s1 = '"%s"' % s1.replace('"', '&quot;')
        else:
            s1 = "'%s'" % s1
    else:
        s1 = '"%s"' % s1
    return s1

def quote_python(inStr):
    if False:
        return 10
    s1 = inStr
    if s1.find("'") == -1:
        if s1.find('\n') == -1:
            return "'%s'" % s1
        else:
            return "'''%s'''" % s1
    else:
        if s1.find('"') != -1:
            s1 = s1.replace('"', '\\"')
        if s1.find('\n') == -1:
            return '"%s"' % s1
        else:
            return '"""%s"""' % s1

class MixedContainer(object):
    CategoryNone = 0
    CategoryText = 1
    CategorySimple = 2
    CategoryComplex = 3
    TypeNone = 0
    TypeText = 1
    TypeString = 2
    TypeInteger = 3
    TypeFloat = 4
    TypeDecimal = 5
    TypeDouble = 6
    TypeBoolean = 7

    def __init__(self, category, content_type, name, value):
        if False:
            return 10
        self.category = category
        self.content_type = content_type
        self.name = name
        self.value = value

    def getCategory(self):
        if False:
            i = 10
            return i + 15
        return self.category

    def getContenttype(self, content_type):
        if False:
            print('Hello World!')
        return self.content_type

    def getValue(self):
        if False:
            return 10
        return self.value

    def getName(self):
        if False:
            return 10
        return self.name

    def export(self, outfile, level, name, namespace):
        if False:
            return 10
        if self.category == MixedContainer.CategoryText:
            outfile.write(self.value)
        elif self.category == MixedContainer.CategorySimple:
            self.exportSimple(outfile, level, name)
        else:
            self.value.export(outfile, level, namespace, name)

    def exportSimple(self, outfile, level, name):
        if False:
            return 10
        if self.content_type == MixedContainer.TypeString:
            outfile.write('<%s>%s</%s>' % (self.name, self.value, self.name))
        elif self.content_type == MixedContainer.TypeInteger or self.content_type == MixedContainer.TypeBoolean:
            outfile.write('<%s>%d</%s>' % (self.name, self.value, self.name))
        elif self.content_type == MixedContainer.TypeFloat or self.content_type == MixedContainer.TypeDecimal:
            outfile.write('<%s>%f</%s>' % (self.name, self.value, self.name))
        elif self.content_type == MixedContainer.TypeDouble:
            outfile.write('<%s>%g</%s>' % (self.name, self.value, self.name))

    def exportLiteral(self, outfile, level, name):
        if False:
            print('Hello World!')
        if self.category == MixedContainer.CategoryText:
            showIndent(outfile, level)
            outfile.write('MixedContainer(%d, %d, "%s", "%s"),\n' % (self.category, self.content_type, self.name, self.value))
        elif self.category == MixedContainer.CategorySimple:
            showIndent(outfile, level)
            outfile.write('MixedContainer(%d, %d, "%s", "%s"),\n' % (self.category, self.content_type, self.name, self.value))
        else:
            showIndent(outfile, level)
            outfile.write('MixedContainer(%d, %d, "%s",\n' % (self.category, self.content_type, self.name))
            self.value.exportLiteral(outfile, level + 1)
            showIndent(outfile, level)
            outfile.write(')\n')

class _MemberSpec(object):

    def __init__(self, name='', data_type='', container=0):
        if False:
            return 10
        self.name = name
        self.data_type = data_type
        self.container = container

    def set_name(self, name):
        if False:
            while True:
                i = 10
        self.name = name

    def get_name(self):
        if False:
            while True:
                i = 10
        return self.name

    def set_data_type(self, data_type):
        if False:
            print('Hello World!')
        self.data_type = data_type

    def get_data_type(self):
        if False:
            while True:
                i = 10
        return self.data_type

    def set_container(self, container):
        if False:
            return 10
        self.container = container

    def get_container(self):
        if False:
            print('Hello World!')
        return self.container

class DoxygenType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, version=None, compound=None):
        if False:
            print('Hello World!')
        self.version = version
        if compound is None:
            self.compound = []
        else:
            self.compound = compound

    def factory(*args_, **kwargs_):
        if False:
            print('Hello World!')
        if DoxygenType.subclass:
            return DoxygenType.subclass(*args_, **kwargs_)
        else:
            return DoxygenType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_compound(self):
        if False:
            print('Hello World!')
        return self.compound

    def set_compound(self, compound):
        if False:
            for i in range(10):
                print('nop')
        self.compound = compound

    def add_compound(self, value):
        if False:
            while True:
                i = 10
        self.compound.append(value)

    def insert_compound(self, index, value):
        if False:
            for i in range(10):
                print('nop')
        self.compound[index] = value

    def get_version(self):
        if False:
            for i in range(10):
                print('nop')
        return self.version

    def set_version(self, version):
        if False:
            print('Hello World!')
        self.version = version

    def export(self, outfile, level, namespace_='', name_='DoxygenType', namespacedef_=''):
        if False:
            return 10
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='DoxygenType')
        if self.hasContent_():
            outfile.write('>\n')
            self.exportChildren(outfile, level + 1, namespace_, name_)
            showIndent(outfile, level)
            outfile.write('</%s%s>\n' % (namespace_, name_))
        else:
            outfile.write(' />\n')

    def exportAttributes(self, outfile, level, namespace_='', name_='DoxygenType'):
        if False:
            while True:
                i = 10
        outfile.write(' version=%s' % (self.format_string(quote_attrib(self.version).encode(ExternalEncoding), input_name='version'),))

    def exportChildren(self, outfile, level, namespace_='', name_='DoxygenType'):
        if False:
            for i in range(10):
                print('nop')
        for compound_ in self.compound:
            compound_.export(outfile, level, namespace_, name_='compound')

    def hasContent_(self):
        if False:
            for i in range(10):
                print('nop')
        if self.compound is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='DoxygenType'):
        if False:
            return 10
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            return 10
        if self.version is not None:
            showIndent(outfile, level)
            outfile.write('version = %s,\n' % (self.version,))

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            return 10
        showIndent(outfile, level)
        outfile.write('compound=[\n')
        level += 1
        for compound in self.compound:
            showIndent(outfile, level)
            outfile.write('model_.compound(\n')
            compound.exportLiteral(outfile, level, name_='compound')
            showIndent(outfile, level)
            outfile.write('),\n')
        level -= 1
        showIndent(outfile, level)
        outfile.write('],\n')

    def build(self, node_):
        if False:
            for i in range(10):
                print('nop')
        attrs = node_.attributes
        self.buildAttributes(attrs)
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            while True:
                i = 10
        if attrs.get('version'):
            self.version = attrs.get('version').value

    def buildChildren(self, child_, nodeName_):
        if False:
            print('Hello World!')
        if child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'compound':
            obj_ = CompoundType.factory()
            obj_.build(child_)
            self.compound.append(obj_)

class CompoundType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, kind=None, refid=None, name=None, member=None):
        if False:
            for i in range(10):
                print('nop')
        self.kind = kind
        self.refid = refid
        self.name = name
        if member is None:
            self.member = []
        else:
            self.member = member

    def factory(*args_, **kwargs_):
        if False:
            while True:
                i = 10
        if CompoundType.subclass:
            return CompoundType.subclass(*args_, **kwargs_)
        else:
            return CompoundType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_name(self):
        if False:
            while True:
                i = 10
        return self.name

    def set_name(self, name):
        if False:
            print('Hello World!')
        self.name = name

    def get_member(self):
        if False:
            print('Hello World!')
        return self.member

    def set_member(self, member):
        if False:
            print('Hello World!')
        self.member = member

    def add_member(self, value):
        if False:
            return 10
        self.member.append(value)

    def insert_member(self, index, value):
        if False:
            i = 10
            return i + 15
        self.member[index] = value

    def get_kind(self):
        if False:
            print('Hello World!')
        return self.kind

    def set_kind(self, kind):
        if False:
            return 10
        self.kind = kind

    def get_refid(self):
        if False:
            for i in range(10):
                print('nop')
        return self.refid

    def set_refid(self, refid):
        if False:
            return 10
        self.refid = refid

    def export(self, outfile, level, namespace_='', name_='CompoundType', namespacedef_=''):
        if False:
            i = 10
            return i + 15
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='CompoundType')
        if self.hasContent_():
            outfile.write('>\n')
            self.exportChildren(outfile, level + 1, namespace_, name_)
            showIndent(outfile, level)
            outfile.write('</%s%s>\n' % (namespace_, name_))
        else:
            outfile.write(' />\n')

    def exportAttributes(self, outfile, level, namespace_='', name_='CompoundType'):
        if False:
            i = 10
            return i + 15
        outfile.write(' kind=%s' % (quote_attrib(self.kind),))
        outfile.write(' refid=%s' % (self.format_string(quote_attrib(self.refid).encode(ExternalEncoding), input_name='refid'),))

    def exportChildren(self, outfile, level, namespace_='', name_='CompoundType'):
        if False:
            print('Hello World!')
        if self.name is not None:
            showIndent(outfile, level)
            outfile.write('<%sname>%s</%sname>\n' % (namespace_, self.format_string(quote_xml(self.name).encode(ExternalEncoding), input_name='name'), namespace_))
        for member_ in self.member:
            member_.export(outfile, level, namespace_, name_='member')

    def hasContent_(self):
        if False:
            return 10
        if self.name is not None or self.member is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='CompoundType'):
        if False:
            while True:
                i = 10
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            while True:
                i = 10
        if self.kind is not None:
            showIndent(outfile, level)
            outfile.write('kind = "%s",\n' % (self.kind,))
        if self.refid is not None:
            showIndent(outfile, level)
            outfile.write('refid = %s,\n' % (self.refid,))

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            print('Hello World!')
        showIndent(outfile, level)
        outfile.write('name=%s,\n' % quote_python(self.name).encode(ExternalEncoding))
        showIndent(outfile, level)
        outfile.write('member=[\n')
        level += 1
        for member in self.member:
            showIndent(outfile, level)
            outfile.write('model_.member(\n')
            member.exportLiteral(outfile, level, name_='member')
            showIndent(outfile, level)
            outfile.write('),\n')
        level -= 1
        showIndent(outfile, level)
        outfile.write('],\n')

    def build(self, node_):
        if False:
            print('Hello World!')
        attrs = node_.attributes
        self.buildAttributes(attrs)
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            return 10
        if attrs.get('kind'):
            self.kind = attrs.get('kind').value
        if attrs.get('refid'):
            self.refid = attrs.get('refid').value

    def buildChildren(self, child_, nodeName_):
        if False:
            for i in range(10):
                print('nop')
        if child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'name':
            name_ = ''
            for text__content_ in child_.childNodes:
                name_ += text__content_.nodeValue
            self.name = name_
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'member':
            obj_ = MemberType.factory()
            obj_.build(child_)
            self.member.append(obj_)

class MemberType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, kind=None, refid=None, name=None):
        if False:
            return 10
        self.kind = kind
        self.refid = refid
        self.name = name

    def factory(*args_, **kwargs_):
        if False:
            for i in range(10):
                print('nop')
        if MemberType.subclass:
            return MemberType.subclass(*args_, **kwargs_)
        else:
            return MemberType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_name(self):
        if False:
            while True:
                i = 10
        return self.name

    def set_name(self, name):
        if False:
            return 10
        self.name = name

    def get_kind(self):
        if False:
            while True:
                i = 10
        return self.kind

    def set_kind(self, kind):
        if False:
            while True:
                i = 10
        self.kind = kind

    def get_refid(self):
        if False:
            for i in range(10):
                print('nop')
        return self.refid

    def set_refid(self, refid):
        if False:
            for i in range(10):
                print('nop')
        self.refid = refid

    def export(self, outfile, level, namespace_='', name_='MemberType', namespacedef_=''):
        if False:
            return 10
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='MemberType')
        if self.hasContent_():
            outfile.write('>\n')
            self.exportChildren(outfile, level + 1, namespace_, name_)
            showIndent(outfile, level)
            outfile.write('</%s%s>\n' % (namespace_, name_))
        else:
            outfile.write(' />\n')

    def exportAttributes(self, outfile, level, namespace_='', name_='MemberType'):
        if False:
            for i in range(10):
                print('nop')
        outfile.write(' kind=%s' % (quote_attrib(self.kind),))
        outfile.write(' refid=%s' % (self.format_string(quote_attrib(self.refid).encode(ExternalEncoding), input_name='refid'),))

    def exportChildren(self, outfile, level, namespace_='', name_='MemberType'):
        if False:
            for i in range(10):
                print('nop')
        if self.name is not None:
            showIndent(outfile, level)
            outfile.write('<%sname>%s</%sname>\n' % (namespace_, self.format_string(quote_xml(self.name).encode(ExternalEncoding), input_name='name'), namespace_))

    def hasContent_(self):
        if False:
            for i in range(10):
                print('nop')
        if self.name is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='MemberType'):
        if False:
            return 10
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            for i in range(10):
                print('nop')
        if self.kind is not None:
            showIndent(outfile, level)
            outfile.write('kind = "%s",\n' % (self.kind,))
        if self.refid is not None:
            showIndent(outfile, level)
            outfile.write('refid = %s,\n' % (self.refid,))

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            for i in range(10):
                print('nop')
        showIndent(outfile, level)
        outfile.write('name=%s,\n' % quote_python(self.name).encode(ExternalEncoding))

    def build(self, node_):
        if False:
            while True:
                i = 10
        attrs = node_.attributes
        self.buildAttributes(attrs)
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            while True:
                i = 10
        if attrs.get('kind'):
            self.kind = attrs.get('kind').value
        if attrs.get('refid'):
            self.refid = attrs.get('refid').value

    def buildChildren(self, child_, nodeName_):
        if False:
            for i in range(10):
                print('nop')
        if child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'name':
            name_ = ''
            for text__content_ in child_.childNodes:
                name_ += text__content_.nodeValue
            self.name = name_
USAGE_TEXT = '\nUsage: python <Parser>.py [ -s ] <in_xml_file>\nOptions:\n    -s        Use the SAX parser, not the minidom parser.\n'

def usage():
    if False:
        while True:
            i = 10
    print(USAGE_TEXT)
    sys.exit(1)

def parse(inFileName):
    if False:
        i = 10
        return i + 15
    doc = minidom.parse(inFileName)
    rootNode = doc.documentElement
    rootObj = DoxygenType.factory()
    rootObj.build(rootNode)
    doc = None
    sys.stdout.write('<?xml version="1.0" ?>\n')
    rootObj.export(sys.stdout, 0, name_='doxygenindex', namespacedef_='')
    return rootObj

def parseString(inString):
    if False:
        print('Hello World!')
    doc = minidom.parseString(inString)
    rootNode = doc.documentElement
    rootObj = DoxygenType.factory()
    rootObj.build(rootNode)
    doc = None
    sys.stdout.write('<?xml version="1.0" ?>\n')
    rootObj.export(sys.stdout, 0, name_='doxygenindex', namespacedef_='')
    return rootObj

def parseLiteral(inFileName):
    if False:
        return 10
    doc = minidom.parse(inFileName)
    rootNode = doc.documentElement
    rootObj = DoxygenType.factory()
    rootObj.build(rootNode)
    doc = None
    sys.stdout.write('from index import *\n\n')
    sys.stdout.write('rootObj = doxygenindex(\n')
    rootObj.exportLiteral(sys.stdout, 0, name_='doxygenindex')
    sys.stdout.write(')\n')
    return rootObj

def main():
    if False:
        return 10
    args = sys.argv[1:]
    if len(args) == 1:
        parse(args[0])
    else:
        usage()
if __name__ == '__main__':
    main()