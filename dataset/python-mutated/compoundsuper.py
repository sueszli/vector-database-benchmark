import sys
from xml.dom import minidom
from xml.dom import Node
try:
    from generatedssuper import GeneratedsSuper
except ImportError as exp:

    class GeneratedsSuper(object):

        def format_string(self, input_data, input_name=''):
            if False:
                return 10
            return input_data

        def format_integer(self, input_data, input_name=''):
            if False:
                for i in range(10):
                    print('nop')
            return '%d' % input_data

        def format_float(self, input_data, input_name=''):
            if False:
                for i in range(10):
                    print('nop')
            return '%f' % input_data

        def format_double(self, input_data, input_name=''):
            if False:
                i = 10
                return i + 15
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
        print('Hello World!')
    s1 = isinstance(inStr, str) and inStr or '%s' % inStr
    s1 = s1.replace('&', '&amp;')
    s1 = s1.replace('<', '&lt;')
    s1 = s1.replace('>', '&gt;')
    return s1

def quote_attrib(inStr):
    if False:
        i = 10
        return i + 15
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
        while True:
            i = 10
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
            for i in range(10):
                print('nop')
        self.category = category
        self.content_type = content_type
        self.name = name
        self.value = value

    def getCategory(self):
        if False:
            for i in range(10):
                print('nop')
        return self.category

    def getContenttype(self, content_type):
        if False:
            i = 10
            return i + 15
        return self.content_type

    def getValue(self):
        if False:
            return 10
        return self.value

    def getName(self):
        if False:
            for i in range(10):
                print('nop')
        return self.name

    def export(self, outfile, level, name, namespace):
        if False:
            while True:
                i = 10
        if self.category == MixedContainer.CategoryText:
            outfile.write(self.value)
        elif self.category == MixedContainer.CategorySimple:
            self.exportSimple(outfile, level, name)
        else:
            self.value.export(outfile, level, namespace, name)

    def exportSimple(self, outfile, level, name):
        if False:
            print('Hello World!')
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
            i = 10
            return i + 15
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
            print('Hello World!')
        return self.name

    def set_data_type(self, data_type):
        if False:
            i = 10
            return i + 15
        self.data_type = data_type

    def get_data_type(self):
        if False:
            for i in range(10):
                print('nop')
        return self.data_type

    def set_container(self, container):
        if False:
            while True:
                i = 10
        self.container = container

    def get_container(self):
        if False:
            return 10
        return self.container

class DoxygenType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, version=None, compounddef=None):
        if False:
            return 10
        self.version = version
        self.compounddef = compounddef

    def factory(*args_, **kwargs_):
        if False:
            i = 10
            return i + 15
        if DoxygenType.subclass:
            return DoxygenType.subclass(*args_, **kwargs_)
        else:
            return DoxygenType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_compounddef(self):
        if False:
            while True:
                i = 10
        return self.compounddef

    def set_compounddef(self, compounddef):
        if False:
            return 10
        self.compounddef = compounddef

    def get_version(self):
        if False:
            print('Hello World!')
        return self.version

    def set_version(self, version):
        if False:
            return 10
        self.version = version

    def export(self, outfile, level, namespace_='', name_='DoxygenType', namespacedef_=''):
        if False:
            print('Hello World!')
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
            print('Hello World!')
        outfile.write(' version=%s' % (quote_attrib(self.version),))

    def exportChildren(self, outfile, level, namespace_='', name_='DoxygenType'):
        if False:
            while True:
                i = 10
        if self.compounddef:
            self.compounddef.export(outfile, level, namespace_, name_='compounddef')

    def hasContent_(self):
        if False:
            return 10
        if self.compounddef is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='DoxygenType'):
        if False:
            for i in range(10):
                print('nop')
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            print('Hello World!')
        if self.version is not None:
            showIndent(outfile, level)
            outfile.write('version = "%s",\n' % (self.version,))

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            print('Hello World!')
        if self.compounddef:
            showIndent(outfile, level)
            outfile.write('compounddef=model_.compounddefType(\n')
            self.compounddef.exportLiteral(outfile, level, name_='compounddef')
            showIndent(outfile, level)
            outfile.write('),\n')

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
            print('Hello World!')
        if attrs.get('version'):
            self.version = attrs.get('version').value

    def buildChildren(self, child_, nodeName_):
        if False:
            for i in range(10):
                print('nop')
        if child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'compounddef':
            obj_ = compounddefType.factory()
            obj_.build(child_)
            self.set_compounddef(obj_)

class compounddefType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, kind=None, prot=None, id=None, compoundname=None, title=None, basecompoundref=None, derivedcompoundref=None, includes=None, includedby=None, incdepgraph=None, invincdepgraph=None, innerdir=None, innerfile=None, innerclass=None, innernamespace=None, innerpage=None, innergroup=None, templateparamlist=None, sectiondef=None, briefdescription=None, detaileddescription=None, inheritancegraph=None, collaborationgraph=None, programlisting=None, location=None, listofallmembers=None):
        if False:
            print('Hello World!')
        self.kind = kind
        self.prot = prot
        self.id = id
        self.compoundname = compoundname
        self.title = title
        if basecompoundref is None:
            self.basecompoundref = []
        else:
            self.basecompoundref = basecompoundref
        if derivedcompoundref is None:
            self.derivedcompoundref = []
        else:
            self.derivedcompoundref = derivedcompoundref
        if includes is None:
            self.includes = []
        else:
            self.includes = includes
        if includedby is None:
            self.includedby = []
        else:
            self.includedby = includedby
        self.incdepgraph = incdepgraph
        self.invincdepgraph = invincdepgraph
        if innerdir is None:
            self.innerdir = []
        else:
            self.innerdir = innerdir
        if innerfile is None:
            self.innerfile = []
        else:
            self.innerfile = innerfile
        if innerclass is None:
            self.innerclass = []
        else:
            self.innerclass = innerclass
        if innernamespace is None:
            self.innernamespace = []
        else:
            self.innernamespace = innernamespace
        if innerpage is None:
            self.innerpage = []
        else:
            self.innerpage = innerpage
        if innergroup is None:
            self.innergroup = []
        else:
            self.innergroup = innergroup
        self.templateparamlist = templateparamlist
        if sectiondef is None:
            self.sectiondef = []
        else:
            self.sectiondef = sectiondef
        self.briefdescription = briefdescription
        self.detaileddescription = detaileddescription
        self.inheritancegraph = inheritancegraph
        self.collaborationgraph = collaborationgraph
        self.programlisting = programlisting
        self.location = location
        self.listofallmembers = listofallmembers

    def factory(*args_, **kwargs_):
        if False:
            for i in range(10):
                print('nop')
        if compounddefType.subclass:
            return compounddefType.subclass(*args_, **kwargs_)
        else:
            return compounddefType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_compoundname(self):
        if False:
            for i in range(10):
                print('nop')
        return self.compoundname

    def set_compoundname(self, compoundname):
        if False:
            while True:
                i = 10
        self.compoundname = compoundname

    def get_title(self):
        if False:
            for i in range(10):
                print('nop')
        return self.title

    def set_title(self, title):
        if False:
            i = 10
            return i + 15
        self.title = title

    def get_basecompoundref(self):
        if False:
            print('Hello World!')
        return self.basecompoundref

    def set_basecompoundref(self, basecompoundref):
        if False:
            while True:
                i = 10
        self.basecompoundref = basecompoundref

    def add_basecompoundref(self, value):
        if False:
            return 10
        self.basecompoundref.append(value)

    def insert_basecompoundref(self, index, value):
        if False:
            return 10
        self.basecompoundref[index] = value

    def get_derivedcompoundref(self):
        if False:
            return 10
        return self.derivedcompoundref

    def set_derivedcompoundref(self, derivedcompoundref):
        if False:
            for i in range(10):
                print('nop')
        self.derivedcompoundref = derivedcompoundref

    def add_derivedcompoundref(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.derivedcompoundref.append(value)

    def insert_derivedcompoundref(self, index, value):
        if False:
            print('Hello World!')
        self.derivedcompoundref[index] = value

    def get_includes(self):
        if False:
            while True:
                i = 10
        return self.includes

    def set_includes(self, includes):
        if False:
            for i in range(10):
                print('nop')
        self.includes = includes

    def add_includes(self, value):
        if False:
            while True:
                i = 10
        self.includes.append(value)

    def insert_includes(self, index, value):
        if False:
            return 10
        self.includes[index] = value

    def get_includedby(self):
        if False:
            while True:
                i = 10
        return self.includedby

    def set_includedby(self, includedby):
        if False:
            i = 10
            return i + 15
        self.includedby = includedby

    def add_includedby(self, value):
        if False:
            i = 10
            return i + 15
        self.includedby.append(value)

    def insert_includedby(self, index, value):
        if False:
            i = 10
            return i + 15
        self.includedby[index] = value

    def get_incdepgraph(self):
        if False:
            while True:
                i = 10
        return self.incdepgraph

    def set_incdepgraph(self, incdepgraph):
        if False:
            print('Hello World!')
        self.incdepgraph = incdepgraph

    def get_invincdepgraph(self):
        if False:
            while True:
                i = 10
        return self.invincdepgraph

    def set_invincdepgraph(self, invincdepgraph):
        if False:
            for i in range(10):
                print('nop')
        self.invincdepgraph = invincdepgraph

    def get_innerdir(self):
        if False:
            for i in range(10):
                print('nop')
        return self.innerdir

    def set_innerdir(self, innerdir):
        if False:
            return 10
        self.innerdir = innerdir

    def add_innerdir(self, value):
        if False:
            while True:
                i = 10
        self.innerdir.append(value)

    def insert_innerdir(self, index, value):
        if False:
            print('Hello World!')
        self.innerdir[index] = value

    def get_innerfile(self):
        if False:
            i = 10
            return i + 15
        return self.innerfile

    def set_innerfile(self, innerfile):
        if False:
            i = 10
            return i + 15
        self.innerfile = innerfile

    def add_innerfile(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.innerfile.append(value)

    def insert_innerfile(self, index, value):
        if False:
            print('Hello World!')
        self.innerfile[index] = value

    def get_innerclass(self):
        if False:
            print('Hello World!')
        return self.innerclass

    def set_innerclass(self, innerclass):
        if False:
            for i in range(10):
                print('nop')
        self.innerclass = innerclass

    def add_innerclass(self, value):
        if False:
            return 10
        self.innerclass.append(value)

    def insert_innerclass(self, index, value):
        if False:
            while True:
                i = 10
        self.innerclass[index] = value

    def get_innernamespace(self):
        if False:
            while True:
                i = 10
        return self.innernamespace

    def set_innernamespace(self, innernamespace):
        if False:
            return 10
        self.innernamespace = innernamespace

    def add_innernamespace(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.innernamespace.append(value)

    def insert_innernamespace(self, index, value):
        if False:
            while True:
                i = 10
        self.innernamespace[index] = value

    def get_innerpage(self):
        if False:
            while True:
                i = 10
        return self.innerpage

    def set_innerpage(self, innerpage):
        if False:
            return 10
        self.innerpage = innerpage

    def add_innerpage(self, value):
        if False:
            return 10
        self.innerpage.append(value)

    def insert_innerpage(self, index, value):
        if False:
            while True:
                i = 10
        self.innerpage[index] = value

    def get_innergroup(self):
        if False:
            print('Hello World!')
        return self.innergroup

    def set_innergroup(self, innergroup):
        if False:
            for i in range(10):
                print('nop')
        self.innergroup = innergroup

    def add_innergroup(self, value):
        if False:
            print('Hello World!')
        self.innergroup.append(value)

    def insert_innergroup(self, index, value):
        if False:
            while True:
                i = 10
        self.innergroup[index] = value

    def get_templateparamlist(self):
        if False:
            while True:
                i = 10
        return self.templateparamlist

    def set_templateparamlist(self, templateparamlist):
        if False:
            for i in range(10):
                print('nop')
        self.templateparamlist = templateparamlist

    def get_sectiondef(self):
        if False:
            i = 10
            return i + 15
        return self.sectiondef

    def set_sectiondef(self, sectiondef):
        if False:
            print('Hello World!')
        self.sectiondef = sectiondef

    def add_sectiondef(self, value):
        if False:
            print('Hello World!')
        self.sectiondef.append(value)

    def insert_sectiondef(self, index, value):
        if False:
            print('Hello World!')
        self.sectiondef[index] = value

    def get_briefdescription(self):
        if False:
            return 10
        return self.briefdescription

    def set_briefdescription(self, briefdescription):
        if False:
            while True:
                i = 10
        self.briefdescription = briefdescription

    def get_detaileddescription(self):
        if False:
            i = 10
            return i + 15
        return self.detaileddescription

    def set_detaileddescription(self, detaileddescription):
        if False:
            print('Hello World!')
        self.detaileddescription = detaileddescription

    def get_inheritancegraph(self):
        if False:
            while True:
                i = 10
        return self.inheritancegraph

    def set_inheritancegraph(self, inheritancegraph):
        if False:
            while True:
                i = 10
        self.inheritancegraph = inheritancegraph

    def get_collaborationgraph(self):
        if False:
            return 10
        return self.collaborationgraph

    def set_collaborationgraph(self, collaborationgraph):
        if False:
            i = 10
            return i + 15
        self.collaborationgraph = collaborationgraph

    def get_programlisting(self):
        if False:
            for i in range(10):
                print('nop')
        return self.programlisting

    def set_programlisting(self, programlisting):
        if False:
            while True:
                i = 10
        self.programlisting = programlisting

    def get_location(self):
        if False:
            for i in range(10):
                print('nop')
        return self.location

    def set_location(self, location):
        if False:
            for i in range(10):
                print('nop')
        self.location = location

    def get_listofallmembers(self):
        if False:
            print('Hello World!')
        return self.listofallmembers

    def set_listofallmembers(self, listofallmembers):
        if False:
            print('Hello World!')
        self.listofallmembers = listofallmembers

    def get_kind(self):
        if False:
            print('Hello World!')
        return self.kind

    def set_kind(self, kind):
        if False:
            while True:
                i = 10
        self.kind = kind

    def get_prot(self):
        if False:
            i = 10
            return i + 15
        return self.prot

    def set_prot(self, prot):
        if False:
            while True:
                i = 10
        self.prot = prot

    def get_id(self):
        if False:
            print('Hello World!')
        return self.id

    def set_id(self, id):
        if False:
            for i in range(10):
                print('nop')
        self.id = id

    def export(self, outfile, level, namespace_='', name_='compounddefType', namespacedef_=''):
        if False:
            while True:
                i = 10
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='compounddefType')
        if self.hasContent_():
            outfile.write('>\n')
            self.exportChildren(outfile, level + 1, namespace_, name_)
            showIndent(outfile, level)
            outfile.write('</%s%s>\n' % (namespace_, name_))
        else:
            outfile.write(' />\n')

    def exportAttributes(self, outfile, level, namespace_='', name_='compounddefType'):
        if False:
            print('Hello World!')
        if self.kind is not None:
            outfile.write(' kind=%s' % (quote_attrib(self.kind),))
        if self.prot is not None:
            outfile.write(' prot=%s' % (quote_attrib(self.prot),))
        if self.id is not None:
            outfile.write(' id=%s' % (self.format_string(quote_attrib(self.id).encode(ExternalEncoding), input_name='id'),))

    def exportChildren(self, outfile, level, namespace_='', name_='compounddefType'):
        if False:
            return 10
        if self.compoundname is not None:
            showIndent(outfile, level)
            outfile.write('<%scompoundname>%s</%scompoundname>\n' % (namespace_, self.format_string(quote_xml(self.compoundname).encode(ExternalEncoding), input_name='compoundname'), namespace_))
        if self.title is not None:
            showIndent(outfile, level)
            outfile.write('<%stitle>%s</%stitle>\n' % (namespace_, self.format_string(quote_xml(self.title).encode(ExternalEncoding), input_name='title'), namespace_))
        for basecompoundref_ in self.basecompoundref:
            basecompoundref_.export(outfile, level, namespace_, name_='basecompoundref')
        for derivedcompoundref_ in self.derivedcompoundref:
            derivedcompoundref_.export(outfile, level, namespace_, name_='derivedcompoundref')
        for includes_ in self.includes:
            includes_.export(outfile, level, namespace_, name_='includes')
        for includedby_ in self.includedby:
            includedby_.export(outfile, level, namespace_, name_='includedby')
        if self.incdepgraph:
            self.incdepgraph.export(outfile, level, namespace_, name_='incdepgraph')
        if self.invincdepgraph:
            self.invincdepgraph.export(outfile, level, namespace_, name_='invincdepgraph')
        for innerdir_ in self.innerdir:
            innerdir_.export(outfile, level, namespace_, name_='innerdir')
        for innerfile_ in self.innerfile:
            innerfile_.export(outfile, level, namespace_, name_='innerfile')
        for innerclass_ in self.innerclass:
            innerclass_.export(outfile, level, namespace_, name_='innerclass')
        for innernamespace_ in self.innernamespace:
            innernamespace_.export(outfile, level, namespace_, name_='innernamespace')
        for innerpage_ in self.innerpage:
            innerpage_.export(outfile, level, namespace_, name_='innerpage')
        for innergroup_ in self.innergroup:
            innergroup_.export(outfile, level, namespace_, name_='innergroup')
        if self.templateparamlist:
            self.templateparamlist.export(outfile, level, namespace_, name_='templateparamlist')
        for sectiondef_ in self.sectiondef:
            sectiondef_.export(outfile, level, namespace_, name_='sectiondef')
        if self.briefdescription:
            self.briefdescription.export(outfile, level, namespace_, name_='briefdescription')
        if self.detaileddescription:
            self.detaileddescription.export(outfile, level, namespace_, name_='detaileddescription')
        if self.inheritancegraph:
            self.inheritancegraph.export(outfile, level, namespace_, name_='inheritancegraph')
        if self.collaborationgraph:
            self.collaborationgraph.export(outfile, level, namespace_, name_='collaborationgraph')
        if self.programlisting:
            self.programlisting.export(outfile, level, namespace_, name_='programlisting')
        if self.location:
            self.location.export(outfile, level, namespace_, name_='location')
        if self.listofallmembers:
            self.listofallmembers.export(outfile, level, namespace_, name_='listofallmembers')

    def hasContent_(self):
        if False:
            while True:
                i = 10
        if self.compoundname is not None or self.title is not None or self.basecompoundref is not None or (self.derivedcompoundref is not None) or (self.includes is not None) or (self.includedby is not None) or (self.incdepgraph is not None) or (self.invincdepgraph is not None) or (self.innerdir is not None) or (self.innerfile is not None) or (self.innerclass is not None) or (self.innernamespace is not None) or (self.innerpage is not None) or (self.innergroup is not None) or (self.templateparamlist is not None) or (self.sectiondef is not None) or (self.briefdescription is not None) or (self.detaileddescription is not None) or (self.inheritancegraph is not None) or (self.collaborationgraph is not None) or (self.programlisting is not None) or (self.location is not None) or (self.listofallmembers is not None):
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='compounddefType'):
        if False:
            i = 10
            return i + 15
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            i = 10
            return i + 15
        if self.kind is not None:
            showIndent(outfile, level)
            outfile.write('kind = "%s",\n' % (self.kind,))
        if self.prot is not None:
            showIndent(outfile, level)
            outfile.write('prot = "%s",\n' % (self.prot,))
        if self.id is not None:
            showIndent(outfile, level)
            outfile.write('id = %s,\n' % (self.id,))

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            while True:
                i = 10
        showIndent(outfile, level)
        outfile.write('compoundname=%s,\n' % quote_python(self.compoundname).encode(ExternalEncoding))
        if self.title:
            showIndent(outfile, level)
            outfile.write('title=model_.xsd_string(\n')
            self.title.exportLiteral(outfile, level, name_='title')
            showIndent(outfile, level)
            outfile.write('),\n')
        showIndent(outfile, level)
        outfile.write('basecompoundref=[\n')
        level += 1
        for basecompoundref in self.basecompoundref:
            showIndent(outfile, level)
            outfile.write('model_.basecompoundref(\n')
            basecompoundref.exportLiteral(outfile, level, name_='basecompoundref')
            showIndent(outfile, level)
            outfile.write('),\n')
        level -= 1
        showIndent(outfile, level)
        outfile.write('],\n')
        showIndent(outfile, level)
        outfile.write('derivedcompoundref=[\n')
        level += 1
        for derivedcompoundref in self.derivedcompoundref:
            showIndent(outfile, level)
            outfile.write('model_.derivedcompoundref(\n')
            derivedcompoundref.exportLiteral(outfile, level, name_='derivedcompoundref')
            showIndent(outfile, level)
            outfile.write('),\n')
        level -= 1
        showIndent(outfile, level)
        outfile.write('],\n')
        showIndent(outfile, level)
        outfile.write('includes=[\n')
        level += 1
        for includes in self.includes:
            showIndent(outfile, level)
            outfile.write('model_.includes(\n')
            includes.exportLiteral(outfile, level, name_='includes')
            showIndent(outfile, level)
            outfile.write('),\n')
        level -= 1
        showIndent(outfile, level)
        outfile.write('],\n')
        showIndent(outfile, level)
        outfile.write('includedby=[\n')
        level += 1
        for includedby in self.includedby:
            showIndent(outfile, level)
            outfile.write('model_.includedby(\n')
            includedby.exportLiteral(outfile, level, name_='includedby')
            showIndent(outfile, level)
            outfile.write('),\n')
        level -= 1
        showIndent(outfile, level)
        outfile.write('],\n')
        if self.incdepgraph:
            showIndent(outfile, level)
            outfile.write('incdepgraph=model_.graphType(\n')
            self.incdepgraph.exportLiteral(outfile, level, name_='incdepgraph')
            showIndent(outfile, level)
            outfile.write('),\n')
        if self.invincdepgraph:
            showIndent(outfile, level)
            outfile.write('invincdepgraph=model_.graphType(\n')
            self.invincdepgraph.exportLiteral(outfile, level, name_='invincdepgraph')
            showIndent(outfile, level)
            outfile.write('),\n')
        showIndent(outfile, level)
        outfile.write('innerdir=[\n')
        level += 1
        for innerdir in self.innerdir:
            showIndent(outfile, level)
            outfile.write('model_.innerdir(\n')
            innerdir.exportLiteral(outfile, level, name_='innerdir')
            showIndent(outfile, level)
            outfile.write('),\n')
        level -= 1
        showIndent(outfile, level)
        outfile.write('],\n')
        showIndent(outfile, level)
        outfile.write('innerfile=[\n')
        level += 1
        for innerfile in self.innerfile:
            showIndent(outfile, level)
            outfile.write('model_.innerfile(\n')
            innerfile.exportLiteral(outfile, level, name_='innerfile')
            showIndent(outfile, level)
            outfile.write('),\n')
        level -= 1
        showIndent(outfile, level)
        outfile.write('],\n')
        showIndent(outfile, level)
        outfile.write('innerclass=[\n')
        level += 1
        for innerclass in self.innerclass:
            showIndent(outfile, level)
            outfile.write('model_.innerclass(\n')
            innerclass.exportLiteral(outfile, level, name_='innerclass')
            showIndent(outfile, level)
            outfile.write('),\n')
        level -= 1
        showIndent(outfile, level)
        outfile.write('],\n')
        showIndent(outfile, level)
        outfile.write('innernamespace=[\n')
        level += 1
        for innernamespace in self.innernamespace:
            showIndent(outfile, level)
            outfile.write('model_.innernamespace(\n')
            innernamespace.exportLiteral(outfile, level, name_='innernamespace')
            showIndent(outfile, level)
            outfile.write('),\n')
        level -= 1
        showIndent(outfile, level)
        outfile.write('],\n')
        showIndent(outfile, level)
        outfile.write('innerpage=[\n')
        level += 1
        for innerpage in self.innerpage:
            showIndent(outfile, level)
            outfile.write('model_.innerpage(\n')
            innerpage.exportLiteral(outfile, level, name_='innerpage')
            showIndent(outfile, level)
            outfile.write('),\n')
        level -= 1
        showIndent(outfile, level)
        outfile.write('],\n')
        showIndent(outfile, level)
        outfile.write('innergroup=[\n')
        level += 1
        for innergroup in self.innergroup:
            showIndent(outfile, level)
            outfile.write('model_.innergroup(\n')
            innergroup.exportLiteral(outfile, level, name_='innergroup')
            showIndent(outfile, level)
            outfile.write('),\n')
        level -= 1
        showIndent(outfile, level)
        outfile.write('],\n')
        if self.templateparamlist:
            showIndent(outfile, level)
            outfile.write('templateparamlist=model_.templateparamlistType(\n')
            self.templateparamlist.exportLiteral(outfile, level, name_='templateparamlist')
            showIndent(outfile, level)
            outfile.write('),\n')
        showIndent(outfile, level)
        outfile.write('sectiondef=[\n')
        level += 1
        for sectiondef in self.sectiondef:
            showIndent(outfile, level)
            outfile.write('model_.sectiondef(\n')
            sectiondef.exportLiteral(outfile, level, name_='sectiondef')
            showIndent(outfile, level)
            outfile.write('),\n')
        level -= 1
        showIndent(outfile, level)
        outfile.write('],\n')
        if self.briefdescription:
            showIndent(outfile, level)
            outfile.write('briefdescription=model_.descriptionType(\n')
            self.briefdescription.exportLiteral(outfile, level, name_='briefdescription')
            showIndent(outfile, level)
            outfile.write('),\n')
        if self.detaileddescription:
            showIndent(outfile, level)
            outfile.write('detaileddescription=model_.descriptionType(\n')
            self.detaileddescription.exportLiteral(outfile, level, name_='detaileddescription')
            showIndent(outfile, level)
            outfile.write('),\n')
        if self.inheritancegraph:
            showIndent(outfile, level)
            outfile.write('inheritancegraph=model_.graphType(\n')
            self.inheritancegraph.exportLiteral(outfile, level, name_='inheritancegraph')
            showIndent(outfile, level)
            outfile.write('),\n')
        if self.collaborationgraph:
            showIndent(outfile, level)
            outfile.write('collaborationgraph=model_.graphType(\n')
            self.collaborationgraph.exportLiteral(outfile, level, name_='collaborationgraph')
            showIndent(outfile, level)
            outfile.write('),\n')
        if self.programlisting:
            showIndent(outfile, level)
            outfile.write('programlisting=model_.listingType(\n')
            self.programlisting.exportLiteral(outfile, level, name_='programlisting')
            showIndent(outfile, level)
            outfile.write('),\n')
        if self.location:
            showIndent(outfile, level)
            outfile.write('location=model_.locationType(\n')
            self.location.exportLiteral(outfile, level, name_='location')
            showIndent(outfile, level)
            outfile.write('),\n')
        if self.listofallmembers:
            showIndent(outfile, level)
            outfile.write('listofallmembers=model_.listofallmembersType(\n')
            self.listofallmembers.exportLiteral(outfile, level, name_='listofallmembers')
            showIndent(outfile, level)
            outfile.write('),\n')

    def build(self, node_):
        if False:
            return 10
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
        if attrs.get('prot'):
            self.prot = attrs.get('prot').value
        if attrs.get('id'):
            self.id = attrs.get('id').value

    def buildChildren(self, child_, nodeName_):
        if False:
            print('Hello World!')
        if child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'compoundname':
            compoundname_ = ''
            for text__content_ in child_.childNodes:
                compoundname_ += text__content_.nodeValue
            self.compoundname = compoundname_
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'title':
            obj_ = docTitleType.factory()
            obj_.build(child_)
            self.set_title(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'basecompoundref':
            obj_ = compoundRefType.factory()
            obj_.build(child_)
            self.basecompoundref.append(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'derivedcompoundref':
            obj_ = compoundRefType.factory()
            obj_.build(child_)
            self.derivedcompoundref.append(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'includes':
            obj_ = incType.factory()
            obj_.build(child_)
            self.includes.append(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'includedby':
            obj_ = incType.factory()
            obj_.build(child_)
            self.includedby.append(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'incdepgraph':
            obj_ = graphType.factory()
            obj_.build(child_)
            self.set_incdepgraph(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'invincdepgraph':
            obj_ = graphType.factory()
            obj_.build(child_)
            self.set_invincdepgraph(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'innerdir':
            obj_ = refType.factory()
            obj_.build(child_)
            self.innerdir.append(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'innerfile':
            obj_ = refType.factory()
            obj_.build(child_)
            self.innerfile.append(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'innerclass':
            obj_ = refType.factory()
            obj_.build(child_)
            self.innerclass.append(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'innernamespace':
            obj_ = refType.factory()
            obj_.build(child_)
            self.innernamespace.append(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'innerpage':
            obj_ = refType.factory()
            obj_.build(child_)
            self.innerpage.append(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'innergroup':
            obj_ = refType.factory()
            obj_.build(child_)
            self.innergroup.append(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'templateparamlist':
            obj_ = templateparamlistType.factory()
            obj_.build(child_)
            self.set_templateparamlist(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'sectiondef':
            obj_ = sectiondefType.factory()
            obj_.build(child_)
            self.sectiondef.append(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'briefdescription':
            obj_ = descriptionType.factory()
            obj_.build(child_)
            self.set_briefdescription(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'detaileddescription':
            obj_ = descriptionType.factory()
            obj_.build(child_)
            self.set_detaileddescription(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'inheritancegraph':
            obj_ = graphType.factory()
            obj_.build(child_)
            self.set_inheritancegraph(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'collaborationgraph':
            obj_ = graphType.factory()
            obj_.build(child_)
            self.set_collaborationgraph(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'programlisting':
            obj_ = listingType.factory()
            obj_.build(child_)
            self.set_programlisting(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'location':
            obj_ = locationType.factory()
            obj_.build(child_)
            self.set_location(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'listofallmembers':
            obj_ = listofallmembersType.factory()
            obj_.build(child_)
            self.set_listofallmembers(obj_)

class listofallmembersType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, member=None):
        if False:
            i = 10
            return i + 15
        if member is None:
            self.member = []
        else:
            self.member = member

    def factory(*args_, **kwargs_):
        if False:
            for i in range(10):
                print('nop')
        if listofallmembersType.subclass:
            return listofallmembersType.subclass(*args_, **kwargs_)
        else:
            return listofallmembersType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_member(self):
        if False:
            for i in range(10):
                print('nop')
        return self.member

    def set_member(self, member):
        if False:
            for i in range(10):
                print('nop')
        self.member = member

    def add_member(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.member.append(value)

    def insert_member(self, index, value):
        if False:
            print('Hello World!')
        self.member[index] = value

    def export(self, outfile, level, namespace_='', name_='listofallmembersType', namespacedef_=''):
        if False:
            return 10
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='listofallmembersType')
        if self.hasContent_():
            outfile.write('>\n')
            self.exportChildren(outfile, level + 1, namespace_, name_)
            showIndent(outfile, level)
            outfile.write('</%s%s>\n' % (namespace_, name_))
        else:
            outfile.write(' />\n')

    def exportAttributes(self, outfile, level, namespace_='', name_='listofallmembersType'):
        if False:
            while True:
                i = 10
        pass

    def exportChildren(self, outfile, level, namespace_='', name_='listofallmembersType'):
        if False:
            return 10
        for member_ in self.member:
            member_.export(outfile, level, namespace_, name_='member')

    def hasContent_(self):
        if False:
            print('Hello World!')
        if self.member is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='listofallmembersType'):
        if False:
            while True:
                i = 10
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            return 10
        pass

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            while True:
                i = 10
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
            while True:
                i = 10
        attrs = node_.attributes
        self.buildAttributes(attrs)
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            print('Hello World!')
        pass

    def buildChildren(self, child_, nodeName_):
        if False:
            while True:
                i = 10
        if child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'member':
            obj_ = memberRefType.factory()
            obj_.build(child_)
            self.member.append(obj_)

class memberRefType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, virt=None, prot=None, refid=None, ambiguityscope=None, scope=None, name=None):
        if False:
            for i in range(10):
                print('nop')
        self.virt = virt
        self.prot = prot
        self.refid = refid
        self.ambiguityscope = ambiguityscope
        self.scope = scope
        self.name = name

    def factory(*args_, **kwargs_):
        if False:
            for i in range(10):
                print('nop')
        if memberRefType.subclass:
            return memberRefType.subclass(*args_, **kwargs_)
        else:
            return memberRefType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_scope(self):
        if False:
            for i in range(10):
                print('nop')
        return self.scope

    def set_scope(self, scope):
        if False:
            print('Hello World!')
        self.scope = scope

    def get_name(self):
        if False:
            print('Hello World!')
        return self.name

    def set_name(self, name):
        if False:
            print('Hello World!')
        self.name = name

    def get_virt(self):
        if False:
            return 10
        return self.virt

    def set_virt(self, virt):
        if False:
            return 10
        self.virt = virt

    def get_prot(self):
        if False:
            while True:
                i = 10
        return self.prot

    def set_prot(self, prot):
        if False:
            print('Hello World!')
        self.prot = prot

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

    def get_ambiguityscope(self):
        if False:
            return 10
        return self.ambiguityscope

    def set_ambiguityscope(self, ambiguityscope):
        if False:
            i = 10
            return i + 15
        self.ambiguityscope = ambiguityscope

    def export(self, outfile, level, namespace_='', name_='memberRefType', namespacedef_=''):
        if False:
            for i in range(10):
                print('nop')
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='memberRefType')
        if self.hasContent_():
            outfile.write('>\n')
            self.exportChildren(outfile, level + 1, namespace_, name_)
            showIndent(outfile, level)
            outfile.write('</%s%s>\n' % (namespace_, name_))
        else:
            outfile.write(' />\n')

    def exportAttributes(self, outfile, level, namespace_='', name_='memberRefType'):
        if False:
            i = 10
            return i + 15
        if self.virt is not None:
            outfile.write(' virt=%s' % (quote_attrib(self.virt),))
        if self.prot is not None:
            outfile.write(' prot=%s' % (quote_attrib(self.prot),))
        if self.refid is not None:
            outfile.write(' refid=%s' % (self.format_string(quote_attrib(self.refid).encode(ExternalEncoding), input_name='refid'),))
        if self.ambiguityscope is not None:
            outfile.write(' ambiguityscope=%s' % (self.format_string(quote_attrib(self.ambiguityscope).encode(ExternalEncoding), input_name='ambiguityscope'),))

    def exportChildren(self, outfile, level, namespace_='', name_='memberRefType'):
        if False:
            i = 10
            return i + 15
        if self.scope is not None:
            showIndent(outfile, level)
            outfile.write('<%sscope>%s</%sscope>\n' % (namespace_, self.format_string(quote_xml(self.scope).encode(ExternalEncoding), input_name='scope'), namespace_))
        if self.name is not None:
            showIndent(outfile, level)
            outfile.write('<%sname>%s</%sname>\n' % (namespace_, self.format_string(quote_xml(self.name).encode(ExternalEncoding), input_name='name'), namespace_))

    def hasContent_(self):
        if False:
            return 10
        if self.scope is not None or self.name is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='memberRefType'):
        if False:
            for i in range(10):
                print('nop')
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            print('Hello World!')
        if self.virt is not None:
            showIndent(outfile, level)
            outfile.write('virt = "%s",\n' % (self.virt,))
        if self.prot is not None:
            showIndent(outfile, level)
            outfile.write('prot = "%s",\n' % (self.prot,))
        if self.refid is not None:
            showIndent(outfile, level)
            outfile.write('refid = %s,\n' % (self.refid,))
        if self.ambiguityscope is not None:
            showIndent(outfile, level)
            outfile.write('ambiguityscope = %s,\n' % (self.ambiguityscope,))

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            return 10
        showIndent(outfile, level)
        outfile.write('scope=%s,\n' % quote_python(self.scope).encode(ExternalEncoding))
        showIndent(outfile, level)
        outfile.write('name=%s,\n' % quote_python(self.name).encode(ExternalEncoding))

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
            i = 10
            return i + 15
        if attrs.get('virt'):
            self.virt = attrs.get('virt').value
        if attrs.get('prot'):
            self.prot = attrs.get('prot').value
        if attrs.get('refid'):
            self.refid = attrs.get('refid').value
        if attrs.get('ambiguityscope'):
            self.ambiguityscope = attrs.get('ambiguityscope').value

    def buildChildren(self, child_, nodeName_):
        if False:
            return 10
        if child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'scope':
            scope_ = ''
            for text__content_ in child_.childNodes:
                scope_ += text__content_.nodeValue
            self.scope = scope_
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'name':
            name_ = ''
            for text__content_ in child_.childNodes:
                name_ += text__content_.nodeValue
            self.name = name_

class scope(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, valueOf_=''):
        if False:
            i = 10
            return i + 15
        self.valueOf_ = valueOf_

    def factory(*args_, **kwargs_):
        if False:
            i = 10
            return i + 15
        if scope.subclass:
            return scope.subclass(*args_, **kwargs_)
        else:
            return scope(*args_, **kwargs_)
    factory = staticmethod(factory)

    def getValueOf_(self):
        if False:
            while True:
                i = 10
        return self.valueOf_

    def setValueOf_(self, valueOf_):
        if False:
            print('Hello World!')
        self.valueOf_ = valueOf_

    def export(self, outfile, level, namespace_='', name_='scope', namespacedef_=''):
        if False:
            while True:
                i = 10
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='scope')
        if self.hasContent_():
            outfile.write('>\n')
            self.exportChildren(outfile, level + 1, namespace_, name_)
            showIndent(outfile, level)
            outfile.write('</%s%s>\n' % (namespace_, name_))
        else:
            outfile.write(' />\n')

    def exportAttributes(self, outfile, level, namespace_='', name_='scope'):
        if False:
            for i in range(10):
                print('nop')
        pass

    def exportChildren(self, outfile, level, namespace_='', name_='scope'):
        if False:
            while True:
                i = 10
        if self.valueOf_.find('![CDATA') > -1:
            value = quote_xml('%s' % self.valueOf_)
            value = value.replace('![CDATA', '<![CDATA')
            value = value.replace(']]', ']]>')
            outfile.write(value)
        else:
            outfile.write(quote_xml('%s' % self.valueOf_))

    def hasContent_(self):
        if False:
            for i in range(10):
                print('nop')
        if self.valueOf_ is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='scope'):
        if False:
            i = 10
            return i + 15
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            return 10
        pass

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            while True:
                i = 10
        showIndent(outfile, level)
        outfile.write('valueOf_ = "%s",\n' % (self.valueOf_,))

    def build(self, node_):
        if False:
            for i in range(10):
                print('nop')
        attrs = node_.attributes
        self.buildAttributes(attrs)
        self.valueOf_ = ''
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            i = 10
            return i + 15
        pass

    def buildChildren(self, child_, nodeName_):
        if False:
            for i in range(10):
                print('nop')
        if child_.nodeType == Node.TEXT_NODE:
            self.valueOf_ += child_.nodeValue
        elif child_.nodeType == Node.CDATA_SECTION_NODE:
            self.valueOf_ += '![CDATA[' + child_.nodeValue + ']]'

class name(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, valueOf_=''):
        if False:
            i = 10
            return i + 15
        self.valueOf_ = valueOf_

    def factory(*args_, **kwargs_):
        if False:
            print('Hello World!')
        if name.subclass:
            return name.subclass(*args_, **kwargs_)
        else:
            return name(*args_, **kwargs_)
    factory = staticmethod(factory)

    def getValueOf_(self):
        if False:
            for i in range(10):
                print('nop')
        return self.valueOf_

    def setValueOf_(self, valueOf_):
        if False:
            i = 10
            return i + 15
        self.valueOf_ = valueOf_

    def export(self, outfile, level, namespace_='', name_='name', namespacedef_=''):
        if False:
            print('Hello World!')
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='name')
        if self.hasContent_():
            outfile.write('>\n')
            self.exportChildren(outfile, level + 1, namespace_, name_)
            showIndent(outfile, level)
            outfile.write('</%s%s>\n' % (namespace_, name_))
        else:
            outfile.write(' />\n')

    def exportAttributes(self, outfile, level, namespace_='', name_='name'):
        if False:
            i = 10
            return i + 15
        pass

    def exportChildren(self, outfile, level, namespace_='', name_='name'):
        if False:
            while True:
                i = 10
        if self.valueOf_.find('![CDATA') > -1:
            value = quote_xml('%s' % self.valueOf_)
            value = value.replace('![CDATA', '<![CDATA')
            value = value.replace(']]', ']]>')
            outfile.write(value)
        else:
            outfile.write(quote_xml('%s' % self.valueOf_))

    def hasContent_(self):
        if False:
            i = 10
            return i + 15
        if self.valueOf_ is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='name'):
        if False:
            print('Hello World!')
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            return 10
        pass

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            return 10
        showIndent(outfile, level)
        outfile.write('valueOf_ = "%s",\n' % (self.valueOf_,))

    def build(self, node_):
        if False:
            for i in range(10):
                print('nop')
        attrs = node_.attributes
        self.buildAttributes(attrs)
        self.valueOf_ = ''
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            while True:
                i = 10
        pass

    def buildChildren(self, child_, nodeName_):
        if False:
            return 10
        if child_.nodeType == Node.TEXT_NODE:
            self.valueOf_ += child_.nodeValue
        elif child_.nodeType == Node.CDATA_SECTION_NODE:
            self.valueOf_ += '![CDATA[' + child_.nodeValue + ']]'

class compoundRefType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, virt=None, prot=None, refid=None, valueOf_='', mixedclass_=None, content_=None):
        if False:
            i = 10
            return i + 15
        self.virt = virt
        self.prot = prot
        self.refid = refid
        if mixedclass_ is None:
            self.mixedclass_ = MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        if content_ is None:
            self.content_ = []
        else:
            self.content_ = content_

    def factory(*args_, **kwargs_):
        if False:
            i = 10
            return i + 15
        if compoundRefType.subclass:
            return compoundRefType.subclass(*args_, **kwargs_)
        else:
            return compoundRefType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_virt(self):
        if False:
            return 10
        return self.virt

    def set_virt(self, virt):
        if False:
            i = 10
            return i + 15
        self.virt = virt

    def get_prot(self):
        if False:
            print('Hello World!')
        return self.prot

    def set_prot(self, prot):
        if False:
            return 10
        self.prot = prot

    def get_refid(self):
        if False:
            while True:
                i = 10
        return self.refid

    def set_refid(self, refid):
        if False:
            for i in range(10):
                print('nop')
        self.refid = refid

    def getValueOf_(self):
        if False:
            for i in range(10):
                print('nop')
        return self.valueOf_

    def setValueOf_(self, valueOf_):
        if False:
            for i in range(10):
                print('nop')
        self.valueOf_ = valueOf_

    def export(self, outfile, level, namespace_='', name_='compoundRefType', namespacedef_=''):
        if False:
            while True:
                i = 10
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='compoundRefType')
        outfile.write('>')
        self.exportChildren(outfile, level + 1, namespace_, name_)
        outfile.write('</%s%s>\n' % (namespace_, name_))

    def exportAttributes(self, outfile, level, namespace_='', name_='compoundRefType'):
        if False:
            print('Hello World!')
        if self.virt is not None:
            outfile.write(' virt=%s' % (quote_attrib(self.virt),))
        if self.prot is not None:
            outfile.write(' prot=%s' % (quote_attrib(self.prot),))
        if self.refid is not None:
            outfile.write(' refid=%s' % (self.format_string(quote_attrib(self.refid).encode(ExternalEncoding), input_name='refid'),))

    def exportChildren(self, outfile, level, namespace_='', name_='compoundRefType'):
        if False:
            for i in range(10):
                print('nop')
        if self.valueOf_.find('![CDATA') > -1:
            value = quote_xml('%s' % self.valueOf_)
            value = value.replace('![CDATA', '<![CDATA')
            value = value.replace(']]', ']]>')
            outfile.write(value)
        else:
            outfile.write(quote_xml('%s' % self.valueOf_))

    def hasContent_(self):
        if False:
            for i in range(10):
                print('nop')
        if self.valueOf_ is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='compoundRefType'):
        if False:
            for i in range(10):
                print('nop')
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            for i in range(10):
                print('nop')
        if self.virt is not None:
            showIndent(outfile, level)
            outfile.write('virt = "%s",\n' % (self.virt,))
        if self.prot is not None:
            showIndent(outfile, level)
            outfile.write('prot = "%s",\n' % (self.prot,))
        if self.refid is not None:
            showIndent(outfile, level)
            outfile.write('refid = %s,\n' % (self.refid,))

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            i = 10
            return i + 15
        showIndent(outfile, level)
        outfile.write('valueOf_ = "%s",\n' % (self.valueOf_,))

    def build(self, node_):
        if False:
            while True:
                i = 10
        attrs = node_.attributes
        self.buildAttributes(attrs)
        self.valueOf_ = ''
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            while True:
                i = 10
        if attrs.get('virt'):
            self.virt = attrs.get('virt').value
        if attrs.get('prot'):
            self.prot = attrs.get('prot').value
        if attrs.get('refid'):
            self.refid = attrs.get('refid').value

    def buildChildren(self, child_, nodeName_):
        if False:
            i = 10
            return i + 15
        if child_.nodeType == Node.TEXT_NODE:
            obj_ = self.mixedclass_(MixedContainer.CategoryText, MixedContainer.TypeNone, '', child_.nodeValue)
            self.content_.append(obj_)
        if child_.nodeType == Node.TEXT_NODE:
            self.valueOf_ += child_.nodeValue
        elif child_.nodeType == Node.CDATA_SECTION_NODE:
            self.valueOf_ += '![CDATA[' + child_.nodeValue + ']]'

class reimplementType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, refid=None, valueOf_='', mixedclass_=None, content_=None):
        if False:
            for i in range(10):
                print('nop')
        self.refid = refid
        if mixedclass_ is None:
            self.mixedclass_ = MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        if content_ is None:
            self.content_ = []
        else:
            self.content_ = content_

    def factory(*args_, **kwargs_):
        if False:
            while True:
                i = 10
        if reimplementType.subclass:
            return reimplementType.subclass(*args_, **kwargs_)
        else:
            return reimplementType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_refid(self):
        if False:
            for i in range(10):
                print('nop')
        return self.refid

    def set_refid(self, refid):
        if False:
            print('Hello World!')
        self.refid = refid

    def getValueOf_(self):
        if False:
            return 10
        return self.valueOf_

    def setValueOf_(self, valueOf_):
        if False:
            print('Hello World!')
        self.valueOf_ = valueOf_

    def export(self, outfile, level, namespace_='', name_='reimplementType', namespacedef_=''):
        if False:
            while True:
                i = 10
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='reimplementType')
        outfile.write('>')
        self.exportChildren(outfile, level + 1, namespace_, name_)
        outfile.write('</%s%s>\n' % (namespace_, name_))

    def exportAttributes(self, outfile, level, namespace_='', name_='reimplementType'):
        if False:
            while True:
                i = 10
        if self.refid is not None:
            outfile.write(' refid=%s' % (self.format_string(quote_attrib(self.refid).encode(ExternalEncoding), input_name='refid'),))

    def exportChildren(self, outfile, level, namespace_='', name_='reimplementType'):
        if False:
            while True:
                i = 10
        if self.valueOf_.find('![CDATA') > -1:
            value = quote_xml('%s' % self.valueOf_)
            value = value.replace('![CDATA', '<![CDATA')
            value = value.replace(']]', ']]>')
            outfile.write(value)
        else:
            outfile.write(quote_xml('%s' % self.valueOf_))

    def hasContent_(self):
        if False:
            for i in range(10):
                print('nop')
        if self.valueOf_ is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='reimplementType'):
        if False:
            i = 10
            return i + 15
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            return 10
        if self.refid is not None:
            showIndent(outfile, level)
            outfile.write('refid = %s,\n' % (self.refid,))

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            i = 10
            return i + 15
        showIndent(outfile, level)
        outfile.write('valueOf_ = "%s",\n' % (self.valueOf_,))

    def build(self, node_):
        if False:
            return 10
        attrs = node_.attributes
        self.buildAttributes(attrs)
        self.valueOf_ = ''
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            return 10
        if attrs.get('refid'):
            self.refid = attrs.get('refid').value

    def buildChildren(self, child_, nodeName_):
        if False:
            for i in range(10):
                print('nop')
        if child_.nodeType == Node.TEXT_NODE:
            obj_ = self.mixedclass_(MixedContainer.CategoryText, MixedContainer.TypeNone, '', child_.nodeValue)
            self.content_.append(obj_)
        if child_.nodeType == Node.TEXT_NODE:
            self.valueOf_ += child_.nodeValue
        elif child_.nodeType == Node.CDATA_SECTION_NODE:
            self.valueOf_ += '![CDATA[' + child_.nodeValue + ']]'

class incType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, local=None, refid=None, valueOf_='', mixedclass_=None, content_=None):
        if False:
            i = 10
            return i + 15
        self.local = local
        self.refid = refid
        if mixedclass_ is None:
            self.mixedclass_ = MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        if content_ is None:
            self.content_ = []
        else:
            self.content_ = content_

    def factory(*args_, **kwargs_):
        if False:
            return 10
        if incType.subclass:
            return incType.subclass(*args_, **kwargs_)
        else:
            return incType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_local(self):
        if False:
            return 10
        return self.local

    def set_local(self, local):
        if False:
            while True:
                i = 10
        self.local = local

    def get_refid(self):
        if False:
            while True:
                i = 10
        return self.refid

    def set_refid(self, refid):
        if False:
            print('Hello World!')
        self.refid = refid

    def getValueOf_(self):
        if False:
            print('Hello World!')
        return self.valueOf_

    def setValueOf_(self, valueOf_):
        if False:
            for i in range(10):
                print('nop')
        self.valueOf_ = valueOf_

    def export(self, outfile, level, namespace_='', name_='incType', namespacedef_=''):
        if False:
            for i in range(10):
                print('nop')
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='incType')
        outfile.write('>')
        self.exportChildren(outfile, level + 1, namespace_, name_)
        outfile.write('</%s%s>\n' % (namespace_, name_))

    def exportAttributes(self, outfile, level, namespace_='', name_='incType'):
        if False:
            i = 10
            return i + 15
        if self.local is not None:
            outfile.write(' local=%s' % (quote_attrib(self.local),))
        if self.refid is not None:
            outfile.write(' refid=%s' % (self.format_string(quote_attrib(self.refid).encode(ExternalEncoding), input_name='refid'),))

    def exportChildren(self, outfile, level, namespace_='', name_='incType'):
        if False:
            while True:
                i = 10
        if self.valueOf_.find('![CDATA') > -1:
            value = quote_xml('%s' % self.valueOf_)
            value = value.replace('![CDATA', '<![CDATA')
            value = value.replace(']]', ']]>')
            outfile.write(value)
        else:
            outfile.write(quote_xml('%s' % self.valueOf_))

    def hasContent_(self):
        if False:
            print('Hello World!')
        if self.valueOf_ is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='incType'):
        if False:
            i = 10
            return i + 15
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            for i in range(10):
                print('nop')
        if self.local is not None:
            showIndent(outfile, level)
            outfile.write('local = "%s",\n' % (self.local,))
        if self.refid is not None:
            showIndent(outfile, level)
            outfile.write('refid = %s,\n' % (self.refid,))

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            print('Hello World!')
        showIndent(outfile, level)
        outfile.write('valueOf_ = "%s",\n' % (self.valueOf_,))

    def build(self, node_):
        if False:
            print('Hello World!')
        attrs = node_.attributes
        self.buildAttributes(attrs)
        self.valueOf_ = ''
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            print('Hello World!')
        if attrs.get('local'):
            self.local = attrs.get('local').value
        if attrs.get('refid'):
            self.refid = attrs.get('refid').value

    def buildChildren(self, child_, nodeName_):
        if False:
            i = 10
            return i + 15
        if child_.nodeType == Node.TEXT_NODE:
            obj_ = self.mixedclass_(MixedContainer.CategoryText, MixedContainer.TypeNone, '', child_.nodeValue)
            self.content_.append(obj_)
        if child_.nodeType == Node.TEXT_NODE:
            self.valueOf_ += child_.nodeValue
        elif child_.nodeType == Node.CDATA_SECTION_NODE:
            self.valueOf_ += '![CDATA[' + child_.nodeValue + ']]'

class refType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, prot=None, refid=None, valueOf_='', mixedclass_=None, content_=None):
        if False:
            for i in range(10):
                print('nop')
        self.prot = prot
        self.refid = refid
        if mixedclass_ is None:
            self.mixedclass_ = MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        if content_ is None:
            self.content_ = []
        else:
            self.content_ = content_

    def factory(*args_, **kwargs_):
        if False:
            for i in range(10):
                print('nop')
        if refType.subclass:
            return refType.subclass(*args_, **kwargs_)
        else:
            return refType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_prot(self):
        if False:
            print('Hello World!')
        return self.prot

    def set_prot(self, prot):
        if False:
            return 10
        self.prot = prot

    def get_refid(self):
        if False:
            for i in range(10):
                print('nop')
        return self.refid

    def set_refid(self, refid):
        if False:
            i = 10
            return i + 15
        self.refid = refid

    def getValueOf_(self):
        if False:
            for i in range(10):
                print('nop')
        return self.valueOf_

    def setValueOf_(self, valueOf_):
        if False:
            for i in range(10):
                print('nop')
        self.valueOf_ = valueOf_

    def export(self, outfile, level, namespace_='', name_='refType', namespacedef_=''):
        if False:
            i = 10
            return i + 15
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='refType')
        outfile.write('>')
        self.exportChildren(outfile, level + 1, namespace_, name_)
        outfile.write('</%s%s>\n' % (namespace_, name_))

    def exportAttributes(self, outfile, level, namespace_='', name_='refType'):
        if False:
            while True:
                i = 10
        if self.prot is not None:
            outfile.write(' prot=%s' % (quote_attrib(self.prot),))
        if self.refid is not None:
            outfile.write(' refid=%s' % (self.format_string(quote_attrib(self.refid).encode(ExternalEncoding), input_name='refid'),))

    def exportChildren(self, outfile, level, namespace_='', name_='refType'):
        if False:
            i = 10
            return i + 15
        if self.valueOf_.find('![CDATA') > -1:
            value = quote_xml('%s' % self.valueOf_)
            value = value.replace('![CDATA', '<![CDATA')
            value = value.replace(']]', ']]>')
            outfile.write(value)
        else:
            outfile.write(quote_xml('%s' % self.valueOf_))

    def hasContent_(self):
        if False:
            return 10
        if self.valueOf_ is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='refType'):
        if False:
            for i in range(10):
                print('nop')
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            i = 10
            return i + 15
        if self.prot is not None:
            showIndent(outfile, level)
            outfile.write('prot = "%s",\n' % (self.prot,))
        if self.refid is not None:
            showIndent(outfile, level)
            outfile.write('refid = %s,\n' % (self.refid,))

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            return 10
        showIndent(outfile, level)
        outfile.write('valueOf_ = "%s",\n' % (self.valueOf_,))

    def build(self, node_):
        if False:
            return 10
        attrs = node_.attributes
        self.buildAttributes(attrs)
        self.valueOf_ = ''
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            while True:
                i = 10
        if attrs.get('prot'):
            self.prot = attrs.get('prot').value
        if attrs.get('refid'):
            self.refid = attrs.get('refid').value

    def buildChildren(self, child_, nodeName_):
        if False:
            i = 10
            return i + 15
        if child_.nodeType == Node.TEXT_NODE:
            obj_ = self.mixedclass_(MixedContainer.CategoryText, MixedContainer.TypeNone, '', child_.nodeValue)
            self.content_.append(obj_)
        if child_.nodeType == Node.TEXT_NODE:
            self.valueOf_ += child_.nodeValue
        elif child_.nodeType == Node.CDATA_SECTION_NODE:
            self.valueOf_ += '![CDATA[' + child_.nodeValue + ']]'

class refTextType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, refid=None, kindref=None, external=None, valueOf_='', mixedclass_=None, content_=None):
        if False:
            return 10
        self.refid = refid
        self.kindref = kindref
        self.external = external
        if mixedclass_ is None:
            self.mixedclass_ = MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        if content_ is None:
            self.content_ = []
        else:
            self.content_ = content_

    def factory(*args_, **kwargs_):
        if False:
            while True:
                i = 10
        if refTextType.subclass:
            return refTextType.subclass(*args_, **kwargs_)
        else:
            return refTextType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_refid(self):
        if False:
            i = 10
            return i + 15
        return self.refid

    def set_refid(self, refid):
        if False:
            while True:
                i = 10
        self.refid = refid

    def get_kindref(self):
        if False:
            print('Hello World!')
        return self.kindref

    def set_kindref(self, kindref):
        if False:
            print('Hello World!')
        self.kindref = kindref

    def get_external(self):
        if False:
            return 10
        return self.external

    def set_external(self, external):
        if False:
            return 10
        self.external = external

    def getValueOf_(self):
        if False:
            return 10
        return self.valueOf_

    def setValueOf_(self, valueOf_):
        if False:
            return 10
        self.valueOf_ = valueOf_

    def export(self, outfile, level, namespace_='', name_='refTextType', namespacedef_=''):
        if False:
            print('Hello World!')
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='refTextType')
        outfile.write('>')
        self.exportChildren(outfile, level + 1, namespace_, name_)
        outfile.write('</%s%s>\n' % (namespace_, name_))

    def exportAttributes(self, outfile, level, namespace_='', name_='refTextType'):
        if False:
            return 10
        if self.refid is not None:
            outfile.write(' refid=%s' % (self.format_string(quote_attrib(self.refid).encode(ExternalEncoding), input_name='refid'),))
        if self.kindref is not None:
            outfile.write(' kindref=%s' % (quote_attrib(self.kindref),))
        if self.external is not None:
            outfile.write(' external=%s' % (self.format_string(quote_attrib(self.external).encode(ExternalEncoding), input_name='external'),))

    def exportChildren(self, outfile, level, namespace_='', name_='refTextType'):
        if False:
            i = 10
            return i + 15
        if self.valueOf_.find('![CDATA') > -1:
            value = quote_xml('%s' % self.valueOf_)
            value = value.replace('![CDATA', '<![CDATA')
            value = value.replace(']]', ']]>')
            outfile.write(value)
        else:
            outfile.write(quote_xml('%s' % self.valueOf_))

    def hasContent_(self):
        if False:
            while True:
                i = 10
        if self.valueOf_ is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='refTextType'):
        if False:
            i = 10
            return i + 15
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            for i in range(10):
                print('nop')
        if self.refid is not None:
            showIndent(outfile, level)
            outfile.write('refid = %s,\n' % (self.refid,))
        if self.kindref is not None:
            showIndent(outfile, level)
            outfile.write('kindref = "%s",\n' % (self.kindref,))
        if self.external is not None:
            showIndent(outfile, level)
            outfile.write('external = %s,\n' % (self.external,))

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            print('Hello World!')
        showIndent(outfile, level)
        outfile.write('valueOf_ = "%s",\n' % (self.valueOf_,))

    def build(self, node_):
        if False:
            for i in range(10):
                print('nop')
        attrs = node_.attributes
        self.buildAttributes(attrs)
        self.valueOf_ = ''
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            while True:
                i = 10
        if attrs.get('refid'):
            self.refid = attrs.get('refid').value
        if attrs.get('kindref'):
            self.kindref = attrs.get('kindref').value
        if attrs.get('external'):
            self.external = attrs.get('external').value

    def buildChildren(self, child_, nodeName_):
        if False:
            return 10
        if child_.nodeType == Node.TEXT_NODE:
            obj_ = self.mixedclass_(MixedContainer.CategoryText, MixedContainer.TypeNone, '', child_.nodeValue)
            self.content_.append(obj_)
        if child_.nodeType == Node.TEXT_NODE:
            self.valueOf_ += child_.nodeValue
        elif child_.nodeType == Node.CDATA_SECTION_NODE:
            self.valueOf_ += '![CDATA[' + child_.nodeValue + ']]'

class sectiondefType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, kind=None, header=None, description=None, memberdef=None):
        if False:
            for i in range(10):
                print('nop')
        self.kind = kind
        self.header = header
        self.description = description
        if memberdef is None:
            self.memberdef = []
        else:
            self.memberdef = memberdef

    def factory(*args_, **kwargs_):
        if False:
            i = 10
            return i + 15
        if sectiondefType.subclass:
            return sectiondefType.subclass(*args_, **kwargs_)
        else:
            return sectiondefType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_header(self):
        if False:
            while True:
                i = 10
        return self.header

    def set_header(self, header):
        if False:
            while True:
                i = 10
        self.header = header

    def get_description(self):
        if False:
            print('Hello World!')
        return self.description

    def set_description(self, description):
        if False:
            for i in range(10):
                print('nop')
        self.description = description

    def get_memberdef(self):
        if False:
            while True:
                i = 10
        return self.memberdef

    def set_memberdef(self, memberdef):
        if False:
            i = 10
            return i + 15
        self.memberdef = memberdef

    def add_memberdef(self, value):
        if False:
            print('Hello World!')
        self.memberdef.append(value)

    def insert_memberdef(self, index, value):
        if False:
            while True:
                i = 10
        self.memberdef[index] = value

    def get_kind(self):
        if False:
            return 10
        return self.kind

    def set_kind(self, kind):
        if False:
            i = 10
            return i + 15
        self.kind = kind

    def export(self, outfile, level, namespace_='', name_='sectiondefType', namespacedef_=''):
        if False:
            print('Hello World!')
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='sectiondefType')
        if self.hasContent_():
            outfile.write('>\n')
            self.exportChildren(outfile, level + 1, namespace_, name_)
            showIndent(outfile, level)
            outfile.write('</%s%s>\n' % (namespace_, name_))
        else:
            outfile.write(' />\n')

    def exportAttributes(self, outfile, level, namespace_='', name_='sectiondefType'):
        if False:
            i = 10
            return i + 15
        if self.kind is not None:
            outfile.write(' kind=%s' % (quote_attrib(self.kind),))

    def exportChildren(self, outfile, level, namespace_='', name_='sectiondefType'):
        if False:
            while True:
                i = 10
        if self.header is not None:
            showIndent(outfile, level)
            outfile.write('<%sheader>%s</%sheader>\n' % (namespace_, self.format_string(quote_xml(self.header).encode(ExternalEncoding), input_name='header'), namespace_))
        if self.description:
            self.description.export(outfile, level, namespace_, name_='description')
        for memberdef_ in self.memberdef:
            memberdef_.export(outfile, level, namespace_, name_='memberdef')

    def hasContent_(self):
        if False:
            return 10
        if self.header is not None or self.description is not None or self.memberdef is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='sectiondefType'):
        if False:
            for i in range(10):
                print('nop')
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            print('Hello World!')
        if self.kind is not None:
            showIndent(outfile, level)
            outfile.write('kind = "%s",\n' % (self.kind,))

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            print('Hello World!')
        showIndent(outfile, level)
        outfile.write('header=%s,\n' % quote_python(self.header).encode(ExternalEncoding))
        if self.description:
            showIndent(outfile, level)
            outfile.write('description=model_.descriptionType(\n')
            self.description.exportLiteral(outfile, level, name_='description')
            showIndent(outfile, level)
            outfile.write('),\n')
        showIndent(outfile, level)
        outfile.write('memberdef=[\n')
        level += 1
        for memberdef in self.memberdef:
            showIndent(outfile, level)
            outfile.write('model_.memberdef(\n')
            memberdef.exportLiteral(outfile, level, name_='memberdef')
            showIndent(outfile, level)
            outfile.write('),\n')
        level -= 1
        showIndent(outfile, level)
        outfile.write('],\n')

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
            i = 10
            return i + 15
        if attrs.get('kind'):
            self.kind = attrs.get('kind').value

    def buildChildren(self, child_, nodeName_):
        if False:
            return 10
        if child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'header':
            header_ = ''
            for text__content_ in child_.childNodes:
                header_ += text__content_.nodeValue
            self.header = header_
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'description':
            obj_ = descriptionType.factory()
            obj_.build(child_)
            self.set_description(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'memberdef':
            obj_ = memberdefType.factory()
            obj_.build(child_)
            self.memberdef.append(obj_)

class memberdefType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, initonly=None, kind=None, volatile=None, const=None, raisexx=None, virt=None, readable=None, prot=None, explicit=None, new=None, final=None, writable=None, add=None, static=None, remove=None, sealed=None, mutable=None, gettable=None, inline=None, settable=None, id=None, templateparamlist=None, type_=None, definition=None, argsstring=None, name=None, read=None, write=None, bitfield=None, reimplements=None, reimplementedby=None, param=None, enumvalue=None, initializer=None, exceptions=None, briefdescription=None, detaileddescription=None, inbodydescription=None, location=None, references=None, referencedby=None):
        if False:
            i = 10
            return i + 15
        self.initonly = initonly
        self.kind = kind
        self.volatile = volatile
        self.const = const
        self.raisexx = raisexx
        self.virt = virt
        self.readable = readable
        self.prot = prot
        self.explicit = explicit
        self.new = new
        self.final = final
        self.writable = writable
        self.add = add
        self.static = static
        self.remove = remove
        self.sealed = sealed
        self.mutable = mutable
        self.gettable = gettable
        self.inline = inline
        self.settable = settable
        self.id = id
        self.templateparamlist = templateparamlist
        self.type_ = type_
        self.definition = definition
        self.argsstring = argsstring
        self.name = name
        self.read = read
        self.write = write
        self.bitfield = bitfield
        if reimplements is None:
            self.reimplements = []
        else:
            self.reimplements = reimplements
        if reimplementedby is None:
            self.reimplementedby = []
        else:
            self.reimplementedby = reimplementedby
        if param is None:
            self.param = []
        else:
            self.param = param
        if enumvalue is None:
            self.enumvalue = []
        else:
            self.enumvalue = enumvalue
        self.initializer = initializer
        self.exceptions = exceptions
        self.briefdescription = briefdescription
        self.detaileddescription = detaileddescription
        self.inbodydescription = inbodydescription
        self.location = location
        if references is None:
            self.references = []
        else:
            self.references = references
        if referencedby is None:
            self.referencedby = []
        else:
            self.referencedby = referencedby

    def factory(*args_, **kwargs_):
        if False:
            for i in range(10):
                print('nop')
        if memberdefType.subclass:
            return memberdefType.subclass(*args_, **kwargs_)
        else:
            return memberdefType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_templateparamlist(self):
        if False:
            print('Hello World!')
        return self.templateparamlist

    def set_templateparamlist(self, templateparamlist):
        if False:
            print('Hello World!')
        self.templateparamlist = templateparamlist

    def get_type(self):
        if False:
            while True:
                i = 10
        return self.type_

    def set_type(self, type_):
        if False:
            for i in range(10):
                print('nop')
        self.type_ = type_

    def get_definition(self):
        if False:
            print('Hello World!')
        return self.definition

    def set_definition(self, definition):
        if False:
            print('Hello World!')
        self.definition = definition

    def get_argsstring(self):
        if False:
            while True:
                i = 10
        return self.argsstring

    def set_argsstring(self, argsstring):
        if False:
            return 10
        self.argsstring = argsstring

    def get_name(self):
        if False:
            while True:
                i = 10
        return self.name

    def set_name(self, name):
        if False:
            for i in range(10):
                print('nop')
        self.name = name

    def get_read(self):
        if False:
            while True:
                i = 10
        return self.read

    def set_read(self, read):
        if False:
            return 10
        self.read = read

    def get_write(self):
        if False:
            i = 10
            return i + 15
        return self.write

    def set_write(self, write):
        if False:
            return 10
        self.write = write

    def get_bitfield(self):
        if False:
            while True:
                i = 10
        return self.bitfield

    def set_bitfield(self, bitfield):
        if False:
            while True:
                i = 10
        self.bitfield = bitfield

    def get_reimplements(self):
        if False:
            for i in range(10):
                print('nop')
        return self.reimplements

    def set_reimplements(self, reimplements):
        if False:
            while True:
                i = 10
        self.reimplements = reimplements

    def add_reimplements(self, value):
        if False:
            print('Hello World!')
        self.reimplements.append(value)

    def insert_reimplements(self, index, value):
        if False:
            for i in range(10):
                print('nop')
        self.reimplements[index] = value

    def get_reimplementedby(self):
        if False:
            return 10
        return self.reimplementedby

    def set_reimplementedby(self, reimplementedby):
        if False:
            print('Hello World!')
        self.reimplementedby = reimplementedby

    def add_reimplementedby(self, value):
        if False:
            while True:
                i = 10
        self.reimplementedby.append(value)

    def insert_reimplementedby(self, index, value):
        if False:
            print('Hello World!')
        self.reimplementedby[index] = value

    def get_param(self):
        if False:
            while True:
                i = 10
        return self.param

    def set_param(self, param):
        if False:
            while True:
                i = 10
        self.param = param

    def add_param(self, value):
        if False:
            i = 10
            return i + 15
        self.param.append(value)

    def insert_param(self, index, value):
        if False:
            return 10
        self.param[index] = value

    def get_enumvalue(self):
        if False:
            i = 10
            return i + 15
        return self.enumvalue

    def set_enumvalue(self, enumvalue):
        if False:
            for i in range(10):
                print('nop')
        self.enumvalue = enumvalue

    def add_enumvalue(self, value):
        if False:
            while True:
                i = 10
        self.enumvalue.append(value)

    def insert_enumvalue(self, index, value):
        if False:
            return 10
        self.enumvalue[index] = value

    def get_initializer(self):
        if False:
            while True:
                i = 10
        return self.initializer

    def set_initializer(self, initializer):
        if False:
            while True:
                i = 10
        self.initializer = initializer

    def get_exceptions(self):
        if False:
            return 10
        return self.exceptions

    def set_exceptions(self, exceptions):
        if False:
            print('Hello World!')
        self.exceptions = exceptions

    def get_briefdescription(self):
        if False:
            i = 10
            return i + 15
        return self.briefdescription

    def set_briefdescription(self, briefdescription):
        if False:
            print('Hello World!')
        self.briefdescription = briefdescription

    def get_detaileddescription(self):
        if False:
            return 10
        return self.detaileddescription

    def set_detaileddescription(self, detaileddescription):
        if False:
            i = 10
            return i + 15
        self.detaileddescription = detaileddescription

    def get_inbodydescription(self):
        if False:
            while True:
                i = 10
        return self.inbodydescription

    def set_inbodydescription(self, inbodydescription):
        if False:
            i = 10
            return i + 15
        self.inbodydescription = inbodydescription

    def get_location(self):
        if False:
            i = 10
            return i + 15
        return self.location

    def set_location(self, location):
        if False:
            return 10
        self.location = location

    def get_references(self):
        if False:
            while True:
                i = 10
        return self.references

    def set_references(self, references):
        if False:
            print('Hello World!')
        self.references = references

    def add_references(self, value):
        if False:
            i = 10
            return i + 15
        self.references.append(value)

    def insert_references(self, index, value):
        if False:
            i = 10
            return i + 15
        self.references[index] = value

    def get_referencedby(self):
        if False:
            i = 10
            return i + 15
        return self.referencedby

    def set_referencedby(self, referencedby):
        if False:
            for i in range(10):
                print('nop')
        self.referencedby = referencedby

    def add_referencedby(self, value):
        if False:
            print('Hello World!')
        self.referencedby.append(value)

    def insert_referencedby(self, index, value):
        if False:
            print('Hello World!')
        self.referencedby[index] = value

    def get_initonly(self):
        if False:
            while True:
                i = 10
        return self.initonly

    def set_initonly(self, initonly):
        if False:
            i = 10
            return i + 15
        self.initonly = initonly

    def get_kind(self):
        if False:
            print('Hello World!')
        return self.kind

    def set_kind(self, kind):
        if False:
            while True:
                i = 10
        self.kind = kind

    def get_volatile(self):
        if False:
            while True:
                i = 10
        return self.volatile

    def set_volatile(self, volatile):
        if False:
            return 10
        self.volatile = volatile

    def get_const(self):
        if False:
            while True:
                i = 10
        return self.const

    def set_const(self, const):
        if False:
            print('Hello World!')
        self.const = const

    def get_raise(self):
        if False:
            while True:
                i = 10
        return self.raisexx

    def set_raise(self, raisexx):
        if False:
            print('Hello World!')
        self.raisexx = raisexx

    def get_virt(self):
        if False:
            return 10
        return self.virt

    def set_virt(self, virt):
        if False:
            i = 10
            return i + 15
        self.virt = virt

    def get_readable(self):
        if False:
            i = 10
            return i + 15
        return self.readable

    def set_readable(self, readable):
        if False:
            i = 10
            return i + 15
        self.readable = readable

    def get_prot(self):
        if False:
            print('Hello World!')
        return self.prot

    def set_prot(self, prot):
        if False:
            i = 10
            return i + 15
        self.prot = prot

    def get_explicit(self):
        if False:
            print('Hello World!')
        return self.explicit

    def set_explicit(self, explicit):
        if False:
            for i in range(10):
                print('nop')
        self.explicit = explicit

    def get_new(self):
        if False:
            while True:
                i = 10
        return self.new

    def set_new(self, new):
        if False:
            for i in range(10):
                print('nop')
        self.new = new

    def get_final(self):
        if False:
            return 10
        return self.final

    def set_final(self, final):
        if False:
            return 10
        self.final = final

    def get_writable(self):
        if False:
            for i in range(10):
                print('nop')
        return self.writable

    def set_writable(self, writable):
        if False:
            return 10
        self.writable = writable

    def get_add(self):
        if False:
            return 10
        return self.add

    def set_add(self, add):
        if False:
            print('Hello World!')
        self.add = add

    def get_static(self):
        if False:
            print('Hello World!')
        return self.static

    def set_static(self, static):
        if False:
            for i in range(10):
                print('nop')
        self.static = static

    def get_remove(self):
        if False:
            print('Hello World!')
        return self.remove

    def set_remove(self, remove):
        if False:
            print('Hello World!')
        self.remove = remove

    def get_sealed(self):
        if False:
            print('Hello World!')
        return self.sealed

    def set_sealed(self, sealed):
        if False:
            return 10
        self.sealed = sealed

    def get_mutable(self):
        if False:
            return 10
        return self.mutable

    def set_mutable(self, mutable):
        if False:
            return 10
        self.mutable = mutable

    def get_gettable(self):
        if False:
            return 10
        return self.gettable

    def set_gettable(self, gettable):
        if False:
            print('Hello World!')
        self.gettable = gettable

    def get_inline(self):
        if False:
            return 10
        return self.inline

    def set_inline(self, inline):
        if False:
            i = 10
            return i + 15
        self.inline = inline

    def get_settable(self):
        if False:
            i = 10
            return i + 15
        return self.settable

    def set_settable(self, settable):
        if False:
            print('Hello World!')
        self.settable = settable

    def get_id(self):
        if False:
            print('Hello World!')
        return self.id

    def set_id(self, id):
        if False:
            while True:
                i = 10
        self.id = id

    def export(self, outfile, level, namespace_='', name_='memberdefType', namespacedef_=''):
        if False:
            i = 10
            return i + 15
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='memberdefType')
        if self.hasContent_():
            outfile.write('>\n')
            self.exportChildren(outfile, level + 1, namespace_, name_)
            showIndent(outfile, level)
            outfile.write('</%s%s>\n' % (namespace_, name_))
        else:
            outfile.write(' />\n')

    def exportAttributes(self, outfile, level, namespace_='', name_='memberdefType'):
        if False:
            i = 10
            return i + 15
        if self.initonly is not None:
            outfile.write(' initonly=%s' % (quote_attrib(self.initonly),))
        if self.kind is not None:
            outfile.write(' kind=%s' % (quote_attrib(self.kind),))
        if self.volatile is not None:
            outfile.write(' volatile=%s' % (quote_attrib(self.volatile),))
        if self.const is not None:
            outfile.write(' const=%s' % (quote_attrib(self.const),))
        if self.raisexx is not None:
            outfile.write(' raise=%s' % (quote_attrib(self.raisexx),))
        if self.virt is not None:
            outfile.write(' virt=%s' % (quote_attrib(self.virt),))
        if self.readable is not None:
            outfile.write(' readable=%s' % (quote_attrib(self.readable),))
        if self.prot is not None:
            outfile.write(' prot=%s' % (quote_attrib(self.prot),))
        if self.explicit is not None:
            outfile.write(' explicit=%s' % (quote_attrib(self.explicit),))
        if self.new is not None:
            outfile.write(' new=%s' % (quote_attrib(self.new),))
        if self.final is not None:
            outfile.write(' final=%s' % (quote_attrib(self.final),))
        if self.writable is not None:
            outfile.write(' writable=%s' % (quote_attrib(self.writable),))
        if self.add is not None:
            outfile.write(' add=%s' % (quote_attrib(self.add),))
        if self.static is not None:
            outfile.write(' static=%s' % (quote_attrib(self.static),))
        if self.remove is not None:
            outfile.write(' remove=%s' % (quote_attrib(self.remove),))
        if self.sealed is not None:
            outfile.write(' sealed=%s' % (quote_attrib(self.sealed),))
        if self.mutable is not None:
            outfile.write(' mutable=%s' % (quote_attrib(self.mutable),))
        if self.gettable is not None:
            outfile.write(' gettable=%s' % (quote_attrib(self.gettable),))
        if self.inline is not None:
            outfile.write(' inline=%s' % (quote_attrib(self.inline),))
        if self.settable is not None:
            outfile.write(' settable=%s' % (quote_attrib(self.settable),))
        if self.id is not None:
            outfile.write(' id=%s' % (self.format_string(quote_attrib(self.id).encode(ExternalEncoding), input_name='id'),))

    def exportChildren(self, outfile, level, namespace_='', name_='memberdefType'):
        if False:
            while True:
                i = 10
        if self.templateparamlist:
            self.templateparamlist.export(outfile, level, namespace_, name_='templateparamlist')
        if self.type_:
            self.type_.export(outfile, level, namespace_, name_='type')
        if self.definition is not None:
            showIndent(outfile, level)
            outfile.write('<%sdefinition>%s</%sdefinition>\n' % (namespace_, self.format_string(quote_xml(self.definition).encode(ExternalEncoding), input_name='definition'), namespace_))
        if self.argsstring is not None:
            showIndent(outfile, level)
            outfile.write('<%sargsstring>%s</%sargsstring>\n' % (namespace_, self.format_string(quote_xml(self.argsstring).encode(ExternalEncoding), input_name='argsstring'), namespace_))
        if self.name is not None:
            showIndent(outfile, level)
            outfile.write('<%sname>%s</%sname>\n' % (namespace_, self.format_string(quote_xml(self.name).encode(ExternalEncoding), input_name='name'), namespace_))
        if self.read is not None:
            showIndent(outfile, level)
            outfile.write('<%sread>%s</%sread>\n' % (namespace_, self.format_string(quote_xml(self.read).encode(ExternalEncoding), input_name='read'), namespace_))
        if self.write is not None:
            showIndent(outfile, level)
            outfile.write('<%swrite>%s</%swrite>\n' % (namespace_, self.format_string(quote_xml(self.write).encode(ExternalEncoding), input_name='write'), namespace_))
        if self.bitfield is not None:
            showIndent(outfile, level)
            outfile.write('<%sbitfield>%s</%sbitfield>\n' % (namespace_, self.format_string(quote_xml(self.bitfield).encode(ExternalEncoding), input_name='bitfield'), namespace_))
        for reimplements_ in self.reimplements:
            reimplements_.export(outfile, level, namespace_, name_='reimplements')
        for reimplementedby_ in self.reimplementedby:
            reimplementedby_.export(outfile, level, namespace_, name_='reimplementedby')
        for param_ in self.param:
            param_.export(outfile, level, namespace_, name_='param')
        for enumvalue_ in self.enumvalue:
            enumvalue_.export(outfile, level, namespace_, name_='enumvalue')
        if self.initializer:
            self.initializer.export(outfile, level, namespace_, name_='initializer')
        if self.exceptions:
            self.exceptions.export(outfile, level, namespace_, name_='exceptions')
        if self.briefdescription:
            self.briefdescription.export(outfile, level, namespace_, name_='briefdescription')
        if self.detaileddescription:
            self.detaileddescription.export(outfile, level, namespace_, name_='detaileddescription')
        if self.inbodydescription:
            self.inbodydescription.export(outfile, level, namespace_, name_='inbodydescription')
        if self.location:
            self.location.export(outfile, level, namespace_, name_='location')
        for references_ in self.references:
            references_.export(outfile, level, namespace_, name_='references')
        for referencedby_ in self.referencedby:
            referencedby_.export(outfile, level, namespace_, name_='referencedby')

    def hasContent_(self):
        if False:
            print('Hello World!')
        if self.templateparamlist is not None or self.type_ is not None or self.definition is not None or (self.argsstring is not None) or (self.name is not None) or (self.read is not None) or (self.write is not None) or (self.bitfield is not None) or (self.reimplements is not None) or (self.reimplementedby is not None) or (self.param is not None) or (self.enumvalue is not None) or (self.initializer is not None) or (self.exceptions is not None) or (self.briefdescription is not None) or (self.detaileddescription is not None) or (self.inbodydescription is not None) or (self.location is not None) or (self.references is not None) or (self.referencedby is not None):
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='memberdefType'):
        if False:
            i = 10
            return i + 15
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            return 10
        if self.initonly is not None:
            showIndent(outfile, level)
            outfile.write('initonly = "%s",\n' % (self.initonly,))
        if self.kind is not None:
            showIndent(outfile, level)
            outfile.write('kind = "%s",\n' % (self.kind,))
        if self.volatile is not None:
            showIndent(outfile, level)
            outfile.write('volatile = "%s",\n' % (self.volatile,))
        if self.const is not None:
            showIndent(outfile, level)
            outfile.write('const = "%s",\n' % (self.const,))
        if self.raisexx is not None:
            showIndent(outfile, level)
            outfile.write('raisexx = "%s",\n' % (self.raisexx,))
        if self.virt is not None:
            showIndent(outfile, level)
            outfile.write('virt = "%s",\n' % (self.virt,))
        if self.readable is not None:
            showIndent(outfile, level)
            outfile.write('readable = "%s",\n' % (self.readable,))
        if self.prot is not None:
            showIndent(outfile, level)
            outfile.write('prot = "%s",\n' % (self.prot,))
        if self.explicit is not None:
            showIndent(outfile, level)
            outfile.write('explicit = "%s",\n' % (self.explicit,))
        if self.new is not None:
            showIndent(outfile, level)
            outfile.write('new = "%s",\n' % (self.new,))
        if self.final is not None:
            showIndent(outfile, level)
            outfile.write('final = "%s",\n' % (self.final,))
        if self.writable is not None:
            showIndent(outfile, level)
            outfile.write('writable = "%s",\n' % (self.writable,))
        if self.add is not None:
            showIndent(outfile, level)
            outfile.write('add = "%s",\n' % (self.add,))
        if self.static is not None:
            showIndent(outfile, level)
            outfile.write('static = "%s",\n' % (self.static,))
        if self.remove is not None:
            showIndent(outfile, level)
            outfile.write('remove = "%s",\n' % (self.remove,))
        if self.sealed is not None:
            showIndent(outfile, level)
            outfile.write('sealed = "%s",\n' % (self.sealed,))
        if self.mutable is not None:
            showIndent(outfile, level)
            outfile.write('mutable = "%s",\n' % (self.mutable,))
        if self.gettable is not None:
            showIndent(outfile, level)
            outfile.write('gettable = "%s",\n' % (self.gettable,))
        if self.inline is not None:
            showIndent(outfile, level)
            outfile.write('inline = "%s",\n' % (self.inline,))
        if self.settable is not None:
            showIndent(outfile, level)
            outfile.write('settable = "%s",\n' % (self.settable,))
        if self.id is not None:
            showIndent(outfile, level)
            outfile.write('id = %s,\n' % (self.id,))

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            i = 10
            return i + 15
        if self.templateparamlist:
            showIndent(outfile, level)
            outfile.write('templateparamlist=model_.templateparamlistType(\n')
            self.templateparamlist.exportLiteral(outfile, level, name_='templateparamlist')
            showIndent(outfile, level)
            outfile.write('),\n')
        if self.type_:
            showIndent(outfile, level)
            outfile.write('type_=model_.linkedTextType(\n')
            self.type_.exportLiteral(outfile, level, name_='type')
            showIndent(outfile, level)
            outfile.write('),\n')
        showIndent(outfile, level)
        outfile.write('definition=%s,\n' % quote_python(self.definition).encode(ExternalEncoding))
        showIndent(outfile, level)
        outfile.write('argsstring=%s,\n' % quote_python(self.argsstring).encode(ExternalEncoding))
        showIndent(outfile, level)
        outfile.write('name=%s,\n' % quote_python(self.name).encode(ExternalEncoding))
        showIndent(outfile, level)
        outfile.write('read=%s,\n' % quote_python(self.read).encode(ExternalEncoding))
        showIndent(outfile, level)
        outfile.write('write=%s,\n' % quote_python(self.write).encode(ExternalEncoding))
        showIndent(outfile, level)
        outfile.write('bitfield=%s,\n' % quote_python(self.bitfield).encode(ExternalEncoding))
        showIndent(outfile, level)
        outfile.write('reimplements=[\n')
        level += 1
        for reimplements in self.reimplements:
            showIndent(outfile, level)
            outfile.write('model_.reimplements(\n')
            reimplements.exportLiteral(outfile, level, name_='reimplements')
            showIndent(outfile, level)
            outfile.write('),\n')
        level -= 1
        showIndent(outfile, level)
        outfile.write('],\n')
        showIndent(outfile, level)
        outfile.write('reimplementedby=[\n')
        level += 1
        for reimplementedby in self.reimplementedby:
            showIndent(outfile, level)
            outfile.write('model_.reimplementedby(\n')
            reimplementedby.exportLiteral(outfile, level, name_='reimplementedby')
            showIndent(outfile, level)
            outfile.write('),\n')
        level -= 1
        showIndent(outfile, level)
        outfile.write('],\n')
        showIndent(outfile, level)
        outfile.write('param=[\n')
        level += 1
        for param in self.param:
            showIndent(outfile, level)
            outfile.write('model_.param(\n')
            param.exportLiteral(outfile, level, name_='param')
            showIndent(outfile, level)
            outfile.write('),\n')
        level -= 1
        showIndent(outfile, level)
        outfile.write('],\n')
        showIndent(outfile, level)
        outfile.write('enumvalue=[\n')
        level += 1
        for enumvalue in self.enumvalue:
            showIndent(outfile, level)
            outfile.write('model_.enumvalue(\n')
            enumvalue.exportLiteral(outfile, level, name_='enumvalue')
            showIndent(outfile, level)
            outfile.write('),\n')
        level -= 1
        showIndent(outfile, level)
        outfile.write('],\n')
        if self.initializer:
            showIndent(outfile, level)
            outfile.write('initializer=model_.linkedTextType(\n')
            self.initializer.exportLiteral(outfile, level, name_='initializer')
            showIndent(outfile, level)
            outfile.write('),\n')
        if self.exceptions:
            showIndent(outfile, level)
            outfile.write('exceptions=model_.linkedTextType(\n')
            self.exceptions.exportLiteral(outfile, level, name_='exceptions')
            showIndent(outfile, level)
            outfile.write('),\n')
        if self.briefdescription:
            showIndent(outfile, level)
            outfile.write('briefdescription=model_.descriptionType(\n')
            self.briefdescription.exportLiteral(outfile, level, name_='briefdescription')
            showIndent(outfile, level)
            outfile.write('),\n')
        if self.detaileddescription:
            showIndent(outfile, level)
            outfile.write('detaileddescription=model_.descriptionType(\n')
            self.detaileddescription.exportLiteral(outfile, level, name_='detaileddescription')
            showIndent(outfile, level)
            outfile.write('),\n')
        if self.inbodydescription:
            showIndent(outfile, level)
            outfile.write('inbodydescription=model_.descriptionType(\n')
            self.inbodydescription.exportLiteral(outfile, level, name_='inbodydescription')
            showIndent(outfile, level)
            outfile.write('),\n')
        if self.location:
            showIndent(outfile, level)
            outfile.write('location=model_.locationType(\n')
            self.location.exportLiteral(outfile, level, name_='location')
            showIndent(outfile, level)
            outfile.write('),\n')
        showIndent(outfile, level)
        outfile.write('references=[\n')
        level += 1
        for references in self.references:
            showIndent(outfile, level)
            outfile.write('model_.references(\n')
            references.exportLiteral(outfile, level, name_='references')
            showIndent(outfile, level)
            outfile.write('),\n')
        level -= 1
        showIndent(outfile, level)
        outfile.write('],\n')
        showIndent(outfile, level)
        outfile.write('referencedby=[\n')
        level += 1
        for referencedby in self.referencedby:
            showIndent(outfile, level)
            outfile.write('model_.referencedby(\n')
            referencedby.exportLiteral(outfile, level, name_='referencedby')
            showIndent(outfile, level)
            outfile.write('),\n')
        level -= 1
        showIndent(outfile, level)
        outfile.write('],\n')

    def build(self, node_):
        if False:
            return 10
        attrs = node_.attributes
        self.buildAttributes(attrs)
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            return 10
        if attrs.get('initonly'):
            self.initonly = attrs.get('initonly').value
        if attrs.get('kind'):
            self.kind = attrs.get('kind').value
        if attrs.get('volatile'):
            self.volatile = attrs.get('volatile').value
        if attrs.get('const'):
            self.const = attrs.get('const').value
        if attrs.get('raise'):
            self.raisexx = attrs.get('raise').value
        if attrs.get('virt'):
            self.virt = attrs.get('virt').value
        if attrs.get('readable'):
            self.readable = attrs.get('readable').value
        if attrs.get('prot'):
            self.prot = attrs.get('prot').value
        if attrs.get('explicit'):
            self.explicit = attrs.get('explicit').value
        if attrs.get('new'):
            self.new = attrs.get('new').value
        if attrs.get('final'):
            self.final = attrs.get('final').value
        if attrs.get('writable'):
            self.writable = attrs.get('writable').value
        if attrs.get('add'):
            self.add = attrs.get('add').value
        if attrs.get('static'):
            self.static = attrs.get('static').value
        if attrs.get('remove'):
            self.remove = attrs.get('remove').value
        if attrs.get('sealed'):
            self.sealed = attrs.get('sealed').value
        if attrs.get('mutable'):
            self.mutable = attrs.get('mutable').value
        if attrs.get('gettable'):
            self.gettable = attrs.get('gettable').value
        if attrs.get('inline'):
            self.inline = attrs.get('inline').value
        if attrs.get('settable'):
            self.settable = attrs.get('settable').value
        if attrs.get('id'):
            self.id = attrs.get('id').value

    def buildChildren(self, child_, nodeName_):
        if False:
            return 10
        if child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'templateparamlist':
            obj_ = templateparamlistType.factory()
            obj_.build(child_)
            self.set_templateparamlist(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'type':
            obj_ = linkedTextType.factory()
            obj_.build(child_)
            self.set_type(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'definition':
            definition_ = ''
            for text__content_ in child_.childNodes:
                definition_ += text__content_.nodeValue
            self.definition = definition_
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'argsstring':
            argsstring_ = ''
            for text__content_ in child_.childNodes:
                argsstring_ += text__content_.nodeValue
            self.argsstring = argsstring_
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'name':
            name_ = ''
            for text__content_ in child_.childNodes:
                name_ += text__content_.nodeValue
            self.name = name_
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'read':
            read_ = ''
            for text__content_ in child_.childNodes:
                read_ += text__content_.nodeValue
            self.read = read_
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'write':
            write_ = ''
            for text__content_ in child_.childNodes:
                write_ += text__content_.nodeValue
            self.write = write_
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'bitfield':
            bitfield_ = ''
            for text__content_ in child_.childNodes:
                bitfield_ += text__content_.nodeValue
            self.bitfield = bitfield_
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'reimplements':
            obj_ = reimplementType.factory()
            obj_.build(child_)
            self.reimplements.append(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'reimplementedby':
            obj_ = reimplementType.factory()
            obj_.build(child_)
            self.reimplementedby.append(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'param':
            obj_ = paramType.factory()
            obj_.build(child_)
            self.param.append(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'enumvalue':
            obj_ = enumvalueType.factory()
            obj_.build(child_)
            self.enumvalue.append(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'initializer':
            obj_ = linkedTextType.factory()
            obj_.build(child_)
            self.set_initializer(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'exceptions':
            obj_ = linkedTextType.factory()
            obj_.build(child_)
            self.set_exceptions(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'briefdescription':
            obj_ = descriptionType.factory()
            obj_.build(child_)
            self.set_briefdescription(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'detaileddescription':
            obj_ = descriptionType.factory()
            obj_.build(child_)
            self.set_detaileddescription(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'inbodydescription':
            obj_ = descriptionType.factory()
            obj_.build(child_)
            self.set_inbodydescription(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'location':
            obj_ = locationType.factory()
            obj_.build(child_)
            self.set_location(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'references':
            obj_ = referenceType.factory()
            obj_.build(child_)
            self.references.append(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'referencedby':
            obj_ = referenceType.factory()
            obj_.build(child_)
            self.referencedby.append(obj_)

class definition(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, valueOf_=''):
        if False:
            return 10
        self.valueOf_ = valueOf_

    def factory(*args_, **kwargs_):
        if False:
            print('Hello World!')
        if definition.subclass:
            return definition.subclass(*args_, **kwargs_)
        else:
            return definition(*args_, **kwargs_)
    factory = staticmethod(factory)

    def getValueOf_(self):
        if False:
            print('Hello World!')
        return self.valueOf_

    def setValueOf_(self, valueOf_):
        if False:
            return 10
        self.valueOf_ = valueOf_

    def export(self, outfile, level, namespace_='', name_='definition', namespacedef_=''):
        if False:
            while True:
                i = 10
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='definition')
        if self.hasContent_():
            outfile.write('>\n')
            self.exportChildren(outfile, level + 1, namespace_, name_)
            showIndent(outfile, level)
            outfile.write('</%s%s>\n' % (namespace_, name_))
        else:
            outfile.write(' />\n')

    def exportAttributes(self, outfile, level, namespace_='', name_='definition'):
        if False:
            return 10
        pass

    def exportChildren(self, outfile, level, namespace_='', name_='definition'):
        if False:
            while True:
                i = 10
        if self.valueOf_.find('![CDATA') > -1:
            value = quote_xml('%s' % self.valueOf_)
            value = value.replace('![CDATA', '<![CDATA')
            value = value.replace(']]', ']]>')
            outfile.write(value)
        else:
            outfile.write(quote_xml('%s' % self.valueOf_))

    def hasContent_(self):
        if False:
            print('Hello World!')
        if self.valueOf_ is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='definition'):
        if False:
            for i in range(10):
                print('nop')
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            for i in range(10):
                print('nop')
        pass

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            i = 10
            return i + 15
        showIndent(outfile, level)
        outfile.write('valueOf_ = "%s",\n' % (self.valueOf_,))

    def build(self, node_):
        if False:
            i = 10
            return i + 15
        attrs = node_.attributes
        self.buildAttributes(attrs)
        self.valueOf_ = ''
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            print('Hello World!')
        pass

    def buildChildren(self, child_, nodeName_):
        if False:
            for i in range(10):
                print('nop')
        if child_.nodeType == Node.TEXT_NODE:
            self.valueOf_ += child_.nodeValue
        elif child_.nodeType == Node.CDATA_SECTION_NODE:
            self.valueOf_ += '![CDATA[' + child_.nodeValue + ']]'

class argsstring(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, valueOf_=''):
        if False:
            i = 10
            return i + 15
        self.valueOf_ = valueOf_

    def factory(*args_, **kwargs_):
        if False:
            for i in range(10):
                print('nop')
        if argsstring.subclass:
            return argsstring.subclass(*args_, **kwargs_)
        else:
            return argsstring(*args_, **kwargs_)
    factory = staticmethod(factory)

    def getValueOf_(self):
        if False:
            print('Hello World!')
        return self.valueOf_

    def setValueOf_(self, valueOf_):
        if False:
            return 10
        self.valueOf_ = valueOf_

    def export(self, outfile, level, namespace_='', name_='argsstring', namespacedef_=''):
        if False:
            while True:
                i = 10
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='argsstring')
        if self.hasContent_():
            outfile.write('>\n')
            self.exportChildren(outfile, level + 1, namespace_, name_)
            showIndent(outfile, level)
            outfile.write('</%s%s>\n' % (namespace_, name_))
        else:
            outfile.write(' />\n')

    def exportAttributes(self, outfile, level, namespace_='', name_='argsstring'):
        if False:
            print('Hello World!')
        pass

    def exportChildren(self, outfile, level, namespace_='', name_='argsstring'):
        if False:
            for i in range(10):
                print('nop')
        if self.valueOf_.find('![CDATA') > -1:
            value = quote_xml('%s' % self.valueOf_)
            value = value.replace('![CDATA', '<![CDATA')
            value = value.replace(']]', ']]>')
            outfile.write(value)
        else:
            outfile.write(quote_xml('%s' % self.valueOf_))

    def hasContent_(self):
        if False:
            while True:
                i = 10
        if self.valueOf_ is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='argsstring'):
        if False:
            print('Hello World!')
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            for i in range(10):
                print('nop')
        pass

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            while True:
                i = 10
        showIndent(outfile, level)
        outfile.write('valueOf_ = "%s",\n' % (self.valueOf_,))

    def build(self, node_):
        if False:
            print('Hello World!')
        attrs = node_.attributes
        self.buildAttributes(attrs)
        self.valueOf_ = ''
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            print('Hello World!')
        pass

    def buildChildren(self, child_, nodeName_):
        if False:
            i = 10
            return i + 15
        if child_.nodeType == Node.TEXT_NODE:
            self.valueOf_ += child_.nodeValue
        elif child_.nodeType == Node.CDATA_SECTION_NODE:
            self.valueOf_ += '![CDATA[' + child_.nodeValue + ']]'

class read(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, valueOf_=''):
        if False:
            for i in range(10):
                print('nop')
        self.valueOf_ = valueOf_

    def factory(*args_, **kwargs_):
        if False:
            while True:
                i = 10
        if read.subclass:
            return read.subclass(*args_, **kwargs_)
        else:
            return read(*args_, **kwargs_)
    factory = staticmethod(factory)

    def getValueOf_(self):
        if False:
            i = 10
            return i + 15
        return self.valueOf_

    def setValueOf_(self, valueOf_):
        if False:
            i = 10
            return i + 15
        self.valueOf_ = valueOf_

    def export(self, outfile, level, namespace_='', name_='read', namespacedef_=''):
        if False:
            i = 10
            return i + 15
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='read')
        if self.hasContent_():
            outfile.write('>\n')
            self.exportChildren(outfile, level + 1, namespace_, name_)
            showIndent(outfile, level)
            outfile.write('</%s%s>\n' % (namespace_, name_))
        else:
            outfile.write(' />\n')

    def exportAttributes(self, outfile, level, namespace_='', name_='read'):
        if False:
            return 10
        pass

    def exportChildren(self, outfile, level, namespace_='', name_='read'):
        if False:
            i = 10
            return i + 15
        if self.valueOf_.find('![CDATA') > -1:
            value = quote_xml('%s' % self.valueOf_)
            value = value.replace('![CDATA', '<![CDATA')
            value = value.replace(']]', ']]>')
            outfile.write(value)
        else:
            outfile.write(quote_xml('%s' % self.valueOf_))

    def hasContent_(self):
        if False:
            for i in range(10):
                print('nop')
        if self.valueOf_ is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='read'):
        if False:
            for i in range(10):
                print('nop')
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            print('Hello World!')
        pass

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            print('Hello World!')
        showIndent(outfile, level)
        outfile.write('valueOf_ = "%s",\n' % (self.valueOf_,))

    def build(self, node_):
        if False:
            return 10
        attrs = node_.attributes
        self.buildAttributes(attrs)
        self.valueOf_ = ''
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            return 10
        pass

    def buildChildren(self, child_, nodeName_):
        if False:
            print('Hello World!')
        if child_.nodeType == Node.TEXT_NODE:
            self.valueOf_ += child_.nodeValue
        elif child_.nodeType == Node.CDATA_SECTION_NODE:
            self.valueOf_ += '![CDATA[' + child_.nodeValue + ']]'

class write(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, valueOf_=''):
        if False:
            while True:
                i = 10
        self.valueOf_ = valueOf_

    def factory(*args_, **kwargs_):
        if False:
            while True:
                i = 10
        if write.subclass:
            return write.subclass(*args_, **kwargs_)
        else:
            return write(*args_, **kwargs_)
    factory = staticmethod(factory)

    def getValueOf_(self):
        if False:
            i = 10
            return i + 15
        return self.valueOf_

    def setValueOf_(self, valueOf_):
        if False:
            print('Hello World!')
        self.valueOf_ = valueOf_

    def export(self, outfile, level, namespace_='', name_='write', namespacedef_=''):
        if False:
            print('Hello World!')
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='write')
        if self.hasContent_():
            outfile.write('>\n')
            self.exportChildren(outfile, level + 1, namespace_, name_)
            showIndent(outfile, level)
            outfile.write('</%s%s>\n' % (namespace_, name_))
        else:
            outfile.write(' />\n')

    def exportAttributes(self, outfile, level, namespace_='', name_='write'):
        if False:
            for i in range(10):
                print('nop')
        pass

    def exportChildren(self, outfile, level, namespace_='', name_='write'):
        if False:
            for i in range(10):
                print('nop')
        if self.valueOf_.find('![CDATA') > -1:
            value = quote_xml('%s' % self.valueOf_)
            value = value.replace('![CDATA', '<![CDATA')
            value = value.replace(']]', ']]>')
            outfile.write(value)
        else:
            outfile.write(quote_xml('%s' % self.valueOf_))

    def hasContent_(self):
        if False:
            for i in range(10):
                print('nop')
        if self.valueOf_ is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='write'):
        if False:
            return 10
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            while True:
                i = 10
        pass

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            print('Hello World!')
        showIndent(outfile, level)
        outfile.write('valueOf_ = "%s",\n' % (self.valueOf_,))

    def build(self, node_):
        if False:
            for i in range(10):
                print('nop')
        attrs = node_.attributes
        self.buildAttributes(attrs)
        self.valueOf_ = ''
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            return 10
        pass

    def buildChildren(self, child_, nodeName_):
        if False:
            print('Hello World!')
        if child_.nodeType == Node.TEXT_NODE:
            self.valueOf_ += child_.nodeValue
        elif child_.nodeType == Node.CDATA_SECTION_NODE:
            self.valueOf_ += '![CDATA[' + child_.nodeValue + ']]'

class bitfield(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, valueOf_=''):
        if False:
            i = 10
            return i + 15
        self.valueOf_ = valueOf_

    def factory(*args_, **kwargs_):
        if False:
            return 10
        if bitfield.subclass:
            return bitfield.subclass(*args_, **kwargs_)
        else:
            return bitfield(*args_, **kwargs_)
    factory = staticmethod(factory)

    def getValueOf_(self):
        if False:
            i = 10
            return i + 15
        return self.valueOf_

    def setValueOf_(self, valueOf_):
        if False:
            return 10
        self.valueOf_ = valueOf_

    def export(self, outfile, level, namespace_='', name_='bitfield', namespacedef_=''):
        if False:
            while True:
                i = 10
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='bitfield')
        if self.hasContent_():
            outfile.write('>\n')
            self.exportChildren(outfile, level + 1, namespace_, name_)
            showIndent(outfile, level)
            outfile.write('</%s%s>\n' % (namespace_, name_))
        else:
            outfile.write(' />\n')

    def exportAttributes(self, outfile, level, namespace_='', name_='bitfield'):
        if False:
            while True:
                i = 10
        pass

    def exportChildren(self, outfile, level, namespace_='', name_='bitfield'):
        if False:
            for i in range(10):
                print('nop')
        if self.valueOf_.find('![CDATA') > -1:
            value = quote_xml('%s' % self.valueOf_)
            value = value.replace('![CDATA', '<![CDATA')
            value = value.replace(']]', ']]>')
            outfile.write(value)
        else:
            outfile.write(quote_xml('%s' % self.valueOf_))

    def hasContent_(self):
        if False:
            while True:
                i = 10
        if self.valueOf_ is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='bitfield'):
        if False:
            print('Hello World!')
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            return 10
        pass

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            return 10
        showIndent(outfile, level)
        outfile.write('valueOf_ = "%s",\n' % (self.valueOf_,))

    def build(self, node_):
        if False:
            i = 10
            return i + 15
        attrs = node_.attributes
        self.buildAttributes(attrs)
        self.valueOf_ = ''
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            return 10
        pass

    def buildChildren(self, child_, nodeName_):
        if False:
            return 10
        if child_.nodeType == Node.TEXT_NODE:
            self.valueOf_ += child_.nodeValue
        elif child_.nodeType == Node.CDATA_SECTION_NODE:
            self.valueOf_ += '![CDATA[' + child_.nodeValue + ']]'

class descriptionType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, title=None, para=None, sect1=None, internal=None, mixedclass_=None, content_=None):
        if False:
            for i in range(10):
                print('nop')
        if mixedclass_ is None:
            self.mixedclass_ = MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        if content_ is None:
            self.content_ = []
        else:
            self.content_ = content_

    def factory(*args_, **kwargs_):
        if False:
            while True:
                i = 10
        if descriptionType.subclass:
            return descriptionType.subclass(*args_, **kwargs_)
        else:
            return descriptionType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_title(self):
        if False:
            i = 10
            return i + 15
        return self.title

    def set_title(self, title):
        if False:
            return 10
        self.title = title

    def get_para(self):
        if False:
            for i in range(10):
                print('nop')
        return self.para

    def set_para(self, para):
        if False:
            print('Hello World!')
        self.para = para

    def add_para(self, value):
        if False:
            while True:
                i = 10
        self.para.append(value)

    def insert_para(self, index, value):
        if False:
            i = 10
            return i + 15
        self.para[index] = value

    def get_sect1(self):
        if False:
            i = 10
            return i + 15
        return self.sect1

    def set_sect1(self, sect1):
        if False:
            print('Hello World!')
        self.sect1 = sect1

    def add_sect1(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.sect1.append(value)

    def insert_sect1(self, index, value):
        if False:
            for i in range(10):
                print('nop')
        self.sect1[index] = value

    def get_internal(self):
        if False:
            print('Hello World!')
        return self.internal

    def set_internal(self, internal):
        if False:
            i = 10
            return i + 15
        self.internal = internal

    def export(self, outfile, level, namespace_='', name_='descriptionType', namespacedef_=''):
        if False:
            print('Hello World!')
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='descriptionType')
        outfile.write('>')
        self.exportChildren(outfile, level + 1, namespace_, name_)
        outfile.write('</%s%s>\n' % (namespace_, name_))

    def exportAttributes(self, outfile, level, namespace_='', name_='descriptionType'):
        if False:
            while True:
                i = 10
        pass

    def exportChildren(self, outfile, level, namespace_='', name_='descriptionType'):
        if False:
            while True:
                i = 10
        for item_ in self.content_:
            item_.export(outfile, level, item_.name, namespace_)

    def hasContent_(self):
        if False:
            for i in range(10):
                print('nop')
        if self.title is not None or self.para is not None or self.sect1 is not None or (self.internal is not None):
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='descriptionType'):
        if False:
            i = 10
            return i + 15
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            return 10
        pass

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            i = 10
            return i + 15
        showIndent(outfile, level)
        outfile.write('content_ = [\n')
        for item_ in self.content_:
            item_.exportLiteral(outfile, level, name_)
        showIndent(outfile, level)
        outfile.write('],\n')
        showIndent(outfile, level)
        outfile.write('content_ = [\n')
        for item_ in self.content_:
            item_.exportLiteral(outfile, level, name_)
        showIndent(outfile, level)
        outfile.write('],\n')
        showIndent(outfile, level)
        outfile.write('content_ = [\n')
        for item_ in self.content_:
            item_.exportLiteral(outfile, level, name_)
        showIndent(outfile, level)
        outfile.write('],\n')
        showIndent(outfile, level)
        outfile.write('content_ = [\n')
        for item_ in self.content_:
            item_.exportLiteral(outfile, level, name_)
        showIndent(outfile, level)
        outfile.write('],\n')

    def build(self, node_):
        if False:
            return 10
        attrs = node_.attributes
        self.buildAttributes(attrs)
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            i = 10
            return i + 15
        pass

    def buildChildren(self, child_, nodeName_):
        if False:
            print('Hello World!')
        if child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'title':
            childobj_ = docTitleType.factory()
            childobj_.build(child_)
            obj_ = self.mixedclass_(MixedContainer.CategoryComplex, MixedContainer.TypeNone, 'title', childobj_)
            self.content_.append(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'para':
            childobj_ = docParaType.factory()
            childobj_.build(child_)
            obj_ = self.mixedclass_(MixedContainer.CategoryComplex, MixedContainer.TypeNone, 'para', childobj_)
            self.content_.append(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'sect1':
            childobj_ = docSect1Type.factory()
            childobj_.build(child_)
            obj_ = self.mixedclass_(MixedContainer.CategoryComplex, MixedContainer.TypeNone, 'sect1', childobj_)
            self.content_.append(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'internal':
            childobj_ = docInternalType.factory()
            childobj_.build(child_)
            obj_ = self.mixedclass_(MixedContainer.CategoryComplex, MixedContainer.TypeNone, 'internal', childobj_)
            self.content_.append(obj_)
        elif child_.nodeType == Node.TEXT_NODE:
            obj_ = self.mixedclass_(MixedContainer.CategoryText, MixedContainer.TypeNone, '', child_.nodeValue)
            self.content_.append(obj_)

class enumvalueType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, prot=None, id=None, name=None, initializer=None, briefdescription=None, detaileddescription=None, mixedclass_=None, content_=None):
        if False:
            while True:
                i = 10
        self.prot = prot
        self.id = id
        if mixedclass_ is None:
            self.mixedclass_ = MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        if content_ is None:
            self.content_ = []
        else:
            self.content_ = content_

    def factory(*args_, **kwargs_):
        if False:
            for i in range(10):
                print('nop')
        if enumvalueType.subclass:
            return enumvalueType.subclass(*args_, **kwargs_)
        else:
            return enumvalueType(*args_, **kwargs_)
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

    def get_initializer(self):
        if False:
            for i in range(10):
                print('nop')
        return self.initializer

    def set_initializer(self, initializer):
        if False:
            i = 10
            return i + 15
        self.initializer = initializer

    def get_briefdescription(self):
        if False:
            i = 10
            return i + 15
        return self.briefdescription

    def set_briefdescription(self, briefdescription):
        if False:
            i = 10
            return i + 15
        self.briefdescription = briefdescription

    def get_detaileddescription(self):
        if False:
            for i in range(10):
                print('nop')
        return self.detaileddescription

    def set_detaileddescription(self, detaileddescription):
        if False:
            while True:
                i = 10
        self.detaileddescription = detaileddescription

    def get_prot(self):
        if False:
            print('Hello World!')
        return self.prot

    def set_prot(self, prot):
        if False:
            while True:
                i = 10
        self.prot = prot

    def get_id(self):
        if False:
            print('Hello World!')
        return self.id

    def set_id(self, id):
        if False:
            return 10
        self.id = id

    def export(self, outfile, level, namespace_='', name_='enumvalueType', namespacedef_=''):
        if False:
            for i in range(10):
                print('nop')
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='enumvalueType')
        outfile.write('>')
        self.exportChildren(outfile, level + 1, namespace_, name_)
        outfile.write('</%s%s>\n' % (namespace_, name_))

    def exportAttributes(self, outfile, level, namespace_='', name_='enumvalueType'):
        if False:
            print('Hello World!')
        if self.prot is not None:
            outfile.write(' prot=%s' % (quote_attrib(self.prot),))
        if self.id is not None:
            outfile.write(' id=%s' % (self.format_string(quote_attrib(self.id).encode(ExternalEncoding), input_name='id'),))

    def exportChildren(self, outfile, level, namespace_='', name_='enumvalueType'):
        if False:
            for i in range(10):
                print('nop')
        for item_ in self.content_:
            item_.export(outfile, level, item_.name, namespace_)

    def hasContent_(self):
        if False:
            return 10
        if self.name is not None or self.initializer is not None or self.briefdescription is not None or (self.detaileddescription is not None):
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='enumvalueType'):
        if False:
            return 10
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            while True:
                i = 10
        if self.prot is not None:
            showIndent(outfile, level)
            outfile.write('prot = "%s",\n' % (self.prot,))
        if self.id is not None:
            showIndent(outfile, level)
            outfile.write('id = %s,\n' % (self.id,))

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            print('Hello World!')
        showIndent(outfile, level)
        outfile.write('content_ = [\n')
        for item_ in self.content_:
            item_.exportLiteral(outfile, level, name_)
        showIndent(outfile, level)
        outfile.write('],\n')
        showIndent(outfile, level)
        outfile.write('content_ = [\n')
        for item_ in self.content_:
            item_.exportLiteral(outfile, level, name_)
        showIndent(outfile, level)
        outfile.write('],\n')
        showIndent(outfile, level)
        outfile.write('content_ = [\n')
        for item_ in self.content_:
            item_.exportLiteral(outfile, level, name_)
        showIndent(outfile, level)
        outfile.write('],\n')
        showIndent(outfile, level)
        outfile.write('content_ = [\n')
        for item_ in self.content_:
            item_.exportLiteral(outfile, level, name_)
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
            i = 10
            return i + 15
        if attrs.get('prot'):
            self.prot = attrs.get('prot').value
        if attrs.get('id'):
            self.id = attrs.get('id').value

    def buildChildren(self, child_, nodeName_):
        if False:
            for i in range(10):
                print('nop')
        if child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'name':
            value_ = []
            for text_ in child_.childNodes:
                value_.append(text_.nodeValue)
            valuestr_ = ''.join(value_)
            obj_ = self.mixedclass_(MixedContainer.CategorySimple, MixedContainer.TypeString, 'name', valuestr_)
            self.content_.append(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'initializer':
            childobj_ = linkedTextType.factory()
            childobj_.build(child_)
            obj_ = self.mixedclass_(MixedContainer.CategoryComplex, MixedContainer.TypeNone, 'initializer', childobj_)
            self.content_.append(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'briefdescription':
            childobj_ = descriptionType.factory()
            childobj_.build(child_)
            obj_ = self.mixedclass_(MixedContainer.CategoryComplex, MixedContainer.TypeNone, 'briefdescription', childobj_)
            self.content_.append(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'detaileddescription':
            childobj_ = descriptionType.factory()
            childobj_.build(child_)
            obj_ = self.mixedclass_(MixedContainer.CategoryComplex, MixedContainer.TypeNone, 'detaileddescription', childobj_)
            self.content_.append(obj_)
        elif child_.nodeType == Node.TEXT_NODE:
            obj_ = self.mixedclass_(MixedContainer.CategoryText, MixedContainer.TypeNone, '', child_.nodeValue)
            self.content_.append(obj_)

class templateparamlistType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, param=None):
        if False:
            while True:
                i = 10
        if param is None:
            self.param = []
        else:
            self.param = param

    def factory(*args_, **kwargs_):
        if False:
            while True:
                i = 10
        if templateparamlistType.subclass:
            return templateparamlistType.subclass(*args_, **kwargs_)
        else:
            return templateparamlistType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_param(self):
        if False:
            return 10
        return self.param

    def set_param(self, param):
        if False:
            while True:
                i = 10
        self.param = param

    def add_param(self, value):
        if False:
            i = 10
            return i + 15
        self.param.append(value)

    def insert_param(self, index, value):
        if False:
            while True:
                i = 10
        self.param[index] = value

    def export(self, outfile, level, namespace_='', name_='templateparamlistType', namespacedef_=''):
        if False:
            i = 10
            return i + 15
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='templateparamlistType')
        if self.hasContent_():
            outfile.write('>\n')
            self.exportChildren(outfile, level + 1, namespace_, name_)
            showIndent(outfile, level)
            outfile.write('</%s%s>\n' % (namespace_, name_))
        else:
            outfile.write(' />\n')

    def exportAttributes(self, outfile, level, namespace_='', name_='templateparamlistType'):
        if False:
            return 10
        pass

    def exportChildren(self, outfile, level, namespace_='', name_='templateparamlistType'):
        if False:
            while True:
                i = 10
        for param_ in self.param:
            param_.export(outfile, level, namespace_, name_='param')

    def hasContent_(self):
        if False:
            for i in range(10):
                print('nop')
        if self.param is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='templateparamlistType'):
        if False:
            print('Hello World!')
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            i = 10
            return i + 15
        pass

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            i = 10
            return i + 15
        showIndent(outfile, level)
        outfile.write('param=[\n')
        level += 1
        for param in self.param:
            showIndent(outfile, level)
            outfile.write('model_.param(\n')
            param.exportLiteral(outfile, level, name_='param')
            showIndent(outfile, level)
            outfile.write('),\n')
        level -= 1
        showIndent(outfile, level)
        outfile.write('],\n')

    def build(self, node_):
        if False:
            i = 10
            return i + 15
        attrs = node_.attributes
        self.buildAttributes(attrs)
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            while True:
                i = 10
        pass

    def buildChildren(self, child_, nodeName_):
        if False:
            i = 10
            return i + 15
        if child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'param':
            obj_ = paramType.factory()
            obj_.build(child_)
            self.param.append(obj_)

class paramType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, type_=None, declname=None, defname=None, array=None, defval=None, briefdescription=None):
        if False:
            print('Hello World!')
        self.type_ = type_
        self.declname = declname
        self.defname = defname
        self.array = array
        self.defval = defval
        self.briefdescription = briefdescription

    def factory(*args_, **kwargs_):
        if False:
            while True:
                i = 10
        if paramType.subclass:
            return paramType.subclass(*args_, **kwargs_)
        else:
            return paramType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_type(self):
        if False:
            i = 10
            return i + 15
        return self.type_

    def set_type(self, type_):
        if False:
            i = 10
            return i + 15
        self.type_ = type_

    def get_declname(self):
        if False:
            i = 10
            return i + 15
        return self.declname

    def set_declname(self, declname):
        if False:
            for i in range(10):
                print('nop')
        self.declname = declname

    def get_defname(self):
        if False:
            for i in range(10):
                print('nop')
        return self.defname

    def set_defname(self, defname):
        if False:
            for i in range(10):
                print('nop')
        self.defname = defname

    def get_array(self):
        if False:
            return 10
        return self.array

    def set_array(self, array):
        if False:
            return 10
        self.array = array

    def get_defval(self):
        if False:
            return 10
        return self.defval

    def set_defval(self, defval):
        if False:
            i = 10
            return i + 15
        self.defval = defval

    def get_briefdescription(self):
        if False:
            for i in range(10):
                print('nop')
        return self.briefdescription

    def set_briefdescription(self, briefdescription):
        if False:
            return 10
        self.briefdescription = briefdescription

    def export(self, outfile, level, namespace_='', name_='paramType', namespacedef_=''):
        if False:
            print('Hello World!')
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='paramType')
        if self.hasContent_():
            outfile.write('>\n')
            self.exportChildren(outfile, level + 1, namespace_, name_)
            showIndent(outfile, level)
            outfile.write('</%s%s>\n' % (namespace_, name_))
        else:
            outfile.write(' />\n')

    def exportAttributes(self, outfile, level, namespace_='', name_='paramType'):
        if False:
            while True:
                i = 10
        pass

    def exportChildren(self, outfile, level, namespace_='', name_='paramType'):
        if False:
            return 10
        if self.type_:
            self.type_.export(outfile, level, namespace_, name_='type')
        if self.declname is not None:
            showIndent(outfile, level)
            outfile.write('<%sdeclname>%s</%sdeclname>\n' % (namespace_, self.format_string(quote_xml(self.declname).encode(ExternalEncoding), input_name='declname'), namespace_))
        if self.defname is not None:
            showIndent(outfile, level)
            outfile.write('<%sdefname>%s</%sdefname>\n' % (namespace_, self.format_string(quote_xml(self.defname).encode(ExternalEncoding), input_name='defname'), namespace_))
        if self.array is not None:
            showIndent(outfile, level)
            outfile.write('<%sarray>%s</%sarray>\n' % (namespace_, self.format_string(quote_xml(self.array).encode(ExternalEncoding), input_name='array'), namespace_))
        if self.defval:
            self.defval.export(outfile, level, namespace_, name_='defval')
        if self.briefdescription:
            self.briefdescription.export(outfile, level, namespace_, name_='briefdescription')

    def hasContent_(self):
        if False:
            while True:
                i = 10
        if self.type_ is not None or self.declname is not None or self.defname is not None or (self.array is not None) or (self.defval is not None) or (self.briefdescription is not None):
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='paramType'):
        if False:
            print('Hello World!')
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            print('Hello World!')
        pass

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            while True:
                i = 10
        if self.type_:
            showIndent(outfile, level)
            outfile.write('type_=model_.linkedTextType(\n')
            self.type_.exportLiteral(outfile, level, name_='type')
            showIndent(outfile, level)
            outfile.write('),\n')
        showIndent(outfile, level)
        outfile.write('declname=%s,\n' % quote_python(self.declname).encode(ExternalEncoding))
        showIndent(outfile, level)
        outfile.write('defname=%s,\n' % quote_python(self.defname).encode(ExternalEncoding))
        showIndent(outfile, level)
        outfile.write('array=%s,\n' % quote_python(self.array).encode(ExternalEncoding))
        if self.defval:
            showIndent(outfile, level)
            outfile.write('defval=model_.linkedTextType(\n')
            self.defval.exportLiteral(outfile, level, name_='defval')
            showIndent(outfile, level)
            outfile.write('),\n')
        if self.briefdescription:
            showIndent(outfile, level)
            outfile.write('briefdescription=model_.descriptionType(\n')
            self.briefdescription.exportLiteral(outfile, level, name_='briefdescription')
            showIndent(outfile, level)
            outfile.write('),\n')

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
            print('Hello World!')
        pass

    def buildChildren(self, child_, nodeName_):
        if False:
            for i in range(10):
                print('nop')
        if child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'type':
            obj_ = linkedTextType.factory()
            obj_.build(child_)
            self.set_type(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'declname':
            declname_ = ''
            for text__content_ in child_.childNodes:
                declname_ += text__content_.nodeValue
            self.declname = declname_
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'defname':
            defname_ = ''
            for text__content_ in child_.childNodes:
                defname_ += text__content_.nodeValue
            self.defname = defname_
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'array':
            array_ = ''
            for text__content_ in child_.childNodes:
                array_ += text__content_.nodeValue
            self.array = array_
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'defval':
            obj_ = linkedTextType.factory()
            obj_.build(child_)
            self.set_defval(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'briefdescription':
            obj_ = descriptionType.factory()
            obj_.build(child_)
            self.set_briefdescription(obj_)

class declname(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, valueOf_=''):
        if False:
            return 10
        self.valueOf_ = valueOf_

    def factory(*args_, **kwargs_):
        if False:
            while True:
                i = 10
        if declname.subclass:
            return declname.subclass(*args_, **kwargs_)
        else:
            return declname(*args_, **kwargs_)
    factory = staticmethod(factory)

    def getValueOf_(self):
        if False:
            print('Hello World!')
        return self.valueOf_

    def setValueOf_(self, valueOf_):
        if False:
            print('Hello World!')
        self.valueOf_ = valueOf_

    def export(self, outfile, level, namespace_='', name_='declname', namespacedef_=''):
        if False:
            while True:
                i = 10
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='declname')
        if self.hasContent_():
            outfile.write('>\n')
            self.exportChildren(outfile, level + 1, namespace_, name_)
            showIndent(outfile, level)
            outfile.write('</%s%s>\n' % (namespace_, name_))
        else:
            outfile.write(' />\n')

    def exportAttributes(self, outfile, level, namespace_='', name_='declname'):
        if False:
            i = 10
            return i + 15
        pass

    def exportChildren(self, outfile, level, namespace_='', name_='declname'):
        if False:
            print('Hello World!')
        if self.valueOf_.find('![CDATA') > -1:
            value = quote_xml('%s' % self.valueOf_)
            value = value.replace('![CDATA', '<![CDATA')
            value = value.replace(']]', ']]>')
            outfile.write(value)
        else:
            outfile.write(quote_xml('%s' % self.valueOf_))

    def hasContent_(self):
        if False:
            return 10
        if self.valueOf_ is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='declname'):
        if False:
            for i in range(10):
                print('nop')
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            while True:
                i = 10
        pass

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            print('Hello World!')
        showIndent(outfile, level)
        outfile.write('valueOf_ = "%s",\n' % (self.valueOf_,))

    def build(self, node_):
        if False:
            while True:
                i = 10
        attrs = node_.attributes
        self.buildAttributes(attrs)
        self.valueOf_ = ''
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            for i in range(10):
                print('nop')
        pass

    def buildChildren(self, child_, nodeName_):
        if False:
            while True:
                i = 10
        if child_.nodeType == Node.TEXT_NODE:
            self.valueOf_ += child_.nodeValue
        elif child_.nodeType == Node.CDATA_SECTION_NODE:
            self.valueOf_ += '![CDATA[' + child_.nodeValue + ']]'

class defname(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, valueOf_=''):
        if False:
            while True:
                i = 10
        self.valueOf_ = valueOf_

    def factory(*args_, **kwargs_):
        if False:
            for i in range(10):
                print('nop')
        if defname.subclass:
            return defname.subclass(*args_, **kwargs_)
        else:
            return defname(*args_, **kwargs_)
    factory = staticmethod(factory)

    def getValueOf_(self):
        if False:
            print('Hello World!')
        return self.valueOf_

    def setValueOf_(self, valueOf_):
        if False:
            for i in range(10):
                print('nop')
        self.valueOf_ = valueOf_

    def export(self, outfile, level, namespace_='', name_='defname', namespacedef_=''):
        if False:
            while True:
                i = 10
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='defname')
        if self.hasContent_():
            outfile.write('>\n')
            self.exportChildren(outfile, level + 1, namespace_, name_)
            showIndent(outfile, level)
            outfile.write('</%s%s>\n' % (namespace_, name_))
        else:
            outfile.write(' />\n')

    def exportAttributes(self, outfile, level, namespace_='', name_='defname'):
        if False:
            print('Hello World!')
        pass

    def exportChildren(self, outfile, level, namespace_='', name_='defname'):
        if False:
            i = 10
            return i + 15
        if self.valueOf_.find('![CDATA') > -1:
            value = quote_xml('%s' % self.valueOf_)
            value = value.replace('![CDATA', '<![CDATA')
            value = value.replace(']]', ']]>')
            outfile.write(value)
        else:
            outfile.write(quote_xml('%s' % self.valueOf_))

    def hasContent_(self):
        if False:
            print('Hello World!')
        if self.valueOf_ is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='defname'):
        if False:
            i = 10
            return i + 15
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            i = 10
            return i + 15
        pass

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            for i in range(10):
                print('nop')
        showIndent(outfile, level)
        outfile.write('valueOf_ = "%s",\n' % (self.valueOf_,))

    def build(self, node_):
        if False:
            while True:
                i = 10
        attrs = node_.attributes
        self.buildAttributes(attrs)
        self.valueOf_ = ''
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            print('Hello World!')
        pass

    def buildChildren(self, child_, nodeName_):
        if False:
            print('Hello World!')
        if child_.nodeType == Node.TEXT_NODE:
            self.valueOf_ += child_.nodeValue
        elif child_.nodeType == Node.CDATA_SECTION_NODE:
            self.valueOf_ += '![CDATA[' + child_.nodeValue + ']]'

class array(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, valueOf_=''):
        if False:
            print('Hello World!')
        self.valueOf_ = valueOf_

    def factory(*args_, **kwargs_):
        if False:
            for i in range(10):
                print('nop')
        if array.subclass:
            return array.subclass(*args_, **kwargs_)
        else:
            return array(*args_, **kwargs_)
    factory = staticmethod(factory)

    def getValueOf_(self):
        if False:
            return 10
        return self.valueOf_

    def setValueOf_(self, valueOf_):
        if False:
            print('Hello World!')
        self.valueOf_ = valueOf_

    def export(self, outfile, level, namespace_='', name_='array', namespacedef_=''):
        if False:
            print('Hello World!')
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='array')
        if self.hasContent_():
            outfile.write('>\n')
            self.exportChildren(outfile, level + 1, namespace_, name_)
            showIndent(outfile, level)
            outfile.write('</%s%s>\n' % (namespace_, name_))
        else:
            outfile.write(' />\n')

    def exportAttributes(self, outfile, level, namespace_='', name_='array'):
        if False:
            while True:
                i = 10
        pass

    def exportChildren(self, outfile, level, namespace_='', name_='array'):
        if False:
            i = 10
            return i + 15
        if self.valueOf_.find('![CDATA') > -1:
            value = quote_xml('%s' % self.valueOf_)
            value = value.replace('![CDATA', '<![CDATA')
            value = value.replace(']]', ']]>')
            outfile.write(value)
        else:
            outfile.write(quote_xml('%s' % self.valueOf_))

    def hasContent_(self):
        if False:
            i = 10
            return i + 15
        if self.valueOf_ is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='array'):
        if False:
            print('Hello World!')
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            i = 10
            return i + 15
        pass

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            for i in range(10):
                print('nop')
        showIndent(outfile, level)
        outfile.write('valueOf_ = "%s",\n' % (self.valueOf_,))

    def build(self, node_):
        if False:
            for i in range(10):
                print('nop')
        attrs = node_.attributes
        self.buildAttributes(attrs)
        self.valueOf_ = ''
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            return 10
        pass

    def buildChildren(self, child_, nodeName_):
        if False:
            for i in range(10):
                print('nop')
        if child_.nodeType == Node.TEXT_NODE:
            self.valueOf_ += child_.nodeValue
        elif child_.nodeType == Node.CDATA_SECTION_NODE:
            self.valueOf_ += '![CDATA[' + child_.nodeValue + ']]'

class linkedTextType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, ref=None, mixedclass_=None, content_=None):
        if False:
            while True:
                i = 10
        if mixedclass_ is None:
            self.mixedclass_ = MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        if content_ is None:
            self.content_ = []
        else:
            self.content_ = content_

    def factory(*args_, **kwargs_):
        if False:
            while True:
                i = 10
        if linkedTextType.subclass:
            return linkedTextType.subclass(*args_, **kwargs_)
        else:
            return linkedTextType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_ref(self):
        if False:
            print('Hello World!')
        return self.ref

    def set_ref(self, ref):
        if False:
            print('Hello World!')
        self.ref = ref

    def add_ref(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.ref.append(value)

    def insert_ref(self, index, value):
        if False:
            while True:
                i = 10
        self.ref[index] = value

    def export(self, outfile, level, namespace_='', name_='linkedTextType', namespacedef_=''):
        if False:
            print('Hello World!')
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='linkedTextType')
        outfile.write('>')
        self.exportChildren(outfile, level + 1, namespace_, name_)
        outfile.write('</%s%s>\n' % (namespace_, name_))

    def exportAttributes(self, outfile, level, namespace_='', name_='linkedTextType'):
        if False:
            while True:
                i = 10
        pass

    def exportChildren(self, outfile, level, namespace_='', name_='linkedTextType'):
        if False:
            for i in range(10):
                print('nop')
        for item_ in self.content_:
            item_.export(outfile, level, item_.name, namespace_)

    def hasContent_(self):
        if False:
            print('Hello World!')
        if self.ref is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='linkedTextType'):
        if False:
            while True:
                i = 10
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            i = 10
            return i + 15
        pass

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            return 10
        showIndent(outfile, level)
        outfile.write('content_ = [\n')
        for item_ in self.content_:
            item_.exportLiteral(outfile, level, name_)
        showIndent(outfile, level)
        outfile.write('],\n')

    def build(self, node_):
        if False:
            return 10
        attrs = node_.attributes
        self.buildAttributes(attrs)
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            return 10
        pass

    def buildChildren(self, child_, nodeName_):
        if False:
            return 10
        if child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'ref':
            childobj_ = docRefTextType.factory()
            childobj_.build(child_)
            obj_ = self.mixedclass_(MixedContainer.CategoryComplex, MixedContainer.TypeNone, 'ref', childobj_)
            self.content_.append(obj_)
        elif child_.nodeType == Node.TEXT_NODE:
            obj_ = self.mixedclass_(MixedContainer.CategoryText, MixedContainer.TypeNone, '', child_.nodeValue)
            self.content_.append(obj_)

class graphType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, node=None):
        if False:
            while True:
                i = 10
        if node is None:
            self.node = []
        else:
            self.node = node

    def factory(*args_, **kwargs_):
        if False:
            for i in range(10):
                print('nop')
        if graphType.subclass:
            return graphType.subclass(*args_, **kwargs_)
        else:
            return graphType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_node(self):
        if False:
            while True:
                i = 10
        return self.node

    def set_node(self, node):
        if False:
            i = 10
            return i + 15
        self.node = node

    def add_node(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.node.append(value)

    def insert_node(self, index, value):
        if False:
            print('Hello World!')
        self.node[index] = value

    def export(self, outfile, level, namespace_='', name_='graphType', namespacedef_=''):
        if False:
            return 10
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='graphType')
        if self.hasContent_():
            outfile.write('>\n')
            self.exportChildren(outfile, level + 1, namespace_, name_)
            showIndent(outfile, level)
            outfile.write('</%s%s>\n' % (namespace_, name_))
        else:
            outfile.write(' />\n')

    def exportAttributes(self, outfile, level, namespace_='', name_='graphType'):
        if False:
            i = 10
            return i + 15
        pass

    def exportChildren(self, outfile, level, namespace_='', name_='graphType'):
        if False:
            return 10
        for node_ in self.node:
            node_.export(outfile, level, namespace_, name_='node')

    def hasContent_(self):
        if False:
            for i in range(10):
                print('nop')
        if self.node is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='graphType'):
        if False:
            while True:
                i = 10
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            return 10
        pass

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            i = 10
            return i + 15
        showIndent(outfile, level)
        outfile.write('node=[\n')
        level += 1
        for node in self.node:
            showIndent(outfile, level)
            outfile.write('model_.node(\n')
            node.exportLiteral(outfile, level, name_='node')
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
            print('Hello World!')
        pass

    def buildChildren(self, child_, nodeName_):
        if False:
            print('Hello World!')
        if child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'node':
            obj_ = nodeType.factory()
            obj_.build(child_)
            self.node.append(obj_)

class nodeType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, id=None, label=None, link=None, childnode=None):
        if False:
            return 10
        self.id = id
        self.label = label
        self.link = link
        if childnode is None:
            self.childnode = []
        else:
            self.childnode = childnode

    def factory(*args_, **kwargs_):
        if False:
            print('Hello World!')
        if nodeType.subclass:
            return nodeType.subclass(*args_, **kwargs_)
        else:
            return nodeType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_label(self):
        if False:
            while True:
                i = 10
        return self.label

    def set_label(self, label):
        if False:
            print('Hello World!')
        self.label = label

    def get_link(self):
        if False:
            print('Hello World!')
        return self.link

    def set_link(self, link):
        if False:
            print('Hello World!')
        self.link = link

    def get_childnode(self):
        if False:
            for i in range(10):
                print('nop')
        return self.childnode

    def set_childnode(self, childnode):
        if False:
            while True:
                i = 10
        self.childnode = childnode

    def add_childnode(self, value):
        if False:
            i = 10
            return i + 15
        self.childnode.append(value)

    def insert_childnode(self, index, value):
        if False:
            i = 10
            return i + 15
        self.childnode[index] = value

    def get_id(self):
        if False:
            i = 10
            return i + 15
        return self.id

    def set_id(self, id):
        if False:
            while True:
                i = 10
        self.id = id

    def export(self, outfile, level, namespace_='', name_='nodeType', namespacedef_=''):
        if False:
            i = 10
            return i + 15
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='nodeType')
        if self.hasContent_():
            outfile.write('>\n')
            self.exportChildren(outfile, level + 1, namespace_, name_)
            showIndent(outfile, level)
            outfile.write('</%s%s>\n' % (namespace_, name_))
        else:
            outfile.write(' />\n')

    def exportAttributes(self, outfile, level, namespace_='', name_='nodeType'):
        if False:
            for i in range(10):
                print('nop')
        if self.id is not None:
            outfile.write(' id=%s' % (self.format_string(quote_attrib(self.id).encode(ExternalEncoding), input_name='id'),))

    def exportChildren(self, outfile, level, namespace_='', name_='nodeType'):
        if False:
            while True:
                i = 10
        if self.label is not None:
            showIndent(outfile, level)
            outfile.write('<%slabel>%s</%slabel>\n' % (namespace_, self.format_string(quote_xml(self.label).encode(ExternalEncoding), input_name='label'), namespace_))
        if self.link:
            self.link.export(outfile, level, namespace_, name_='link')
        for childnode_ in self.childnode:
            childnode_.export(outfile, level, namespace_, name_='childnode')

    def hasContent_(self):
        if False:
            for i in range(10):
                print('nop')
        if self.label is not None or self.link is not None or self.childnode is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='nodeType'):
        if False:
            for i in range(10):
                print('nop')
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            i = 10
            return i + 15
        if self.id is not None:
            showIndent(outfile, level)
            outfile.write('id = %s,\n' % (self.id,))

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            i = 10
            return i + 15
        showIndent(outfile, level)
        outfile.write('label=%s,\n' % quote_python(self.label).encode(ExternalEncoding))
        if self.link:
            showIndent(outfile, level)
            outfile.write('link=model_.linkType(\n')
            self.link.exportLiteral(outfile, level, name_='link')
            showIndent(outfile, level)
            outfile.write('),\n')
        showIndent(outfile, level)
        outfile.write('childnode=[\n')
        level += 1
        for childnode in self.childnode:
            showIndent(outfile, level)
            outfile.write('model_.childnode(\n')
            childnode.exportLiteral(outfile, level, name_='childnode')
            showIndent(outfile, level)
            outfile.write('),\n')
        level -= 1
        showIndent(outfile, level)
        outfile.write('],\n')

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
            i = 10
            return i + 15
        if attrs.get('id'):
            self.id = attrs.get('id').value

    def buildChildren(self, child_, nodeName_):
        if False:
            for i in range(10):
                print('nop')
        if child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'label':
            label_ = ''
            for text__content_ in child_.childNodes:
                label_ += text__content_.nodeValue
            self.label = label_
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'link':
            obj_ = linkType.factory()
            obj_.build(child_)
            self.set_link(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'childnode':
            obj_ = childnodeType.factory()
            obj_.build(child_)
            self.childnode.append(obj_)

class label(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, valueOf_=''):
        if False:
            i = 10
            return i + 15
        self.valueOf_ = valueOf_

    def factory(*args_, **kwargs_):
        if False:
            print('Hello World!')
        if label.subclass:
            return label.subclass(*args_, **kwargs_)
        else:
            return label(*args_, **kwargs_)
    factory = staticmethod(factory)

    def getValueOf_(self):
        if False:
            while True:
                i = 10
        return self.valueOf_

    def setValueOf_(self, valueOf_):
        if False:
            return 10
        self.valueOf_ = valueOf_

    def export(self, outfile, level, namespace_='', name_='label', namespacedef_=''):
        if False:
            return 10
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='label')
        if self.hasContent_():
            outfile.write('>\n')
            self.exportChildren(outfile, level + 1, namespace_, name_)
            showIndent(outfile, level)
            outfile.write('</%s%s>\n' % (namespace_, name_))
        else:
            outfile.write(' />\n')

    def exportAttributes(self, outfile, level, namespace_='', name_='label'):
        if False:
            for i in range(10):
                print('nop')
        pass

    def exportChildren(self, outfile, level, namespace_='', name_='label'):
        if False:
            while True:
                i = 10
        if self.valueOf_.find('![CDATA') > -1:
            value = quote_xml('%s' % self.valueOf_)
            value = value.replace('![CDATA', '<![CDATA')
            value = value.replace(']]', ']]>')
            outfile.write(value)
        else:
            outfile.write(quote_xml('%s' % self.valueOf_))

    def hasContent_(self):
        if False:
            return 10
        if self.valueOf_ is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='label'):
        if False:
            return 10
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            return 10
        pass

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            print('Hello World!')
        showIndent(outfile, level)
        outfile.write('valueOf_ = "%s",\n' % (self.valueOf_,))

    def build(self, node_):
        if False:
            i = 10
            return i + 15
        attrs = node_.attributes
        self.buildAttributes(attrs)
        self.valueOf_ = ''
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            for i in range(10):
                print('nop')
        pass

    def buildChildren(self, child_, nodeName_):
        if False:
            i = 10
            return i + 15
        if child_.nodeType == Node.TEXT_NODE:
            self.valueOf_ += child_.nodeValue
        elif child_.nodeType == Node.CDATA_SECTION_NODE:
            self.valueOf_ += '![CDATA[' + child_.nodeValue + ']]'

class childnodeType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, relation=None, refid=None, edgelabel=None):
        if False:
            print('Hello World!')
        self.relation = relation
        self.refid = refid
        if edgelabel is None:
            self.edgelabel = []
        else:
            self.edgelabel = edgelabel

    def factory(*args_, **kwargs_):
        if False:
            while True:
                i = 10
        if childnodeType.subclass:
            return childnodeType.subclass(*args_, **kwargs_)
        else:
            return childnodeType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_edgelabel(self):
        if False:
            while True:
                i = 10
        return self.edgelabel

    def set_edgelabel(self, edgelabel):
        if False:
            while True:
                i = 10
        self.edgelabel = edgelabel

    def add_edgelabel(self, value):
        if False:
            while True:
                i = 10
        self.edgelabel.append(value)

    def insert_edgelabel(self, index, value):
        if False:
            while True:
                i = 10
        self.edgelabel[index] = value

    def get_relation(self):
        if False:
            i = 10
            return i + 15
        return self.relation

    def set_relation(self, relation):
        if False:
            for i in range(10):
                print('nop')
        self.relation = relation

    def get_refid(self):
        if False:
            print('Hello World!')
        return self.refid

    def set_refid(self, refid):
        if False:
            for i in range(10):
                print('nop')
        self.refid = refid

    def export(self, outfile, level, namespace_='', name_='childnodeType', namespacedef_=''):
        if False:
            i = 10
            return i + 15
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='childnodeType')
        if self.hasContent_():
            outfile.write('>\n')
            self.exportChildren(outfile, level + 1, namespace_, name_)
            showIndent(outfile, level)
            outfile.write('</%s%s>\n' % (namespace_, name_))
        else:
            outfile.write(' />\n')

    def exportAttributes(self, outfile, level, namespace_='', name_='childnodeType'):
        if False:
            i = 10
            return i + 15
        if self.relation is not None:
            outfile.write(' relation=%s' % (quote_attrib(self.relation),))
        if self.refid is not None:
            outfile.write(' refid=%s' % (self.format_string(quote_attrib(self.refid).encode(ExternalEncoding), input_name='refid'),))

    def exportChildren(self, outfile, level, namespace_='', name_='childnodeType'):
        if False:
            i = 10
            return i + 15
        for edgelabel_ in self.edgelabel:
            showIndent(outfile, level)
            outfile.write('<%sedgelabel>%s</%sedgelabel>\n' % (namespace_, self.format_string(quote_xml(edgelabel_).encode(ExternalEncoding), input_name='edgelabel'), namespace_))

    def hasContent_(self):
        if False:
            return 10
        if self.edgelabel is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='childnodeType'):
        if False:
            for i in range(10):
                print('nop')
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            while True:
                i = 10
        if self.relation is not None:
            showIndent(outfile, level)
            outfile.write('relation = "%s",\n' % (self.relation,))
        if self.refid is not None:
            showIndent(outfile, level)
            outfile.write('refid = %s,\n' % (self.refid,))

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            return 10
        showIndent(outfile, level)
        outfile.write('edgelabel=[\n')
        level += 1
        for edgelabel in self.edgelabel:
            showIndent(outfile, level)
            outfile.write('%s,\n' % quote_python(edgelabel).encode(ExternalEncoding))
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
        if attrs.get('relation'):
            self.relation = attrs.get('relation').value
        if attrs.get('refid'):
            self.refid = attrs.get('refid').value

    def buildChildren(self, child_, nodeName_):
        if False:
            i = 10
            return i + 15
        if child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'edgelabel':
            edgelabel_ = ''
            for text__content_ in child_.childNodes:
                edgelabel_ += text__content_.nodeValue
            self.edgelabel.append(edgelabel_)

class edgelabel(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, valueOf_=''):
        if False:
            for i in range(10):
                print('nop')
        self.valueOf_ = valueOf_

    def factory(*args_, **kwargs_):
        if False:
            i = 10
            return i + 15
        if edgelabel.subclass:
            return edgelabel.subclass(*args_, **kwargs_)
        else:
            return edgelabel(*args_, **kwargs_)
    factory = staticmethod(factory)

    def getValueOf_(self):
        if False:
            for i in range(10):
                print('nop')
        return self.valueOf_

    def setValueOf_(self, valueOf_):
        if False:
            while True:
                i = 10
        self.valueOf_ = valueOf_

    def export(self, outfile, level, namespace_='', name_='edgelabel', namespacedef_=''):
        if False:
            return 10
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='edgelabel')
        if self.hasContent_():
            outfile.write('>\n')
            self.exportChildren(outfile, level + 1, namespace_, name_)
            showIndent(outfile, level)
            outfile.write('</%s%s>\n' % (namespace_, name_))
        else:
            outfile.write(' />\n')

    def exportAttributes(self, outfile, level, namespace_='', name_='edgelabel'):
        if False:
            print('Hello World!')
        pass

    def exportChildren(self, outfile, level, namespace_='', name_='edgelabel'):
        if False:
            return 10
        if self.valueOf_.find('![CDATA') > -1:
            value = quote_xml('%s' % self.valueOf_)
            value = value.replace('![CDATA', '<![CDATA')
            value = value.replace(']]', ']]>')
            outfile.write(value)
        else:
            outfile.write(quote_xml('%s' % self.valueOf_))

    def hasContent_(self):
        if False:
            print('Hello World!')
        if self.valueOf_ is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='edgelabel'):
        if False:
            print('Hello World!')
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            print('Hello World!')
        pass

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            return 10
        showIndent(outfile, level)
        outfile.write('valueOf_ = "%s",\n' % (self.valueOf_,))

    def build(self, node_):
        if False:
            print('Hello World!')
        attrs = node_.attributes
        self.buildAttributes(attrs)
        self.valueOf_ = ''
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            return 10
        pass

    def buildChildren(self, child_, nodeName_):
        if False:
            i = 10
            return i + 15
        if child_.nodeType == Node.TEXT_NODE:
            self.valueOf_ += child_.nodeValue
        elif child_.nodeType == Node.CDATA_SECTION_NODE:
            self.valueOf_ += '![CDATA[' + child_.nodeValue + ']]'

class linkType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, refid=None, external=None, valueOf_=''):
        if False:
            for i in range(10):
                print('nop')
        self.refid = refid
        self.external = external
        self.valueOf_ = valueOf_

    def factory(*args_, **kwargs_):
        if False:
            for i in range(10):
                print('nop')
        if linkType.subclass:
            return linkType.subclass(*args_, **kwargs_)
        else:
            return linkType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_refid(self):
        if False:
            print('Hello World!')
        return self.refid

    def set_refid(self, refid):
        if False:
            while True:
                i = 10
        self.refid = refid

    def get_external(self):
        if False:
            for i in range(10):
                print('nop')
        return self.external

    def set_external(self, external):
        if False:
            return 10
        self.external = external

    def getValueOf_(self):
        if False:
            return 10
        return self.valueOf_

    def setValueOf_(self, valueOf_):
        if False:
            print('Hello World!')
        self.valueOf_ = valueOf_

    def export(self, outfile, level, namespace_='', name_='linkType', namespacedef_=''):
        if False:
            print('Hello World!')
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='linkType')
        if self.hasContent_():
            outfile.write('>\n')
            self.exportChildren(outfile, level + 1, namespace_, name_)
            showIndent(outfile, level)
            outfile.write('</%s%s>\n' % (namespace_, name_))
        else:
            outfile.write(' />\n')

    def exportAttributes(self, outfile, level, namespace_='', name_='linkType'):
        if False:
            i = 10
            return i + 15
        if self.refid is not None:
            outfile.write(' refid=%s' % (self.format_string(quote_attrib(self.refid).encode(ExternalEncoding), input_name='refid'),))
        if self.external is not None:
            outfile.write(' external=%s' % (self.format_string(quote_attrib(self.external).encode(ExternalEncoding), input_name='external'),))

    def exportChildren(self, outfile, level, namespace_='', name_='linkType'):
        if False:
            for i in range(10):
                print('nop')
        if self.valueOf_.find('![CDATA') > -1:
            value = quote_xml('%s' % self.valueOf_)
            value = value.replace('![CDATA', '<![CDATA')
            value = value.replace(']]', ']]>')
            outfile.write(value)
        else:
            outfile.write(quote_xml('%s' % self.valueOf_))

    def hasContent_(self):
        if False:
            print('Hello World!')
        if self.valueOf_ is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='linkType'):
        if False:
            print('Hello World!')
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            while True:
                i = 10
        if self.refid is not None:
            showIndent(outfile, level)
            outfile.write('refid = %s,\n' % (self.refid,))
        if self.external is not None:
            showIndent(outfile, level)
            outfile.write('external = %s,\n' % (self.external,))

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            i = 10
            return i + 15
        showIndent(outfile, level)
        outfile.write('valueOf_ = "%s",\n' % (self.valueOf_,))

    def build(self, node_):
        if False:
            while True:
                i = 10
        attrs = node_.attributes
        self.buildAttributes(attrs)
        self.valueOf_ = ''
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            i = 10
            return i + 15
        if attrs.get('refid'):
            self.refid = attrs.get('refid').value
        if attrs.get('external'):
            self.external = attrs.get('external').value

    def buildChildren(self, child_, nodeName_):
        if False:
            for i in range(10):
                print('nop')
        if child_.nodeType == Node.TEXT_NODE:
            self.valueOf_ += child_.nodeValue
        elif child_.nodeType == Node.CDATA_SECTION_NODE:
            self.valueOf_ += '![CDATA[' + child_.nodeValue + ']]'

class listingType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, codeline=None):
        if False:
            while True:
                i = 10
        if codeline is None:
            self.codeline = []
        else:
            self.codeline = codeline

    def factory(*args_, **kwargs_):
        if False:
            return 10
        if listingType.subclass:
            return listingType.subclass(*args_, **kwargs_)
        else:
            return listingType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_codeline(self):
        if False:
            while True:
                i = 10
        return self.codeline

    def set_codeline(self, codeline):
        if False:
            while True:
                i = 10
        self.codeline = codeline

    def add_codeline(self, value):
        if False:
            return 10
        self.codeline.append(value)

    def insert_codeline(self, index, value):
        if False:
            for i in range(10):
                print('nop')
        self.codeline[index] = value

    def export(self, outfile, level, namespace_='', name_='listingType', namespacedef_=''):
        if False:
            i = 10
            return i + 15
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='listingType')
        if self.hasContent_():
            outfile.write('>\n')
            self.exportChildren(outfile, level + 1, namespace_, name_)
            showIndent(outfile, level)
            outfile.write('</%s%s>\n' % (namespace_, name_))
        else:
            outfile.write(' />\n')

    def exportAttributes(self, outfile, level, namespace_='', name_='listingType'):
        if False:
            while True:
                i = 10
        pass

    def exportChildren(self, outfile, level, namespace_='', name_='listingType'):
        if False:
            print('Hello World!')
        for codeline_ in self.codeline:
            codeline_.export(outfile, level, namespace_, name_='codeline')

    def hasContent_(self):
        if False:
            i = 10
            return i + 15
        if self.codeline is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='listingType'):
        if False:
            print('Hello World!')
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            i = 10
            return i + 15
        pass

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            i = 10
            return i + 15
        showIndent(outfile, level)
        outfile.write('codeline=[\n')
        level += 1
        for codeline in self.codeline:
            showIndent(outfile, level)
            outfile.write('model_.codeline(\n')
            codeline.exportLiteral(outfile, level, name_='codeline')
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
            print('Hello World!')
        pass

    def buildChildren(self, child_, nodeName_):
        if False:
            for i in range(10):
                print('nop')
        if child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'codeline':
            obj_ = codelineType.factory()
            obj_.build(child_)
            self.codeline.append(obj_)

class codelineType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, external=None, lineno=None, refkind=None, refid=None, highlight=None):
        if False:
            return 10
        self.external = external
        self.lineno = lineno
        self.refkind = refkind
        self.refid = refid
        if highlight is None:
            self.highlight = []
        else:
            self.highlight = highlight

    def factory(*args_, **kwargs_):
        if False:
            i = 10
            return i + 15
        if codelineType.subclass:
            return codelineType.subclass(*args_, **kwargs_)
        else:
            return codelineType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_highlight(self):
        if False:
            print('Hello World!')
        return self.highlight

    def set_highlight(self, highlight):
        if False:
            while True:
                i = 10
        self.highlight = highlight

    def add_highlight(self, value):
        if False:
            print('Hello World!')
        self.highlight.append(value)

    def insert_highlight(self, index, value):
        if False:
            for i in range(10):
                print('nop')
        self.highlight[index] = value

    def get_external(self):
        if False:
            for i in range(10):
                print('nop')
        return self.external

    def set_external(self, external):
        if False:
            print('Hello World!')
        self.external = external

    def get_lineno(self):
        if False:
            i = 10
            return i + 15
        return self.lineno

    def set_lineno(self, lineno):
        if False:
            print('Hello World!')
        self.lineno = lineno

    def get_refkind(self):
        if False:
            return 10
        return self.refkind

    def set_refkind(self, refkind):
        if False:
            for i in range(10):
                print('nop')
        self.refkind = refkind

    def get_refid(self):
        if False:
            print('Hello World!')
        return self.refid

    def set_refid(self, refid):
        if False:
            print('Hello World!')
        self.refid = refid

    def export(self, outfile, level, namespace_='', name_='codelineType', namespacedef_=''):
        if False:
            print('Hello World!')
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='codelineType')
        if self.hasContent_():
            outfile.write('>\n')
            self.exportChildren(outfile, level + 1, namespace_, name_)
            showIndent(outfile, level)
            outfile.write('</%s%s>\n' % (namespace_, name_))
        else:
            outfile.write(' />\n')

    def exportAttributes(self, outfile, level, namespace_='', name_='codelineType'):
        if False:
            i = 10
            return i + 15
        if self.external is not None:
            outfile.write(' external=%s' % (quote_attrib(self.external),))
        if self.lineno is not None:
            outfile.write(' lineno="%s"' % self.format_integer(self.lineno, input_name='lineno'))
        if self.refkind is not None:
            outfile.write(' refkind=%s' % (quote_attrib(self.refkind),))
        if self.refid is not None:
            outfile.write(' refid=%s' % (self.format_string(quote_attrib(self.refid).encode(ExternalEncoding), input_name='refid'),))

    def exportChildren(self, outfile, level, namespace_='', name_='codelineType'):
        if False:
            print('Hello World!')
        for highlight_ in self.highlight:
            highlight_.export(outfile, level, namespace_, name_='highlight')

    def hasContent_(self):
        if False:
            for i in range(10):
                print('nop')
        if self.highlight is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='codelineType'):
        if False:
            while True:
                i = 10
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            i = 10
            return i + 15
        if self.external is not None:
            showIndent(outfile, level)
            outfile.write('external = "%s",\n' % (self.external,))
        if self.lineno is not None:
            showIndent(outfile, level)
            outfile.write('lineno = %s,\n' % (self.lineno,))
        if self.refkind is not None:
            showIndent(outfile, level)
            outfile.write('refkind = "%s",\n' % (self.refkind,))
        if self.refid is not None:
            showIndent(outfile, level)
            outfile.write('refid = %s,\n' % (self.refid,))

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            i = 10
            return i + 15
        showIndent(outfile, level)
        outfile.write('highlight=[\n')
        level += 1
        for highlight in self.highlight:
            showIndent(outfile, level)
            outfile.write('model_.highlight(\n')
            highlight.exportLiteral(outfile, level, name_='highlight')
            showIndent(outfile, level)
            outfile.write('),\n')
        level -= 1
        showIndent(outfile, level)
        outfile.write('],\n')

    def build(self, node_):
        if False:
            i = 10
            return i + 15
        attrs = node_.attributes
        self.buildAttributes(attrs)
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            while True:
                i = 10
        if attrs.get('external'):
            self.external = attrs.get('external').value
        if attrs.get('lineno'):
            try:
                self.lineno = int(attrs.get('lineno').value)
            except ValueError as exp:
                raise ValueError('Bad integer attribute (lineno): %s' % exp)
        if attrs.get('refkind'):
            self.refkind = attrs.get('refkind').value
        if attrs.get('refid'):
            self.refid = attrs.get('refid').value

    def buildChildren(self, child_, nodeName_):
        if False:
            while True:
                i = 10
        if child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'highlight':
            obj_ = highlightType.factory()
            obj_.build(child_)
            self.highlight.append(obj_)

class highlightType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, classxx=None, sp=None, ref=None, mixedclass_=None, content_=None):
        if False:
            print('Hello World!')
        self.classxx = classxx
        if mixedclass_ is None:
            self.mixedclass_ = MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        if content_ is None:
            self.content_ = []
        else:
            self.content_ = content_

    def factory(*args_, **kwargs_):
        if False:
            i = 10
            return i + 15
        if highlightType.subclass:
            return highlightType.subclass(*args_, **kwargs_)
        else:
            return highlightType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_sp(self):
        if False:
            i = 10
            return i + 15
        return self.sp

    def set_sp(self, sp):
        if False:
            for i in range(10):
                print('nop')
        self.sp = sp

    def add_sp(self, value):
        if False:
            return 10
        self.sp.append(value)

    def insert_sp(self, index, value):
        if False:
            while True:
                i = 10
        self.sp[index] = value

    def get_ref(self):
        if False:
            while True:
                i = 10
        return self.ref

    def set_ref(self, ref):
        if False:
            print('Hello World!')
        self.ref = ref

    def add_ref(self, value):
        if False:
            while True:
                i = 10
        self.ref.append(value)

    def insert_ref(self, index, value):
        if False:
            return 10
        self.ref[index] = value

    def get_class(self):
        if False:
            for i in range(10):
                print('nop')
        return self.classxx

    def set_class(self, classxx):
        if False:
            print('Hello World!')
        self.classxx = classxx

    def export(self, outfile, level, namespace_='', name_='highlightType', namespacedef_=''):
        if False:
            print('Hello World!')
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='highlightType')
        outfile.write('>')
        self.exportChildren(outfile, level + 1, namespace_, name_)
        outfile.write('</%s%s>\n' % (namespace_, name_))

    def exportAttributes(self, outfile, level, namespace_='', name_='highlightType'):
        if False:
            return 10
        if self.classxx is not None:
            outfile.write(' class=%s' % (quote_attrib(self.classxx),))

    def exportChildren(self, outfile, level, namespace_='', name_='highlightType'):
        if False:
            for i in range(10):
                print('nop')
        for item_ in self.content_:
            item_.export(outfile, level, item_.name, namespace_)

    def hasContent_(self):
        if False:
            i = 10
            return i + 15
        if self.sp is not None or self.ref is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='highlightType'):
        if False:
            return 10
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            print('Hello World!')
        if self.classxx is not None:
            showIndent(outfile, level)
            outfile.write('classxx = "%s",\n' % (self.classxx,))

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            return 10
        showIndent(outfile, level)
        outfile.write('content_ = [\n')
        for item_ in self.content_:
            item_.exportLiteral(outfile, level, name_)
        showIndent(outfile, level)
        outfile.write('],\n')
        showIndent(outfile, level)
        outfile.write('content_ = [\n')
        for item_ in self.content_:
            item_.exportLiteral(outfile, level, name_)
        showIndent(outfile, level)
        outfile.write('],\n')

    def build(self, node_):
        if False:
            i = 10
            return i + 15
        attrs = node_.attributes
        self.buildAttributes(attrs)
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            while True:
                i = 10
        if attrs.get('class'):
            self.classxx = attrs.get('class').value

    def buildChildren(self, child_, nodeName_):
        if False:
            for i in range(10):
                print('nop')
        if child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'sp':
            value_ = []
            for text_ in child_.childNodes:
                value_.append(text_.nodeValue)
            valuestr_ = ''.join(value_)
            obj_ = self.mixedclass_(MixedContainer.CategorySimple, MixedContainer.TypeString, 'sp', valuestr_)
            self.content_.append(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'ref':
            childobj_ = docRefTextType.factory()
            childobj_.build(child_)
            obj_ = self.mixedclass_(MixedContainer.CategoryComplex, MixedContainer.TypeNone, 'ref', childobj_)
            self.content_.append(obj_)
        elif child_.nodeType == Node.TEXT_NODE:
            obj_ = self.mixedclass_(MixedContainer.CategoryText, MixedContainer.TypeNone, '', child_.nodeValue)
            self.content_.append(obj_)

class sp(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, valueOf_=''):
        if False:
            while True:
                i = 10
        self.valueOf_ = valueOf_

    def factory(*args_, **kwargs_):
        if False:
            return 10
        if sp.subclass:
            return sp.subclass(*args_, **kwargs_)
        else:
            return sp(*args_, **kwargs_)
    factory = staticmethod(factory)

    def getValueOf_(self):
        if False:
            while True:
                i = 10
        return self.valueOf_

    def setValueOf_(self, valueOf_):
        if False:
            return 10
        self.valueOf_ = valueOf_

    def export(self, outfile, level, namespace_='', name_='sp', namespacedef_=''):
        if False:
            i = 10
            return i + 15
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='sp')
        if self.hasContent_():
            outfile.write('>\n')
            self.exportChildren(outfile, level + 1, namespace_, name_)
            showIndent(outfile, level)
            outfile.write('</%s%s>\n' % (namespace_, name_))
        else:
            outfile.write(' />\n')

    def exportAttributes(self, outfile, level, namespace_='', name_='sp'):
        if False:
            while True:
                i = 10
        pass

    def exportChildren(self, outfile, level, namespace_='', name_='sp'):
        if False:
            i = 10
            return i + 15
        if self.valueOf_.find('![CDATA') > -1:
            value = quote_xml('%s' % self.valueOf_)
            value = value.replace('![CDATA', '<![CDATA')
            value = value.replace(']]', ']]>')
            outfile.write(value)
        else:
            outfile.write(quote_xml('%s' % self.valueOf_))

    def hasContent_(self):
        if False:
            for i in range(10):
                print('nop')
        if self.valueOf_ is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='sp'):
        if False:
            return 10
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            return 10
        pass

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            i = 10
            return i + 15
        showIndent(outfile, level)
        outfile.write('valueOf_ = "%s",\n' % (self.valueOf_,))

    def build(self, node_):
        if False:
            return 10
        attrs = node_.attributes
        self.buildAttributes(attrs)
        self.valueOf_ = ''
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            for i in range(10):
                print('nop')
        pass

    def buildChildren(self, child_, nodeName_):
        if False:
            return 10
        if child_.nodeType == Node.TEXT_NODE:
            self.valueOf_ += child_.nodeValue
        elif child_.nodeType == Node.CDATA_SECTION_NODE:
            self.valueOf_ += '![CDATA[' + child_.nodeValue + ']]'

class referenceType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, endline=None, startline=None, refid=None, compoundref=None, valueOf_='', mixedclass_=None, content_=None):
        if False:
            return 10
        self.endline = endline
        self.startline = startline
        self.refid = refid
        self.compoundref = compoundref
        if mixedclass_ is None:
            self.mixedclass_ = MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        if content_ is None:
            self.content_ = []
        else:
            self.content_ = content_

    def factory(*args_, **kwargs_):
        if False:
            return 10
        if referenceType.subclass:
            return referenceType.subclass(*args_, **kwargs_)
        else:
            return referenceType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_endline(self):
        if False:
            return 10
        return self.endline

    def set_endline(self, endline):
        if False:
            print('Hello World!')
        self.endline = endline

    def get_startline(self):
        if False:
            while True:
                i = 10
        return self.startline

    def set_startline(self, startline):
        if False:
            return 10
        self.startline = startline

    def get_refid(self):
        if False:
            return 10
        return self.refid

    def set_refid(self, refid):
        if False:
            i = 10
            return i + 15
        self.refid = refid

    def get_compoundref(self):
        if False:
            return 10
        return self.compoundref

    def set_compoundref(self, compoundref):
        if False:
            for i in range(10):
                print('nop')
        self.compoundref = compoundref

    def getValueOf_(self):
        if False:
            i = 10
            return i + 15
        return self.valueOf_

    def setValueOf_(self, valueOf_):
        if False:
            return 10
        self.valueOf_ = valueOf_

    def export(self, outfile, level, namespace_='', name_='referenceType', namespacedef_=''):
        if False:
            while True:
                i = 10
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='referenceType')
        outfile.write('>')
        self.exportChildren(outfile, level + 1, namespace_, name_)
        outfile.write('</%s%s>\n' % (namespace_, name_))

    def exportAttributes(self, outfile, level, namespace_='', name_='referenceType'):
        if False:
            i = 10
            return i + 15
        if self.endline is not None:
            outfile.write(' endline="%s"' % self.format_integer(self.endline, input_name='endline'))
        if self.startline is not None:
            outfile.write(' startline="%s"' % self.format_integer(self.startline, input_name='startline'))
        if self.refid is not None:
            outfile.write(' refid=%s' % (self.format_string(quote_attrib(self.refid).encode(ExternalEncoding), input_name='refid'),))
        if self.compoundref is not None:
            outfile.write(' compoundref=%s' % (self.format_string(quote_attrib(self.compoundref).encode(ExternalEncoding), input_name='compoundref'),))

    def exportChildren(self, outfile, level, namespace_='', name_='referenceType'):
        if False:
            return 10
        if self.valueOf_.find('![CDATA') > -1:
            value = quote_xml('%s' % self.valueOf_)
            value = value.replace('![CDATA', '<![CDATA')
            value = value.replace(']]', ']]>')
            outfile.write(value)
        else:
            outfile.write(quote_xml('%s' % self.valueOf_))

    def hasContent_(self):
        if False:
            print('Hello World!')
        if self.valueOf_ is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='referenceType'):
        if False:
            print('Hello World!')
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            return 10
        if self.endline is not None:
            showIndent(outfile, level)
            outfile.write('endline = %s,\n' % (self.endline,))
        if self.startline is not None:
            showIndent(outfile, level)
            outfile.write('startline = %s,\n' % (self.startline,))
        if self.refid is not None:
            showIndent(outfile, level)
            outfile.write('refid = %s,\n' % (self.refid,))
        if self.compoundref is not None:
            showIndent(outfile, level)
            outfile.write('compoundref = %s,\n' % (self.compoundref,))

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            i = 10
            return i + 15
        showIndent(outfile, level)
        outfile.write('valueOf_ = "%s",\n' % (self.valueOf_,))

    def build(self, node_):
        if False:
            print('Hello World!')
        attrs = node_.attributes
        self.buildAttributes(attrs)
        self.valueOf_ = ''
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            while True:
                i = 10
        if attrs.get('endline'):
            try:
                self.endline = int(attrs.get('endline').value)
            except ValueError as exp:
                raise ValueError('Bad integer attribute (endline): %s' % exp)
        if attrs.get('startline'):
            try:
                self.startline = int(attrs.get('startline').value)
            except ValueError as exp:
                raise ValueError('Bad integer attribute (startline): %s' % exp)
        if attrs.get('refid'):
            self.refid = attrs.get('refid').value
        if attrs.get('compoundref'):
            self.compoundref = attrs.get('compoundref').value

    def buildChildren(self, child_, nodeName_):
        if False:
            while True:
                i = 10
        if child_.nodeType == Node.TEXT_NODE:
            obj_ = self.mixedclass_(MixedContainer.CategoryText, MixedContainer.TypeNone, '', child_.nodeValue)
            self.content_.append(obj_)
        if child_.nodeType == Node.TEXT_NODE:
            self.valueOf_ += child_.nodeValue
        elif child_.nodeType == Node.CDATA_SECTION_NODE:
            self.valueOf_ += '![CDATA[' + child_.nodeValue + ']]'

class locationType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, bodystart=None, line=None, bodyend=None, bodyfile=None, file=None, valueOf_=''):
        if False:
            i = 10
            return i + 15
        self.bodystart = bodystart
        self.line = line
        self.bodyend = bodyend
        self.bodyfile = bodyfile
        self.file = file
        self.valueOf_ = valueOf_

    def factory(*args_, **kwargs_):
        if False:
            i = 10
            return i + 15
        if locationType.subclass:
            return locationType.subclass(*args_, **kwargs_)
        else:
            return locationType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_bodystart(self):
        if False:
            i = 10
            return i + 15
        return self.bodystart

    def set_bodystart(self, bodystart):
        if False:
            for i in range(10):
                print('nop')
        self.bodystart = bodystart

    def get_line(self):
        if False:
            i = 10
            return i + 15
        return self.line

    def set_line(self, line):
        if False:
            return 10
        self.line = line

    def get_bodyend(self):
        if False:
            return 10
        return self.bodyend

    def set_bodyend(self, bodyend):
        if False:
            i = 10
            return i + 15
        self.bodyend = bodyend

    def get_bodyfile(self):
        if False:
            i = 10
            return i + 15
        return self.bodyfile

    def set_bodyfile(self, bodyfile):
        if False:
            return 10
        self.bodyfile = bodyfile

    def get_file(self):
        if False:
            print('Hello World!')
        return self.file

    def set_file(self, file):
        if False:
            while True:
                i = 10
        self.file = file

    def getValueOf_(self):
        if False:
            while True:
                i = 10
        return self.valueOf_

    def setValueOf_(self, valueOf_):
        if False:
            print('Hello World!')
        self.valueOf_ = valueOf_

    def export(self, outfile, level, namespace_='', name_='locationType', namespacedef_=''):
        if False:
            for i in range(10):
                print('nop')
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='locationType')
        if self.hasContent_():
            outfile.write('>\n')
            self.exportChildren(outfile, level + 1, namespace_, name_)
            showIndent(outfile, level)
            outfile.write('</%s%s>\n' % (namespace_, name_))
        else:
            outfile.write(' />\n')

    def exportAttributes(self, outfile, level, namespace_='', name_='locationType'):
        if False:
            for i in range(10):
                print('nop')
        if self.bodystart is not None:
            outfile.write(' bodystart="%s"' % self.format_integer(self.bodystart, input_name='bodystart'))
        if self.line is not None:
            outfile.write(' line="%s"' % self.format_integer(self.line, input_name='line'))
        if self.bodyend is not None:
            outfile.write(' bodyend="%s"' % self.format_integer(self.bodyend, input_name='bodyend'))
        if self.bodyfile is not None:
            outfile.write(' bodyfile=%s' % (self.format_string(quote_attrib(self.bodyfile).encode(ExternalEncoding), input_name='bodyfile'),))
        if self.file is not None:
            outfile.write(' file=%s' % (self.format_string(quote_attrib(self.file).encode(ExternalEncoding), input_name='file'),))

    def exportChildren(self, outfile, level, namespace_='', name_='locationType'):
        if False:
            print('Hello World!')
        if self.valueOf_.find('![CDATA') > -1:
            value = quote_xml('%s' % self.valueOf_)
            value = value.replace('![CDATA', '<![CDATA')
            value = value.replace(']]', ']]>')
            outfile.write(value)
        else:
            outfile.write(quote_xml('%s' % self.valueOf_))

    def hasContent_(self):
        if False:
            for i in range(10):
                print('nop')
        if self.valueOf_ is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='locationType'):
        if False:
            for i in range(10):
                print('nop')
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            while True:
                i = 10
        if self.bodystart is not None:
            showIndent(outfile, level)
            outfile.write('bodystart = %s,\n' % (self.bodystart,))
        if self.line is not None:
            showIndent(outfile, level)
            outfile.write('line = %s,\n' % (self.line,))
        if self.bodyend is not None:
            showIndent(outfile, level)
            outfile.write('bodyend = %s,\n' % (self.bodyend,))
        if self.bodyfile is not None:
            showIndent(outfile, level)
            outfile.write('bodyfile = %s,\n' % (self.bodyfile,))
        if self.file is not None:
            showIndent(outfile, level)
            outfile.write('file = %s,\n' % (self.file,))

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            i = 10
            return i + 15
        showIndent(outfile, level)
        outfile.write('valueOf_ = "%s",\n' % (self.valueOf_,))

    def build(self, node_):
        if False:
            for i in range(10):
                print('nop')
        attrs = node_.attributes
        self.buildAttributes(attrs)
        self.valueOf_ = ''
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            print('Hello World!')
        if attrs.get('bodystart'):
            try:
                self.bodystart = int(attrs.get('bodystart').value)
            except ValueError as exp:
                raise ValueError('Bad integer attribute (bodystart): %s' % exp)
        if attrs.get('line'):
            try:
                self.line = int(attrs.get('line').value)
            except ValueError as exp:
                raise ValueError('Bad integer attribute (line): %s' % exp)
        if attrs.get('bodyend'):
            try:
                self.bodyend = int(attrs.get('bodyend').value)
            except ValueError as exp:
                raise ValueError('Bad integer attribute (bodyend): %s' % exp)
        if attrs.get('bodyfile'):
            self.bodyfile = attrs.get('bodyfile').value
        if attrs.get('file'):
            self.file = attrs.get('file').value

    def buildChildren(self, child_, nodeName_):
        if False:
            while True:
                i = 10
        if child_.nodeType == Node.TEXT_NODE:
            self.valueOf_ += child_.nodeValue
        elif child_.nodeType == Node.CDATA_SECTION_NODE:
            self.valueOf_ += '![CDATA[' + child_.nodeValue + ']]'

class docSect1Type(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, id=None, title=None, para=None, sect2=None, internal=None, mixedclass_=None, content_=None):
        if False:
            print('Hello World!')
        self.id = id
        if mixedclass_ is None:
            self.mixedclass_ = MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        if content_ is None:
            self.content_ = []
        else:
            self.content_ = content_

    def factory(*args_, **kwargs_):
        if False:
            while True:
                i = 10
        if docSect1Type.subclass:
            return docSect1Type.subclass(*args_, **kwargs_)
        else:
            return docSect1Type(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_title(self):
        if False:
            return 10
        return self.title

    def set_title(self, title):
        if False:
            i = 10
            return i + 15
        self.title = title

    def get_para(self):
        if False:
            while True:
                i = 10
        return self.para

    def set_para(self, para):
        if False:
            i = 10
            return i + 15
        self.para = para

    def add_para(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.para.append(value)

    def insert_para(self, index, value):
        if False:
            while True:
                i = 10
        self.para[index] = value

    def get_sect2(self):
        if False:
            i = 10
            return i + 15
        return self.sect2

    def set_sect2(self, sect2):
        if False:
            i = 10
            return i + 15
        self.sect2 = sect2

    def add_sect2(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.sect2.append(value)

    def insert_sect2(self, index, value):
        if False:
            i = 10
            return i + 15
        self.sect2[index] = value

    def get_internal(self):
        if False:
            for i in range(10):
                print('nop')
        return self.internal

    def set_internal(self, internal):
        if False:
            print('Hello World!')
        self.internal = internal

    def get_id(self):
        if False:
            while True:
                i = 10
        return self.id

    def set_id(self, id):
        if False:
            return 10
        self.id = id

    def export(self, outfile, level, namespace_='', name_='docSect1Type', namespacedef_=''):
        if False:
            i = 10
            return i + 15
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='docSect1Type')
        outfile.write('>')
        self.exportChildren(outfile, level + 1, namespace_, name_)
        outfile.write('</%s%s>\n' % (namespace_, name_))

    def exportAttributes(self, outfile, level, namespace_='', name_='docSect1Type'):
        if False:
            return 10
        if self.id is not None:
            outfile.write(' id=%s' % (self.format_string(quote_attrib(self.id).encode(ExternalEncoding), input_name='id'),))

    def exportChildren(self, outfile, level, namespace_='', name_='docSect1Type'):
        if False:
            for i in range(10):
                print('nop')
        for item_ in self.content_:
            item_.export(outfile, level, item_.name, namespace_)

    def hasContent_(self):
        if False:
            return 10
        if self.title is not None or self.para is not None or self.sect2 is not None or (self.internal is not None):
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='docSect1Type'):
        if False:
            print('Hello World!')
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            for i in range(10):
                print('nop')
        if self.id is not None:
            showIndent(outfile, level)
            outfile.write('id = %s,\n' % (self.id,))

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            while True:
                i = 10
        showIndent(outfile, level)
        outfile.write('content_ = [\n')
        for item_ in self.content_:
            item_.exportLiteral(outfile, level, name_)
        showIndent(outfile, level)
        outfile.write('],\n')
        showIndent(outfile, level)
        outfile.write('content_ = [\n')
        for item_ in self.content_:
            item_.exportLiteral(outfile, level, name_)
        showIndent(outfile, level)
        outfile.write('],\n')
        showIndent(outfile, level)
        outfile.write('content_ = [\n')
        for item_ in self.content_:
            item_.exportLiteral(outfile, level, name_)
        showIndent(outfile, level)
        outfile.write('],\n')
        showIndent(outfile, level)
        outfile.write('content_ = [\n')
        for item_ in self.content_:
            item_.exportLiteral(outfile, level, name_)
        showIndent(outfile, level)
        outfile.write('],\n')

    def build(self, node_):
        if False:
            i = 10
            return i + 15
        attrs = node_.attributes
        self.buildAttributes(attrs)
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            for i in range(10):
                print('nop')
        if attrs.get('id'):
            self.id = attrs.get('id').value

    def buildChildren(self, child_, nodeName_):
        if False:
            i = 10
            return i + 15
        if child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'title':
            childobj_ = docTitleType.factory()
            childobj_.build(child_)
            obj_ = self.mixedclass_(MixedContainer.CategoryComplex, MixedContainer.TypeNone, 'title', childobj_)
            self.content_.append(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'para':
            childobj_ = docParaType.factory()
            childobj_.build(child_)
            obj_ = self.mixedclass_(MixedContainer.CategoryComplex, MixedContainer.TypeNone, 'para', childobj_)
            self.content_.append(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'sect2':
            childobj_ = docSect2Type.factory()
            childobj_.build(child_)
            obj_ = self.mixedclass_(MixedContainer.CategoryComplex, MixedContainer.TypeNone, 'sect2', childobj_)
            self.content_.append(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'internal':
            childobj_ = docInternalS1Type.factory()
            childobj_.build(child_)
            obj_ = self.mixedclass_(MixedContainer.CategoryComplex, MixedContainer.TypeNone, 'internal', childobj_)
            self.content_.append(obj_)
        elif child_.nodeType == Node.TEXT_NODE:
            obj_ = self.mixedclass_(MixedContainer.CategoryText, MixedContainer.TypeNone, '', child_.nodeValue)
            self.content_.append(obj_)

class docSect2Type(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, id=None, title=None, para=None, sect3=None, internal=None, mixedclass_=None, content_=None):
        if False:
            for i in range(10):
                print('nop')
        self.id = id
        if mixedclass_ is None:
            self.mixedclass_ = MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        if content_ is None:
            self.content_ = []
        else:
            self.content_ = content_

    def factory(*args_, **kwargs_):
        if False:
            return 10
        if docSect2Type.subclass:
            return docSect2Type.subclass(*args_, **kwargs_)
        else:
            return docSect2Type(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_title(self):
        if False:
            for i in range(10):
                print('nop')
        return self.title

    def set_title(self, title):
        if False:
            print('Hello World!')
        self.title = title

    def get_para(self):
        if False:
            return 10
        return self.para

    def set_para(self, para):
        if False:
            i = 10
            return i + 15
        self.para = para

    def add_para(self, value):
        if False:
            return 10
        self.para.append(value)

    def insert_para(self, index, value):
        if False:
            for i in range(10):
                print('nop')
        self.para[index] = value

    def get_sect3(self):
        if False:
            return 10
        return self.sect3

    def set_sect3(self, sect3):
        if False:
            return 10
        self.sect3 = sect3

    def add_sect3(self, value):
        if False:
            i = 10
            return i + 15
        self.sect3.append(value)

    def insert_sect3(self, index, value):
        if False:
            print('Hello World!')
        self.sect3[index] = value

    def get_internal(self):
        if False:
            print('Hello World!')
        return self.internal

    def set_internal(self, internal):
        if False:
            print('Hello World!')
        self.internal = internal

    def get_id(self):
        if False:
            while True:
                i = 10
        return self.id

    def set_id(self, id):
        if False:
            for i in range(10):
                print('nop')
        self.id = id

    def export(self, outfile, level, namespace_='', name_='docSect2Type', namespacedef_=''):
        if False:
            while True:
                i = 10
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='docSect2Type')
        outfile.write('>')
        self.exportChildren(outfile, level + 1, namespace_, name_)
        outfile.write('</%s%s>\n' % (namespace_, name_))

    def exportAttributes(self, outfile, level, namespace_='', name_='docSect2Type'):
        if False:
            while True:
                i = 10
        if self.id is not None:
            outfile.write(' id=%s' % (self.format_string(quote_attrib(self.id).encode(ExternalEncoding), input_name='id'),))

    def exportChildren(self, outfile, level, namespace_='', name_='docSect2Type'):
        if False:
            print('Hello World!')
        for item_ in self.content_:
            item_.export(outfile, level, item_.name, namespace_)

    def hasContent_(self):
        if False:
            print('Hello World!')
        if self.title is not None or self.para is not None or self.sect3 is not None or (self.internal is not None):
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='docSect2Type'):
        if False:
            print('Hello World!')
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            for i in range(10):
                print('nop')
        if self.id is not None:
            showIndent(outfile, level)
            outfile.write('id = %s,\n' % (self.id,))

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            i = 10
            return i + 15
        showIndent(outfile, level)
        outfile.write('content_ = [\n')
        for item_ in self.content_:
            item_.exportLiteral(outfile, level, name_)
        showIndent(outfile, level)
        outfile.write('],\n')
        showIndent(outfile, level)
        outfile.write('content_ = [\n')
        for item_ in self.content_:
            item_.exportLiteral(outfile, level, name_)
        showIndent(outfile, level)
        outfile.write('],\n')
        showIndent(outfile, level)
        outfile.write('content_ = [\n')
        for item_ in self.content_:
            item_.exportLiteral(outfile, level, name_)
        showIndent(outfile, level)
        outfile.write('],\n')
        showIndent(outfile, level)
        outfile.write('content_ = [\n')
        for item_ in self.content_:
            item_.exportLiteral(outfile, level, name_)
        showIndent(outfile, level)
        outfile.write('],\n')

    def build(self, node_):
        if False:
            return 10
        attrs = node_.attributes
        self.buildAttributes(attrs)
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            return 10
        if attrs.get('id'):
            self.id = attrs.get('id').value

    def buildChildren(self, child_, nodeName_):
        if False:
            return 10
        if child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'title':
            childobj_ = docTitleType.factory()
            childobj_.build(child_)
            obj_ = self.mixedclass_(MixedContainer.CategoryComplex, MixedContainer.TypeNone, 'title', childobj_)
            self.content_.append(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'para':
            childobj_ = docParaType.factory()
            childobj_.build(child_)
            obj_ = self.mixedclass_(MixedContainer.CategoryComplex, MixedContainer.TypeNone, 'para', childobj_)
            self.content_.append(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'sect3':
            childobj_ = docSect3Type.factory()
            childobj_.build(child_)
            obj_ = self.mixedclass_(MixedContainer.CategoryComplex, MixedContainer.TypeNone, 'sect3', childobj_)
            self.content_.append(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'internal':
            childobj_ = docInternalS2Type.factory()
            childobj_.build(child_)
            obj_ = self.mixedclass_(MixedContainer.CategoryComplex, MixedContainer.TypeNone, 'internal', childobj_)
            self.content_.append(obj_)
        elif child_.nodeType == Node.TEXT_NODE:
            obj_ = self.mixedclass_(MixedContainer.CategoryText, MixedContainer.TypeNone, '', child_.nodeValue)
            self.content_.append(obj_)

class docSect3Type(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, id=None, title=None, para=None, sect4=None, internal=None, mixedclass_=None, content_=None):
        if False:
            for i in range(10):
                print('nop')
        self.id = id
        if mixedclass_ is None:
            self.mixedclass_ = MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        if content_ is None:
            self.content_ = []
        else:
            self.content_ = content_

    def factory(*args_, **kwargs_):
        if False:
            for i in range(10):
                print('nop')
        if docSect3Type.subclass:
            return docSect3Type.subclass(*args_, **kwargs_)
        else:
            return docSect3Type(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_title(self):
        if False:
            for i in range(10):
                print('nop')
        return self.title

    def set_title(self, title):
        if False:
            i = 10
            return i + 15
        self.title = title

    def get_para(self):
        if False:
            print('Hello World!')
        return self.para

    def set_para(self, para):
        if False:
            return 10
        self.para = para

    def add_para(self, value):
        if False:
            i = 10
            return i + 15
        self.para.append(value)

    def insert_para(self, index, value):
        if False:
            while True:
                i = 10
        self.para[index] = value

    def get_sect4(self):
        if False:
            i = 10
            return i + 15
        return self.sect4

    def set_sect4(self, sect4):
        if False:
            return 10
        self.sect4 = sect4

    def add_sect4(self, value):
        if False:
            print('Hello World!')
        self.sect4.append(value)

    def insert_sect4(self, index, value):
        if False:
            while True:
                i = 10
        self.sect4[index] = value

    def get_internal(self):
        if False:
            for i in range(10):
                print('nop')
        return self.internal

    def set_internal(self, internal):
        if False:
            print('Hello World!')
        self.internal = internal

    def get_id(self):
        if False:
            print('Hello World!')
        return self.id

    def set_id(self, id):
        if False:
            print('Hello World!')
        self.id = id

    def export(self, outfile, level, namespace_='', name_='docSect3Type', namespacedef_=''):
        if False:
            i = 10
            return i + 15
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='docSect3Type')
        outfile.write('>')
        self.exportChildren(outfile, level + 1, namespace_, name_)
        outfile.write('</%s%s>\n' % (namespace_, name_))

    def exportAttributes(self, outfile, level, namespace_='', name_='docSect3Type'):
        if False:
            print('Hello World!')
        if self.id is not None:
            outfile.write(' id=%s' % (self.format_string(quote_attrib(self.id).encode(ExternalEncoding), input_name='id'),))

    def exportChildren(self, outfile, level, namespace_='', name_='docSect3Type'):
        if False:
            for i in range(10):
                print('nop')
        for item_ in self.content_:
            item_.export(outfile, level, item_.name, namespace_)

    def hasContent_(self):
        if False:
            i = 10
            return i + 15
        if self.title is not None or self.para is not None or self.sect4 is not None or (self.internal is not None):
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='docSect3Type'):
        if False:
            i = 10
            return i + 15
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            return 10
        if self.id is not None:
            showIndent(outfile, level)
            outfile.write('id = %s,\n' % (self.id,))

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            print('Hello World!')
        showIndent(outfile, level)
        outfile.write('content_ = [\n')
        for item_ in self.content_:
            item_.exportLiteral(outfile, level, name_)
        showIndent(outfile, level)
        outfile.write('],\n')
        showIndent(outfile, level)
        outfile.write('content_ = [\n')
        for item_ in self.content_:
            item_.exportLiteral(outfile, level, name_)
        showIndent(outfile, level)
        outfile.write('],\n')
        showIndent(outfile, level)
        outfile.write('content_ = [\n')
        for item_ in self.content_:
            item_.exportLiteral(outfile, level, name_)
        showIndent(outfile, level)
        outfile.write('],\n')
        showIndent(outfile, level)
        outfile.write('content_ = [\n')
        for item_ in self.content_:
            item_.exportLiteral(outfile, level, name_)
        showIndent(outfile, level)
        outfile.write('],\n')

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
            for i in range(10):
                print('nop')
        if attrs.get('id'):
            self.id = attrs.get('id').value

    def buildChildren(self, child_, nodeName_):
        if False:
            for i in range(10):
                print('nop')
        if child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'title':
            childobj_ = docTitleType.factory()
            childobj_.build(child_)
            obj_ = self.mixedclass_(MixedContainer.CategoryComplex, MixedContainer.TypeNone, 'title', childobj_)
            self.content_.append(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'para':
            childobj_ = docParaType.factory()
            childobj_.build(child_)
            obj_ = self.mixedclass_(MixedContainer.CategoryComplex, MixedContainer.TypeNone, 'para', childobj_)
            self.content_.append(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'sect4':
            childobj_ = docSect4Type.factory()
            childobj_.build(child_)
            obj_ = self.mixedclass_(MixedContainer.CategoryComplex, MixedContainer.TypeNone, 'sect4', childobj_)
            self.content_.append(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'internal':
            childobj_ = docInternalS3Type.factory()
            childobj_.build(child_)
            obj_ = self.mixedclass_(MixedContainer.CategoryComplex, MixedContainer.TypeNone, 'internal', childobj_)
            self.content_.append(obj_)
        elif child_.nodeType == Node.TEXT_NODE:
            obj_ = self.mixedclass_(MixedContainer.CategoryText, MixedContainer.TypeNone, '', child_.nodeValue)
            self.content_.append(obj_)

class docSect4Type(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, id=None, title=None, para=None, internal=None, mixedclass_=None, content_=None):
        if False:
            return 10
        self.id = id
        if mixedclass_ is None:
            self.mixedclass_ = MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        if content_ is None:
            self.content_ = []
        else:
            self.content_ = content_

    def factory(*args_, **kwargs_):
        if False:
            i = 10
            return i + 15
        if docSect4Type.subclass:
            return docSect4Type.subclass(*args_, **kwargs_)
        else:
            return docSect4Type(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_title(self):
        if False:
            return 10
        return self.title

    def set_title(self, title):
        if False:
            i = 10
            return i + 15
        self.title = title

    def get_para(self):
        if False:
            for i in range(10):
                print('nop')
        return self.para

    def set_para(self, para):
        if False:
            return 10
        self.para = para

    def add_para(self, value):
        if False:
            while True:
                i = 10
        self.para.append(value)

    def insert_para(self, index, value):
        if False:
            for i in range(10):
                print('nop')
        self.para[index] = value

    def get_internal(self):
        if False:
            return 10
        return self.internal

    def set_internal(self, internal):
        if False:
            i = 10
            return i + 15
        self.internal = internal

    def get_id(self):
        if False:
            i = 10
            return i + 15
        return self.id

    def set_id(self, id):
        if False:
            while True:
                i = 10
        self.id = id

    def export(self, outfile, level, namespace_='', name_='docSect4Type', namespacedef_=''):
        if False:
            print('Hello World!')
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='docSect4Type')
        outfile.write('>')
        self.exportChildren(outfile, level + 1, namespace_, name_)
        outfile.write('</%s%s>\n' % (namespace_, name_))

    def exportAttributes(self, outfile, level, namespace_='', name_='docSect4Type'):
        if False:
            i = 10
            return i + 15
        if self.id is not None:
            outfile.write(' id=%s' % (self.format_string(quote_attrib(self.id).encode(ExternalEncoding), input_name='id'),))

    def exportChildren(self, outfile, level, namespace_='', name_='docSect4Type'):
        if False:
            return 10
        for item_ in self.content_:
            item_.export(outfile, level, item_.name, namespace_)

    def hasContent_(self):
        if False:
            return 10
        if self.title is not None or self.para is not None or self.internal is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='docSect4Type'):
        if False:
            while True:
                i = 10
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            i = 10
            return i + 15
        if self.id is not None:
            showIndent(outfile, level)
            outfile.write('id = %s,\n' % (self.id,))

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            for i in range(10):
                print('nop')
        showIndent(outfile, level)
        outfile.write('content_ = [\n')
        for item_ in self.content_:
            item_.exportLiteral(outfile, level, name_)
        showIndent(outfile, level)
        outfile.write('],\n')
        showIndent(outfile, level)
        outfile.write('content_ = [\n')
        for item_ in self.content_:
            item_.exportLiteral(outfile, level, name_)
        showIndent(outfile, level)
        outfile.write('],\n')
        showIndent(outfile, level)
        outfile.write('content_ = [\n')
        for item_ in self.content_:
            item_.exportLiteral(outfile, level, name_)
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
            print('Hello World!')
        if attrs.get('id'):
            self.id = attrs.get('id').value

    def buildChildren(self, child_, nodeName_):
        if False:
            return 10
        if child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'title':
            childobj_ = docTitleType.factory()
            childobj_.build(child_)
            obj_ = self.mixedclass_(MixedContainer.CategoryComplex, MixedContainer.TypeNone, 'title', childobj_)
            self.content_.append(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'para':
            childobj_ = docParaType.factory()
            childobj_.build(child_)
            obj_ = self.mixedclass_(MixedContainer.CategoryComplex, MixedContainer.TypeNone, 'para', childobj_)
            self.content_.append(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'internal':
            childobj_ = docInternalS4Type.factory()
            childobj_.build(child_)
            obj_ = self.mixedclass_(MixedContainer.CategoryComplex, MixedContainer.TypeNone, 'internal', childobj_)
            self.content_.append(obj_)
        elif child_.nodeType == Node.TEXT_NODE:
            obj_ = self.mixedclass_(MixedContainer.CategoryText, MixedContainer.TypeNone, '', child_.nodeValue)
            self.content_.append(obj_)

class docInternalType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, para=None, sect1=None, mixedclass_=None, content_=None):
        if False:
            print('Hello World!')
        if mixedclass_ is None:
            self.mixedclass_ = MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        if content_ is None:
            self.content_ = []
        else:
            self.content_ = content_

    def factory(*args_, **kwargs_):
        if False:
            while True:
                i = 10
        if docInternalType.subclass:
            return docInternalType.subclass(*args_, **kwargs_)
        else:
            return docInternalType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_para(self):
        if False:
            return 10
        return self.para

    def set_para(self, para):
        if False:
            print('Hello World!')
        self.para = para

    def add_para(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.para.append(value)

    def insert_para(self, index, value):
        if False:
            return 10
        self.para[index] = value

    def get_sect1(self):
        if False:
            i = 10
            return i + 15
        return self.sect1

    def set_sect1(self, sect1):
        if False:
            return 10
        self.sect1 = sect1

    def add_sect1(self, value):
        if False:
            print('Hello World!')
        self.sect1.append(value)

    def insert_sect1(self, index, value):
        if False:
            print('Hello World!')
        self.sect1[index] = value

    def export(self, outfile, level, namespace_='', name_='docInternalType', namespacedef_=''):
        if False:
            i = 10
            return i + 15
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='docInternalType')
        outfile.write('>')
        self.exportChildren(outfile, level + 1, namespace_, name_)
        outfile.write('</%s%s>\n' % (namespace_, name_))

    def exportAttributes(self, outfile, level, namespace_='', name_='docInternalType'):
        if False:
            print('Hello World!')
        pass

    def exportChildren(self, outfile, level, namespace_='', name_='docInternalType'):
        if False:
            return 10
        for item_ in self.content_:
            item_.export(outfile, level, item_.name, namespace_)

    def hasContent_(self):
        if False:
            while True:
                i = 10
        if self.para is not None or self.sect1 is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='docInternalType'):
        if False:
            return 10
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            print('Hello World!')
        pass

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            for i in range(10):
                print('nop')
        showIndent(outfile, level)
        outfile.write('content_ = [\n')
        for item_ in self.content_:
            item_.exportLiteral(outfile, level, name_)
        showIndent(outfile, level)
        outfile.write('],\n')
        showIndent(outfile, level)
        outfile.write('content_ = [\n')
        for item_ in self.content_:
            item_.exportLiteral(outfile, level, name_)
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
        pass

    def buildChildren(self, child_, nodeName_):
        if False:
            print('Hello World!')
        if child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'para':
            childobj_ = docParaType.factory()
            childobj_.build(child_)
            obj_ = self.mixedclass_(MixedContainer.CategoryComplex, MixedContainer.TypeNone, 'para', childobj_)
            self.content_.append(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'sect1':
            childobj_ = docSect1Type.factory()
            childobj_.build(child_)
            obj_ = self.mixedclass_(MixedContainer.CategoryComplex, MixedContainer.TypeNone, 'sect1', childobj_)
            self.content_.append(obj_)
        elif child_.nodeType == Node.TEXT_NODE:
            obj_ = self.mixedclass_(MixedContainer.CategoryText, MixedContainer.TypeNone, '', child_.nodeValue)
            self.content_.append(obj_)

class docInternalS1Type(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, para=None, sect2=None, mixedclass_=None, content_=None):
        if False:
            for i in range(10):
                print('nop')
        if mixedclass_ is None:
            self.mixedclass_ = MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        if content_ is None:
            self.content_ = []
        else:
            self.content_ = content_

    def factory(*args_, **kwargs_):
        if False:
            return 10
        if docInternalS1Type.subclass:
            return docInternalS1Type.subclass(*args_, **kwargs_)
        else:
            return docInternalS1Type(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_para(self):
        if False:
            for i in range(10):
                print('nop')
        return self.para

    def set_para(self, para):
        if False:
            print('Hello World!')
        self.para = para

    def add_para(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.para.append(value)

    def insert_para(self, index, value):
        if False:
            i = 10
            return i + 15
        self.para[index] = value

    def get_sect2(self):
        if False:
            for i in range(10):
                print('nop')
        return self.sect2

    def set_sect2(self, sect2):
        if False:
            for i in range(10):
                print('nop')
        self.sect2 = sect2

    def add_sect2(self, value):
        if False:
            i = 10
            return i + 15
        self.sect2.append(value)

    def insert_sect2(self, index, value):
        if False:
            for i in range(10):
                print('nop')
        self.sect2[index] = value

    def export(self, outfile, level, namespace_='', name_='docInternalS1Type', namespacedef_=''):
        if False:
            while True:
                i = 10
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='docInternalS1Type')
        outfile.write('>')
        self.exportChildren(outfile, level + 1, namespace_, name_)
        outfile.write('</%s%s>\n' % (namespace_, name_))

    def exportAttributes(self, outfile, level, namespace_='', name_='docInternalS1Type'):
        if False:
            for i in range(10):
                print('nop')
        pass

    def exportChildren(self, outfile, level, namespace_='', name_='docInternalS1Type'):
        if False:
            for i in range(10):
                print('nop')
        for item_ in self.content_:
            item_.export(outfile, level, item_.name, namespace_)

    def hasContent_(self):
        if False:
            i = 10
            return i + 15
        if self.para is not None or self.sect2 is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='docInternalS1Type'):
        if False:
            print('Hello World!')
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            return 10
        pass

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            return 10
        showIndent(outfile, level)
        outfile.write('content_ = [\n')
        for item_ in self.content_:
            item_.exportLiteral(outfile, level, name_)
        showIndent(outfile, level)
        outfile.write('],\n')
        showIndent(outfile, level)
        outfile.write('content_ = [\n')
        for item_ in self.content_:
            item_.exportLiteral(outfile, level, name_)
        showIndent(outfile, level)
        outfile.write('],\n')

    def build(self, node_):
        if False:
            i = 10
            return i + 15
        attrs = node_.attributes
        self.buildAttributes(attrs)
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            i = 10
            return i + 15
        pass

    def buildChildren(self, child_, nodeName_):
        if False:
            while True:
                i = 10
        if child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'para':
            childobj_ = docParaType.factory()
            childobj_.build(child_)
            obj_ = self.mixedclass_(MixedContainer.CategoryComplex, MixedContainer.TypeNone, 'para', childobj_)
            self.content_.append(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'sect2':
            childobj_ = docSect2Type.factory()
            childobj_.build(child_)
            obj_ = self.mixedclass_(MixedContainer.CategoryComplex, MixedContainer.TypeNone, 'sect2', childobj_)
            self.content_.append(obj_)
        elif child_.nodeType == Node.TEXT_NODE:
            obj_ = self.mixedclass_(MixedContainer.CategoryText, MixedContainer.TypeNone, '', child_.nodeValue)
            self.content_.append(obj_)

class docInternalS2Type(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, para=None, sect3=None, mixedclass_=None, content_=None):
        if False:
            i = 10
            return i + 15
        if mixedclass_ is None:
            self.mixedclass_ = MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        if content_ is None:
            self.content_ = []
        else:
            self.content_ = content_

    def factory(*args_, **kwargs_):
        if False:
            while True:
                i = 10
        if docInternalS2Type.subclass:
            return docInternalS2Type.subclass(*args_, **kwargs_)
        else:
            return docInternalS2Type(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_para(self):
        if False:
            i = 10
            return i + 15
        return self.para

    def set_para(self, para):
        if False:
            i = 10
            return i + 15
        self.para = para

    def add_para(self, value):
        if False:
            return 10
        self.para.append(value)

    def insert_para(self, index, value):
        if False:
            while True:
                i = 10
        self.para[index] = value

    def get_sect3(self):
        if False:
            for i in range(10):
                print('nop')
        return self.sect3

    def set_sect3(self, sect3):
        if False:
            for i in range(10):
                print('nop')
        self.sect3 = sect3

    def add_sect3(self, value):
        if False:
            i = 10
            return i + 15
        self.sect3.append(value)

    def insert_sect3(self, index, value):
        if False:
            i = 10
            return i + 15
        self.sect3[index] = value

    def export(self, outfile, level, namespace_='', name_='docInternalS2Type', namespacedef_=''):
        if False:
            while True:
                i = 10
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='docInternalS2Type')
        outfile.write('>')
        self.exportChildren(outfile, level + 1, namespace_, name_)
        outfile.write('</%s%s>\n' % (namespace_, name_))

    def exportAttributes(self, outfile, level, namespace_='', name_='docInternalS2Type'):
        if False:
            print('Hello World!')
        pass

    def exportChildren(self, outfile, level, namespace_='', name_='docInternalS2Type'):
        if False:
            print('Hello World!')
        for item_ in self.content_:
            item_.export(outfile, level, item_.name, namespace_)

    def hasContent_(self):
        if False:
            while True:
                i = 10
        if self.para is not None or self.sect3 is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='docInternalS2Type'):
        if False:
            return 10
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            return 10
        pass

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            while True:
                i = 10
        showIndent(outfile, level)
        outfile.write('content_ = [\n')
        for item_ in self.content_:
            item_.exportLiteral(outfile, level, name_)
        showIndent(outfile, level)
        outfile.write('],\n')
        showIndent(outfile, level)
        outfile.write('content_ = [\n')
        for item_ in self.content_:
            item_.exportLiteral(outfile, level, name_)
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
            while True:
                i = 10
        pass

    def buildChildren(self, child_, nodeName_):
        if False:
            while True:
                i = 10
        if child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'para':
            childobj_ = docParaType.factory()
            childobj_.build(child_)
            obj_ = self.mixedclass_(MixedContainer.CategoryComplex, MixedContainer.TypeNone, 'para', childobj_)
            self.content_.append(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'sect3':
            childobj_ = docSect3Type.factory()
            childobj_.build(child_)
            obj_ = self.mixedclass_(MixedContainer.CategoryComplex, MixedContainer.TypeNone, 'sect3', childobj_)
            self.content_.append(obj_)
        elif child_.nodeType == Node.TEXT_NODE:
            obj_ = self.mixedclass_(MixedContainer.CategoryText, MixedContainer.TypeNone, '', child_.nodeValue)
            self.content_.append(obj_)

class docInternalS3Type(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, para=None, sect3=None, mixedclass_=None, content_=None):
        if False:
            i = 10
            return i + 15
        if mixedclass_ is None:
            self.mixedclass_ = MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        if content_ is None:
            self.content_ = []
        else:
            self.content_ = content_

    def factory(*args_, **kwargs_):
        if False:
            print('Hello World!')
        if docInternalS3Type.subclass:
            return docInternalS3Type.subclass(*args_, **kwargs_)
        else:
            return docInternalS3Type(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_para(self):
        if False:
            return 10
        return self.para

    def set_para(self, para):
        if False:
            for i in range(10):
                print('nop')
        self.para = para

    def add_para(self, value):
        if False:
            return 10
        self.para.append(value)

    def insert_para(self, index, value):
        if False:
            i = 10
            return i + 15
        self.para[index] = value

    def get_sect3(self):
        if False:
            return 10
        return self.sect3

    def set_sect3(self, sect3):
        if False:
            for i in range(10):
                print('nop')
        self.sect3 = sect3

    def add_sect3(self, value):
        if False:
            print('Hello World!')
        self.sect3.append(value)

    def insert_sect3(self, index, value):
        if False:
            return 10
        self.sect3[index] = value

    def export(self, outfile, level, namespace_='', name_='docInternalS3Type', namespacedef_=''):
        if False:
            i = 10
            return i + 15
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='docInternalS3Type')
        outfile.write('>')
        self.exportChildren(outfile, level + 1, namespace_, name_)
        outfile.write('</%s%s>\n' % (namespace_, name_))

    def exportAttributes(self, outfile, level, namespace_='', name_='docInternalS3Type'):
        if False:
            for i in range(10):
                print('nop')
        pass

    def exportChildren(self, outfile, level, namespace_='', name_='docInternalS3Type'):
        if False:
            i = 10
            return i + 15
        for item_ in self.content_:
            item_.export(outfile, level, item_.name, namespace_)

    def hasContent_(self):
        if False:
            while True:
                i = 10
        if self.para is not None or self.sect3 is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='docInternalS3Type'):
        if False:
            for i in range(10):
                print('nop')
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            print('Hello World!')
        pass

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            print('Hello World!')
        showIndent(outfile, level)
        outfile.write('content_ = [\n')
        for item_ in self.content_:
            item_.exportLiteral(outfile, level, name_)
        showIndent(outfile, level)
        outfile.write('],\n')
        showIndent(outfile, level)
        outfile.write('content_ = [\n')
        for item_ in self.content_:
            item_.exportLiteral(outfile, level, name_)
        showIndent(outfile, level)
        outfile.write('],\n')

    def build(self, node_):
        if False:
            i = 10
            return i + 15
        attrs = node_.attributes
        self.buildAttributes(attrs)
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            i = 10
            return i + 15
        pass

    def buildChildren(self, child_, nodeName_):
        if False:
            while True:
                i = 10
        if child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'para':
            childobj_ = docParaType.factory()
            childobj_.build(child_)
            obj_ = self.mixedclass_(MixedContainer.CategoryComplex, MixedContainer.TypeNone, 'para', childobj_)
            self.content_.append(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'sect3':
            childobj_ = docSect4Type.factory()
            childobj_.build(child_)
            obj_ = self.mixedclass_(MixedContainer.CategoryComplex, MixedContainer.TypeNone, 'sect3', childobj_)
            self.content_.append(obj_)
        elif child_.nodeType == Node.TEXT_NODE:
            obj_ = self.mixedclass_(MixedContainer.CategoryText, MixedContainer.TypeNone, '', child_.nodeValue)
            self.content_.append(obj_)

class docInternalS4Type(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, para=None, mixedclass_=None, content_=None):
        if False:
            i = 10
            return i + 15
        if mixedclass_ is None:
            self.mixedclass_ = MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        if content_ is None:
            self.content_ = []
        else:
            self.content_ = content_

    def factory(*args_, **kwargs_):
        if False:
            for i in range(10):
                print('nop')
        if docInternalS4Type.subclass:
            return docInternalS4Type.subclass(*args_, **kwargs_)
        else:
            return docInternalS4Type(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_para(self):
        if False:
            while True:
                i = 10
        return self.para

    def set_para(self, para):
        if False:
            print('Hello World!')
        self.para = para

    def add_para(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.para.append(value)

    def insert_para(self, index, value):
        if False:
            return 10
        self.para[index] = value

    def export(self, outfile, level, namespace_='', name_='docInternalS4Type', namespacedef_=''):
        if False:
            return 10
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='docInternalS4Type')
        outfile.write('>')
        self.exportChildren(outfile, level + 1, namespace_, name_)
        outfile.write('</%s%s>\n' % (namespace_, name_))

    def exportAttributes(self, outfile, level, namespace_='', name_='docInternalS4Type'):
        if False:
            print('Hello World!')
        pass

    def exportChildren(self, outfile, level, namespace_='', name_='docInternalS4Type'):
        if False:
            i = 10
            return i + 15
        for item_ in self.content_:
            item_.export(outfile, level, item_.name, namespace_)

    def hasContent_(self):
        if False:
            return 10
        if self.para is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='docInternalS4Type'):
        if False:
            return 10
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            while True:
                i = 10
        pass

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            while True:
                i = 10
        showIndent(outfile, level)
        outfile.write('content_ = [\n')
        for item_ in self.content_:
            item_.exportLiteral(outfile, level, name_)
        showIndent(outfile, level)
        outfile.write('],\n')

    def build(self, node_):
        if False:
            i = 10
            return i + 15
        attrs = node_.attributes
        self.buildAttributes(attrs)
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            return 10
        pass

    def buildChildren(self, child_, nodeName_):
        if False:
            i = 10
            return i + 15
        if child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'para':
            childobj_ = docParaType.factory()
            childobj_.build(child_)
            obj_ = self.mixedclass_(MixedContainer.CategoryComplex, MixedContainer.TypeNone, 'para', childobj_)
            self.content_.append(obj_)
        elif child_.nodeType == Node.TEXT_NODE:
            obj_ = self.mixedclass_(MixedContainer.CategoryText, MixedContainer.TypeNone, '', child_.nodeValue)
            self.content_.append(obj_)

class docTitleType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, valueOf_='', mixedclass_=None, content_=None):
        if False:
            print('Hello World!')
        if mixedclass_ is None:
            self.mixedclass_ = MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        if content_ is None:
            self.content_ = []
        else:
            self.content_ = content_

    def factory(*args_, **kwargs_):
        if False:
            i = 10
            return i + 15
        if docTitleType.subclass:
            return docTitleType.subclass(*args_, **kwargs_)
        else:
            return docTitleType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def getValueOf_(self):
        if False:
            for i in range(10):
                print('nop')
        return self.valueOf_

    def setValueOf_(self, valueOf_):
        if False:
            while True:
                i = 10
        self.valueOf_ = valueOf_

    def export(self, outfile, level, namespace_='', name_='docTitleType', namespacedef_=''):
        if False:
            i = 10
            return i + 15
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='docTitleType')
        outfile.write('>')
        self.exportChildren(outfile, level + 1, namespace_, name_)
        outfile.write('</%s%s>\n' % (namespace_, name_))

    def exportAttributes(self, outfile, level, namespace_='', name_='docTitleType'):
        if False:
            for i in range(10):
                print('nop')
        pass

    def exportChildren(self, outfile, level, namespace_='', name_='docTitleType'):
        if False:
            while True:
                i = 10
        if self.valueOf_.find('![CDATA') > -1:
            value = quote_xml('%s' % self.valueOf_)
            value = value.replace('![CDATA', '<![CDATA')
            value = value.replace(']]', ']]>')
            outfile.write(value)
        else:
            outfile.write(quote_xml('%s' % self.valueOf_))

    def hasContent_(self):
        if False:
            while True:
                i = 10
        if self.valueOf_ is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='docTitleType'):
        if False:
            i = 10
            return i + 15
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            print('Hello World!')
        pass

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            i = 10
            return i + 15
        showIndent(outfile, level)
        outfile.write('valueOf_ = "%s",\n' % (self.valueOf_,))

    def build(self, node_):
        if False:
            print('Hello World!')
        attrs = node_.attributes
        self.buildAttributes(attrs)
        self.valueOf_ = ''
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            for i in range(10):
                print('nop')
        pass

    def buildChildren(self, child_, nodeName_):
        if False:
            return 10
        if child_.nodeType == Node.TEXT_NODE:
            obj_ = self.mixedclass_(MixedContainer.CategoryText, MixedContainer.TypeNone, '', child_.nodeValue)
            self.content_.append(obj_)
        if child_.nodeType == Node.TEXT_NODE:
            self.valueOf_ += child_.nodeValue
        elif child_.nodeType == Node.CDATA_SECTION_NODE:
            self.valueOf_ += '![CDATA[' + child_.nodeValue + ']]'

class docParaType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, valueOf_='', mixedclass_=None, content_=None):
        if False:
            i = 10
            return i + 15
        if mixedclass_ is None:
            self.mixedclass_ = MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        if content_ is None:
            self.content_ = []
        else:
            self.content_ = content_

    def factory(*args_, **kwargs_):
        if False:
            for i in range(10):
                print('nop')
        if docParaType.subclass:
            return docParaType.subclass(*args_, **kwargs_)
        else:
            return docParaType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def getValueOf_(self):
        if False:
            while True:
                i = 10
        return self.valueOf_

    def setValueOf_(self, valueOf_):
        if False:
            i = 10
            return i + 15
        self.valueOf_ = valueOf_

    def export(self, outfile, level, namespace_='', name_='docParaType', namespacedef_=''):
        if False:
            for i in range(10):
                print('nop')
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='docParaType')
        outfile.write('>')
        self.exportChildren(outfile, level + 1, namespace_, name_)
        outfile.write('</%s%s>\n' % (namespace_, name_))

    def exportAttributes(self, outfile, level, namespace_='', name_='docParaType'):
        if False:
            i = 10
            return i + 15
        pass

    def exportChildren(self, outfile, level, namespace_='', name_='docParaType'):
        if False:
            print('Hello World!')
        if self.valueOf_.find('![CDATA') > -1:
            value = quote_xml('%s' % self.valueOf_)
            value = value.replace('![CDATA', '<![CDATA')
            value = value.replace(']]', ']]>')
            outfile.write(value)
        else:
            outfile.write(quote_xml('%s' % self.valueOf_))

    def hasContent_(self):
        if False:
            for i in range(10):
                print('nop')
        if self.valueOf_ is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='docParaType'):
        if False:
            print('Hello World!')
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            for i in range(10):
                print('nop')
        pass

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            while True:
                i = 10
        showIndent(outfile, level)
        outfile.write('valueOf_ = "%s",\n' % (self.valueOf_,))

    def build(self, node_):
        if False:
            return 10
        attrs = node_.attributes
        self.buildAttributes(attrs)
        self.valueOf_ = ''
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            print('Hello World!')
        pass

    def buildChildren(self, child_, nodeName_):
        if False:
            i = 10
            return i + 15
        if child_.nodeType == Node.TEXT_NODE:
            obj_ = self.mixedclass_(MixedContainer.CategoryText, MixedContainer.TypeNone, '', child_.nodeValue)
            self.content_.append(obj_)
        if child_.nodeType == Node.TEXT_NODE:
            self.valueOf_ += child_.nodeValue
        elif child_.nodeType == Node.CDATA_SECTION_NODE:
            self.valueOf_ += '![CDATA[' + child_.nodeValue + ']]'

class docMarkupType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, valueOf_='', mixedclass_=None, content_=None):
        if False:
            return 10
        if mixedclass_ is None:
            self.mixedclass_ = MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        if content_ is None:
            self.content_ = []
        else:
            self.content_ = content_

    def factory(*args_, **kwargs_):
        if False:
            return 10
        if docMarkupType.subclass:
            return docMarkupType.subclass(*args_, **kwargs_)
        else:
            return docMarkupType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def getValueOf_(self):
        if False:
            for i in range(10):
                print('nop')
        return self.valueOf_

    def setValueOf_(self, valueOf_):
        if False:
            i = 10
            return i + 15
        self.valueOf_ = valueOf_

    def export(self, outfile, level, namespace_='', name_='docMarkupType', namespacedef_=''):
        if False:
            i = 10
            return i + 15
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='docMarkupType')
        outfile.write('>')
        self.exportChildren(outfile, level + 1, namespace_, name_)
        outfile.write('</%s%s>\n' % (namespace_, name_))

    def exportAttributes(self, outfile, level, namespace_='', name_='docMarkupType'):
        if False:
            i = 10
            return i + 15
        pass

    def exportChildren(self, outfile, level, namespace_='', name_='docMarkupType'):
        if False:
            return 10
        if self.valueOf_.find('![CDATA') > -1:
            value = quote_xml('%s' % self.valueOf_)
            value = value.replace('![CDATA', '<![CDATA')
            value = value.replace(']]', ']]>')
            outfile.write(value)
        else:
            outfile.write(quote_xml('%s' % self.valueOf_))

    def hasContent_(self):
        if False:
            print('Hello World!')
        if self.valueOf_ is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='docMarkupType'):
        if False:
            while True:
                i = 10
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            for i in range(10):
                print('nop')
        pass

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            for i in range(10):
                print('nop')
        showIndent(outfile, level)
        outfile.write('valueOf_ = "%s",\n' % (self.valueOf_,))

    def build(self, node_):
        if False:
            return 10
        attrs = node_.attributes
        self.buildAttributes(attrs)
        self.valueOf_ = ''
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            i = 10
            return i + 15
        pass

    def buildChildren(self, child_, nodeName_):
        if False:
            return 10
        if child_.nodeType == Node.TEXT_NODE:
            obj_ = self.mixedclass_(MixedContainer.CategoryText, MixedContainer.TypeNone, '', child_.nodeValue)
            self.content_.append(obj_)
        if child_.nodeType == Node.TEXT_NODE:
            self.valueOf_ += child_.nodeValue
        elif child_.nodeType == Node.CDATA_SECTION_NODE:
            self.valueOf_ += '![CDATA[' + child_.nodeValue + ']]'

class docURLLink(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, url=None, valueOf_='', mixedclass_=None, content_=None):
        if False:
            print('Hello World!')
        self.url = url
        if mixedclass_ is None:
            self.mixedclass_ = MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        if content_ is None:
            self.content_ = []
        else:
            self.content_ = content_

    def factory(*args_, **kwargs_):
        if False:
            print('Hello World!')
        if docURLLink.subclass:
            return docURLLink.subclass(*args_, **kwargs_)
        else:
            return docURLLink(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_url(self):
        if False:
            i = 10
            return i + 15
        return self.url

    def set_url(self, url):
        if False:
            for i in range(10):
                print('nop')
        self.url = url

    def getValueOf_(self):
        if False:
            i = 10
            return i + 15
        return self.valueOf_

    def setValueOf_(self, valueOf_):
        if False:
            return 10
        self.valueOf_ = valueOf_

    def export(self, outfile, level, namespace_='', name_='docURLLink', namespacedef_=''):
        if False:
            i = 10
            return i + 15
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='docURLLink')
        outfile.write('>')
        self.exportChildren(outfile, level + 1, namespace_, name_)
        outfile.write('</%s%s>\n' % (namespace_, name_))

    def exportAttributes(self, outfile, level, namespace_='', name_='docURLLink'):
        if False:
            for i in range(10):
                print('nop')
        if self.url is not None:
            outfile.write(' url=%s' % (self.format_string(quote_attrib(self.url).encode(ExternalEncoding), input_name='url'),))

    def exportChildren(self, outfile, level, namespace_='', name_='docURLLink'):
        if False:
            i = 10
            return i + 15
        if self.valueOf_.find('![CDATA') > -1:
            value = quote_xml('%s' % self.valueOf_)
            value = value.replace('![CDATA', '<![CDATA')
            value = value.replace(']]', ']]>')
            outfile.write(value)
        else:
            outfile.write(quote_xml('%s' % self.valueOf_))

    def hasContent_(self):
        if False:
            return 10
        if self.valueOf_ is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='docURLLink'):
        if False:
            print('Hello World!')
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            return 10
        if self.url is not None:
            showIndent(outfile, level)
            outfile.write('url = %s,\n' % (self.url,))

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            i = 10
            return i + 15
        showIndent(outfile, level)
        outfile.write('valueOf_ = "%s",\n' % (self.valueOf_,))

    def build(self, node_):
        if False:
            for i in range(10):
                print('nop')
        attrs = node_.attributes
        self.buildAttributes(attrs)
        self.valueOf_ = ''
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            i = 10
            return i + 15
        if attrs.get('url'):
            self.url = attrs.get('url').value

    def buildChildren(self, child_, nodeName_):
        if False:
            return 10
        if child_.nodeType == Node.TEXT_NODE:
            obj_ = self.mixedclass_(MixedContainer.CategoryText, MixedContainer.TypeNone, '', child_.nodeValue)
            self.content_.append(obj_)
        if child_.nodeType == Node.TEXT_NODE:
            self.valueOf_ += child_.nodeValue
        elif child_.nodeType == Node.CDATA_SECTION_NODE:
            self.valueOf_ += '![CDATA[' + child_.nodeValue + ']]'

class docAnchorType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, id=None, valueOf_='', mixedclass_=None, content_=None):
        if False:
            for i in range(10):
                print('nop')
        self.id = id
        if mixedclass_ is None:
            self.mixedclass_ = MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        if content_ is None:
            self.content_ = []
        else:
            self.content_ = content_

    def factory(*args_, **kwargs_):
        if False:
            while True:
                i = 10
        if docAnchorType.subclass:
            return docAnchorType.subclass(*args_, **kwargs_)
        else:
            return docAnchorType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_id(self):
        if False:
            while True:
                i = 10
        return self.id

    def set_id(self, id):
        if False:
            while True:
                i = 10
        self.id = id

    def getValueOf_(self):
        if False:
            print('Hello World!')
        return self.valueOf_

    def setValueOf_(self, valueOf_):
        if False:
            print('Hello World!')
        self.valueOf_ = valueOf_

    def export(self, outfile, level, namespace_='', name_='docAnchorType', namespacedef_=''):
        if False:
            print('Hello World!')
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='docAnchorType')
        outfile.write('>')
        self.exportChildren(outfile, level + 1, namespace_, name_)
        outfile.write('</%s%s>\n' % (namespace_, name_))

    def exportAttributes(self, outfile, level, namespace_='', name_='docAnchorType'):
        if False:
            print('Hello World!')
        if self.id is not None:
            outfile.write(' id=%s' % (self.format_string(quote_attrib(self.id).encode(ExternalEncoding), input_name='id'),))

    def exportChildren(self, outfile, level, namespace_='', name_='docAnchorType'):
        if False:
            print('Hello World!')
        if self.valueOf_.find('![CDATA') > -1:
            value = quote_xml('%s' % self.valueOf_)
            value = value.replace('![CDATA', '<![CDATA')
            value = value.replace(']]', ']]>')
            outfile.write(value)
        else:
            outfile.write(quote_xml('%s' % self.valueOf_))

    def hasContent_(self):
        if False:
            while True:
                i = 10
        if self.valueOf_ is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='docAnchorType'):
        if False:
            while True:
                i = 10
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            i = 10
            return i + 15
        if self.id is not None:
            showIndent(outfile, level)
            outfile.write('id = %s,\n' % (self.id,))

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            print('Hello World!')
        showIndent(outfile, level)
        outfile.write('valueOf_ = "%s",\n' % (self.valueOf_,))

    def build(self, node_):
        if False:
            print('Hello World!')
        attrs = node_.attributes
        self.buildAttributes(attrs)
        self.valueOf_ = ''
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            print('Hello World!')
        if attrs.get('id'):
            self.id = attrs.get('id').value

    def buildChildren(self, child_, nodeName_):
        if False:
            for i in range(10):
                print('nop')
        if child_.nodeType == Node.TEXT_NODE:
            obj_ = self.mixedclass_(MixedContainer.CategoryText, MixedContainer.TypeNone, '', child_.nodeValue)
            self.content_.append(obj_)
        if child_.nodeType == Node.TEXT_NODE:
            self.valueOf_ += child_.nodeValue
        elif child_.nodeType == Node.CDATA_SECTION_NODE:
            self.valueOf_ += '![CDATA[' + child_.nodeValue + ']]'

class docFormulaType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, id=None, valueOf_='', mixedclass_=None, content_=None):
        if False:
            i = 10
            return i + 15
        self.id = id
        if mixedclass_ is None:
            self.mixedclass_ = MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        if content_ is None:
            self.content_ = []
        else:
            self.content_ = content_

    def factory(*args_, **kwargs_):
        if False:
            return 10
        if docFormulaType.subclass:
            return docFormulaType.subclass(*args_, **kwargs_)
        else:
            return docFormulaType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_id(self):
        if False:
            for i in range(10):
                print('nop')
        return self.id

    def set_id(self, id):
        if False:
            i = 10
            return i + 15
        self.id = id

    def getValueOf_(self):
        if False:
            print('Hello World!')
        return self.valueOf_

    def setValueOf_(self, valueOf_):
        if False:
            for i in range(10):
                print('nop')
        self.valueOf_ = valueOf_

    def export(self, outfile, level, namespace_='', name_='docFormulaType', namespacedef_=''):
        if False:
            while True:
                i = 10
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='docFormulaType')
        outfile.write('>')
        self.exportChildren(outfile, level + 1, namespace_, name_)
        outfile.write('</%s%s>\n' % (namespace_, name_))

    def exportAttributes(self, outfile, level, namespace_='', name_='docFormulaType'):
        if False:
            while True:
                i = 10
        if self.id is not None:
            outfile.write(' id=%s' % (self.format_string(quote_attrib(self.id).encode(ExternalEncoding), input_name='id'),))

    def exportChildren(self, outfile, level, namespace_='', name_='docFormulaType'):
        if False:
            print('Hello World!')
        if self.valueOf_.find('![CDATA') > -1:
            value = quote_xml('%s' % self.valueOf_)
            value = value.replace('![CDATA', '<![CDATA')
            value = value.replace(']]', ']]>')
            outfile.write(value)
        else:
            outfile.write(quote_xml('%s' % self.valueOf_))

    def hasContent_(self):
        if False:
            for i in range(10):
                print('nop')
        if self.valueOf_ is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='docFormulaType'):
        if False:
            print('Hello World!')
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            print('Hello World!')
        if self.id is not None:
            showIndent(outfile, level)
            outfile.write('id = %s,\n' % (self.id,))

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            return 10
        showIndent(outfile, level)
        outfile.write('valueOf_ = "%s",\n' % (self.valueOf_,))

    def build(self, node_):
        if False:
            while True:
                i = 10
        attrs = node_.attributes
        self.buildAttributes(attrs)
        self.valueOf_ = ''
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            print('Hello World!')
        if attrs.get('id'):
            self.id = attrs.get('id').value

    def buildChildren(self, child_, nodeName_):
        if False:
            i = 10
            return i + 15
        if child_.nodeType == Node.TEXT_NODE:
            obj_ = self.mixedclass_(MixedContainer.CategoryText, MixedContainer.TypeNone, '', child_.nodeValue)
            self.content_.append(obj_)
        if child_.nodeType == Node.TEXT_NODE:
            self.valueOf_ += child_.nodeValue
        elif child_.nodeType == Node.CDATA_SECTION_NODE:
            self.valueOf_ += '![CDATA[' + child_.nodeValue + ']]'

class docIndexEntryType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, primaryie=None, secondaryie=None):
        if False:
            i = 10
            return i + 15
        self.primaryie = primaryie
        self.secondaryie = secondaryie

    def factory(*args_, **kwargs_):
        if False:
            for i in range(10):
                print('nop')
        if docIndexEntryType.subclass:
            return docIndexEntryType.subclass(*args_, **kwargs_)
        else:
            return docIndexEntryType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_primaryie(self):
        if False:
            return 10
        return self.primaryie

    def set_primaryie(self, primaryie):
        if False:
            return 10
        self.primaryie = primaryie

    def get_secondaryie(self):
        if False:
            i = 10
            return i + 15
        return self.secondaryie

    def set_secondaryie(self, secondaryie):
        if False:
            return 10
        self.secondaryie = secondaryie

    def export(self, outfile, level, namespace_='', name_='docIndexEntryType', namespacedef_=''):
        if False:
            while True:
                i = 10
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='docIndexEntryType')
        if self.hasContent_():
            outfile.write('>\n')
            self.exportChildren(outfile, level + 1, namespace_, name_)
            showIndent(outfile, level)
            outfile.write('</%s%s>\n' % (namespace_, name_))
        else:
            outfile.write(' />\n')

    def exportAttributes(self, outfile, level, namespace_='', name_='docIndexEntryType'):
        if False:
            return 10
        pass

    def exportChildren(self, outfile, level, namespace_='', name_='docIndexEntryType'):
        if False:
            i = 10
            return i + 15
        if self.primaryie is not None:
            showIndent(outfile, level)
            outfile.write('<%sprimaryie>%s</%sprimaryie>\n' % (namespace_, self.format_string(quote_xml(self.primaryie).encode(ExternalEncoding), input_name='primaryie'), namespace_))
        if self.secondaryie is not None:
            showIndent(outfile, level)
            outfile.write('<%ssecondaryie>%s</%ssecondaryie>\n' % (namespace_, self.format_string(quote_xml(self.secondaryie).encode(ExternalEncoding), input_name='secondaryie'), namespace_))

    def hasContent_(self):
        if False:
            while True:
                i = 10
        if self.primaryie is not None or self.secondaryie is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='docIndexEntryType'):
        if False:
            i = 10
            return i + 15
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            return 10
        pass

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            for i in range(10):
                print('nop')
        showIndent(outfile, level)
        outfile.write('primaryie=%s,\n' % quote_python(self.primaryie).encode(ExternalEncoding))
        showIndent(outfile, level)
        outfile.write('secondaryie=%s,\n' % quote_python(self.secondaryie).encode(ExternalEncoding))

    def build(self, node_):
        if False:
            return 10
        attrs = node_.attributes
        self.buildAttributes(attrs)
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            return 10
        pass

    def buildChildren(self, child_, nodeName_):
        if False:
            i = 10
            return i + 15
        if child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'primaryie':
            primaryie_ = ''
            for text__content_ in child_.childNodes:
                primaryie_ += text__content_.nodeValue
            self.primaryie = primaryie_
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'secondaryie':
            secondaryie_ = ''
            for text__content_ in child_.childNodes:
                secondaryie_ += text__content_.nodeValue
            self.secondaryie = secondaryie_

class docListType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, listitem=None):
        if False:
            for i in range(10):
                print('nop')
        if listitem is None:
            self.listitem = []
        else:
            self.listitem = listitem

    def factory(*args_, **kwargs_):
        if False:
            for i in range(10):
                print('nop')
        if docListType.subclass:
            return docListType.subclass(*args_, **kwargs_)
        else:
            return docListType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_listitem(self):
        if False:
            while True:
                i = 10
        return self.listitem

    def set_listitem(self, listitem):
        if False:
            i = 10
            return i + 15
        self.listitem = listitem

    def add_listitem(self, value):
        if False:
            i = 10
            return i + 15
        self.listitem.append(value)

    def insert_listitem(self, index, value):
        if False:
            while True:
                i = 10
        self.listitem[index] = value

    def export(self, outfile, level, namespace_='', name_='docListType', namespacedef_=''):
        if False:
            for i in range(10):
                print('nop')
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='docListType')
        if self.hasContent_():
            outfile.write('>\n')
            self.exportChildren(outfile, level + 1, namespace_, name_)
            showIndent(outfile, level)
            outfile.write('</%s%s>\n' % (namespace_, name_))
        else:
            outfile.write(' />\n')

    def exportAttributes(self, outfile, level, namespace_='', name_='docListType'):
        if False:
            i = 10
            return i + 15
        pass

    def exportChildren(self, outfile, level, namespace_='', name_='docListType'):
        if False:
            while True:
                i = 10
        for listitem_ in self.listitem:
            listitem_.export(outfile, level, namespace_, name_='listitem')

    def hasContent_(self):
        if False:
            i = 10
            return i + 15
        if self.listitem is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='docListType'):
        if False:
            for i in range(10):
                print('nop')
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            print('Hello World!')
        pass

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            return 10
        showIndent(outfile, level)
        outfile.write('listitem=[\n')
        level += 1
        for listitem in self.listitem:
            showIndent(outfile, level)
            outfile.write('model_.listitem(\n')
            listitem.exportLiteral(outfile, level, name_='listitem')
            showIndent(outfile, level)
            outfile.write('),\n')
        level -= 1
        showIndent(outfile, level)
        outfile.write('],\n')

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
            return 10
        pass

    def buildChildren(self, child_, nodeName_):
        if False:
            for i in range(10):
                print('nop')
        if child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'listitem':
            obj_ = docListItemType.factory()
            obj_.build(child_)
            self.listitem.append(obj_)

class docListItemType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, para=None):
        if False:
            return 10
        if para is None:
            self.para = []
        else:
            self.para = para

    def factory(*args_, **kwargs_):
        if False:
            print('Hello World!')
        if docListItemType.subclass:
            return docListItemType.subclass(*args_, **kwargs_)
        else:
            return docListItemType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_para(self):
        if False:
            print('Hello World!')
        return self.para

    def set_para(self, para):
        if False:
            i = 10
            return i + 15
        self.para = para

    def add_para(self, value):
        if False:
            i = 10
            return i + 15
        self.para.append(value)

    def insert_para(self, index, value):
        if False:
            for i in range(10):
                print('nop')
        self.para[index] = value

    def export(self, outfile, level, namespace_='', name_='docListItemType', namespacedef_=''):
        if False:
            i = 10
            return i + 15
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='docListItemType')
        if self.hasContent_():
            outfile.write('>\n')
            self.exportChildren(outfile, level + 1, namespace_, name_)
            showIndent(outfile, level)
            outfile.write('</%s%s>\n' % (namespace_, name_))
        else:
            outfile.write(' />\n')

    def exportAttributes(self, outfile, level, namespace_='', name_='docListItemType'):
        if False:
            return 10
        pass

    def exportChildren(self, outfile, level, namespace_='', name_='docListItemType'):
        if False:
            for i in range(10):
                print('nop')
        for para_ in self.para:
            para_.export(outfile, level, namespace_, name_='para')

    def hasContent_(self):
        if False:
            print('Hello World!')
        if self.para is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='docListItemType'):
        if False:
            return 10
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            return 10
        pass

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            print('Hello World!')
        showIndent(outfile, level)
        outfile.write('para=[\n')
        level += 1
        for para in self.para:
            showIndent(outfile, level)
            outfile.write('model_.para(\n')
            para.exportLiteral(outfile, level, name_='para')
            showIndent(outfile, level)
            outfile.write('),\n')
        level -= 1
        showIndent(outfile, level)
        outfile.write('],\n')

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
            i = 10
            return i + 15
        pass

    def buildChildren(self, child_, nodeName_):
        if False:
            i = 10
            return i + 15
        if child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'para':
            obj_ = docParaType.factory()
            obj_.build(child_)
            self.para.append(obj_)

class docSimpleSectType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, kind=None, title=None, para=None):
        if False:
            while True:
                i = 10
        self.kind = kind
        self.title = title
        if para is None:
            self.para = []
        else:
            self.para = para

    def factory(*args_, **kwargs_):
        if False:
            return 10
        if docSimpleSectType.subclass:
            return docSimpleSectType.subclass(*args_, **kwargs_)
        else:
            return docSimpleSectType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_title(self):
        if False:
            print('Hello World!')
        return self.title

    def set_title(self, title):
        if False:
            return 10
        self.title = title

    def get_para(self):
        if False:
            while True:
                i = 10
        return self.para

    def set_para(self, para):
        if False:
            print('Hello World!')
        self.para = para

    def add_para(self, value):
        if False:
            while True:
                i = 10
        self.para.append(value)

    def insert_para(self, index, value):
        if False:
            for i in range(10):
                print('nop')
        self.para[index] = value

    def get_kind(self):
        if False:
            i = 10
            return i + 15
        return self.kind

    def set_kind(self, kind):
        if False:
            while True:
                i = 10
        self.kind = kind

    def export(self, outfile, level, namespace_='', name_='docSimpleSectType', namespacedef_=''):
        if False:
            for i in range(10):
                print('nop')
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='docSimpleSectType')
        if self.hasContent_():
            outfile.write('>\n')
            self.exportChildren(outfile, level + 1, namespace_, name_)
            showIndent(outfile, level)
            outfile.write('</%s%s>\n' % (namespace_, name_))
        else:
            outfile.write(' />\n')

    def exportAttributes(self, outfile, level, namespace_='', name_='docSimpleSectType'):
        if False:
            while True:
                i = 10
        if self.kind is not None:
            outfile.write(' kind=%s' % (quote_attrib(self.kind),))

    def exportChildren(self, outfile, level, namespace_='', name_='docSimpleSectType'):
        if False:
            for i in range(10):
                print('nop')
        if self.title:
            self.title.export(outfile, level, namespace_, name_='title')
        for para_ in self.para:
            para_.export(outfile, level, namespace_, name_='para')

    def hasContent_(self):
        if False:
            print('Hello World!')
        if self.title is not None or self.para is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='docSimpleSectType'):
        if False:
            while True:
                i = 10
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            print('Hello World!')
        if self.kind is not None:
            showIndent(outfile, level)
            outfile.write('kind = "%s",\n' % (self.kind,))

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            for i in range(10):
                print('nop')
        if self.title:
            showIndent(outfile, level)
            outfile.write('title=model_.docTitleType(\n')
            self.title.exportLiteral(outfile, level, name_='title')
            showIndent(outfile, level)
            outfile.write('),\n')
        showIndent(outfile, level)
        outfile.write('para=[\n')
        level += 1
        for para in self.para:
            showIndent(outfile, level)
            outfile.write('model_.para(\n')
            para.exportLiteral(outfile, level, name_='para')
            showIndent(outfile, level)
            outfile.write('),\n')
        level -= 1
        showIndent(outfile, level)
        outfile.write('],\n')

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
            for i in range(10):
                print('nop')
        if attrs.get('kind'):
            self.kind = attrs.get('kind').value

    def buildChildren(self, child_, nodeName_):
        if False:
            while True:
                i = 10
        if child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'title':
            obj_ = docTitleType.factory()
            obj_.build(child_)
            self.set_title(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'para':
            obj_ = docParaType.factory()
            obj_.build(child_)
            self.para.append(obj_)

class docVarListEntryType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, term=None):
        if False:
            print('Hello World!')
        self.term = term

    def factory(*args_, **kwargs_):
        if False:
            for i in range(10):
                print('nop')
        if docVarListEntryType.subclass:
            return docVarListEntryType.subclass(*args_, **kwargs_)
        else:
            return docVarListEntryType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_term(self):
        if False:
            for i in range(10):
                print('nop')
        return self.term

    def set_term(self, term):
        if False:
            return 10
        self.term = term

    def export(self, outfile, level, namespace_='', name_='docVarListEntryType', namespacedef_=''):
        if False:
            print('Hello World!')
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='docVarListEntryType')
        if self.hasContent_():
            outfile.write('>\n')
            self.exportChildren(outfile, level + 1, namespace_, name_)
            showIndent(outfile, level)
            outfile.write('</%s%s>\n' % (namespace_, name_))
        else:
            outfile.write(' />\n')

    def exportAttributes(self, outfile, level, namespace_='', name_='docVarListEntryType'):
        if False:
            while True:
                i = 10
        pass

    def exportChildren(self, outfile, level, namespace_='', name_='docVarListEntryType'):
        if False:
            i = 10
            return i + 15
        if self.term:
            self.term.export(outfile, level, namespace_, name_='term')

    def hasContent_(self):
        if False:
            for i in range(10):
                print('nop')
        if self.term is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='docVarListEntryType'):
        if False:
            for i in range(10):
                print('nop')
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            return 10
        pass

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            i = 10
            return i + 15
        if self.term:
            showIndent(outfile, level)
            outfile.write('term=model_.docTitleType(\n')
            self.term.exportLiteral(outfile, level, name_='term')
            showIndent(outfile, level)
            outfile.write('),\n')

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
        pass

    def buildChildren(self, child_, nodeName_):
        if False:
            while True:
                i = 10
        if child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'term':
            obj_ = docTitleType.factory()
            obj_.build(child_)
            self.set_term(obj_)

class docVariableListType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, valueOf_=''):
        if False:
            print('Hello World!')
        self.valueOf_ = valueOf_

    def factory(*args_, **kwargs_):
        if False:
            i = 10
            return i + 15
        if docVariableListType.subclass:
            return docVariableListType.subclass(*args_, **kwargs_)
        else:
            return docVariableListType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def getValueOf_(self):
        if False:
            for i in range(10):
                print('nop')
        return self.valueOf_

    def setValueOf_(self, valueOf_):
        if False:
            i = 10
            return i + 15
        self.valueOf_ = valueOf_

    def export(self, outfile, level, namespace_='', name_='docVariableListType', namespacedef_=''):
        if False:
            for i in range(10):
                print('nop')
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='docVariableListType')
        if self.hasContent_():
            outfile.write('>\n')
            self.exportChildren(outfile, level + 1, namespace_, name_)
            showIndent(outfile, level)
            outfile.write('</%s%s>\n' % (namespace_, name_))
        else:
            outfile.write(' />\n')

    def exportAttributes(self, outfile, level, namespace_='', name_='docVariableListType'):
        if False:
            while True:
                i = 10
        pass

    def exportChildren(self, outfile, level, namespace_='', name_='docVariableListType'):
        if False:
            i = 10
            return i + 15
        if self.valueOf_.find('![CDATA') > -1:
            value = quote_xml('%s' % self.valueOf_)
            value = value.replace('![CDATA', '<![CDATA')
            value = value.replace(']]', ']]>')
            outfile.write(value)
        else:
            outfile.write(quote_xml('%s' % self.valueOf_))

    def hasContent_(self):
        if False:
            for i in range(10):
                print('nop')
        if self.valueOf_ is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='docVariableListType'):
        if False:
            while True:
                i = 10
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            for i in range(10):
                print('nop')
        pass

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            while True:
                i = 10
        showIndent(outfile, level)
        outfile.write('valueOf_ = "%s",\n' % (self.valueOf_,))

    def build(self, node_):
        if False:
            print('Hello World!')
        attrs = node_.attributes
        self.buildAttributes(attrs)
        self.valueOf_ = ''
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            for i in range(10):
                print('nop')
        pass

    def buildChildren(self, child_, nodeName_):
        if False:
            while True:
                i = 10
        if child_.nodeType == Node.TEXT_NODE:
            self.valueOf_ += child_.nodeValue
        elif child_.nodeType == Node.CDATA_SECTION_NODE:
            self.valueOf_ += '![CDATA[' + child_.nodeValue + ']]'

class docRefTextType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, refid=None, kindref=None, external=None, valueOf_='', mixedclass_=None, content_=None):
        if False:
            for i in range(10):
                print('nop')
        self.refid = refid
        self.kindref = kindref
        self.external = external
        if mixedclass_ is None:
            self.mixedclass_ = MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        if content_ is None:
            self.content_ = []
        else:
            self.content_ = content_

    def factory(*args_, **kwargs_):
        if False:
            while True:
                i = 10
        if docRefTextType.subclass:
            return docRefTextType.subclass(*args_, **kwargs_)
        else:
            return docRefTextType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_refid(self):
        if False:
            for i in range(10):
                print('nop')
        return self.refid

    def set_refid(self, refid):
        if False:
            while True:
                i = 10
        self.refid = refid

    def get_kindref(self):
        if False:
            return 10
        return self.kindref

    def set_kindref(self, kindref):
        if False:
            for i in range(10):
                print('nop')
        self.kindref = kindref

    def get_external(self):
        if False:
            while True:
                i = 10
        return self.external

    def set_external(self, external):
        if False:
            i = 10
            return i + 15
        self.external = external

    def getValueOf_(self):
        if False:
            while True:
                i = 10
        return self.valueOf_

    def setValueOf_(self, valueOf_):
        if False:
            i = 10
            return i + 15
        self.valueOf_ = valueOf_

    def export(self, outfile, level, namespace_='', name_='docRefTextType', namespacedef_=''):
        if False:
            for i in range(10):
                print('nop')
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='docRefTextType')
        outfile.write('>')
        self.exportChildren(outfile, level + 1, namespace_, name_)
        outfile.write('</%s%s>\n' % (namespace_, name_))

    def exportAttributes(self, outfile, level, namespace_='', name_='docRefTextType'):
        if False:
            i = 10
            return i + 15
        if self.refid is not None:
            outfile.write(' refid=%s' % (self.format_string(quote_attrib(self.refid).encode(ExternalEncoding), input_name='refid'),))
        if self.kindref is not None:
            outfile.write(' kindref=%s' % (quote_attrib(self.kindref),))
        if self.external is not None:
            outfile.write(' external=%s' % (self.format_string(quote_attrib(self.external).encode(ExternalEncoding), input_name='external'),))

    def exportChildren(self, outfile, level, namespace_='', name_='docRefTextType'):
        if False:
            while True:
                i = 10
        if self.valueOf_.find('![CDATA') > -1:
            value = quote_xml('%s' % self.valueOf_)
            value = value.replace('![CDATA', '<![CDATA')
            value = value.replace(']]', ']]>')
            outfile.write(value)
        else:
            outfile.write(quote_xml('%s' % self.valueOf_))

    def hasContent_(self):
        if False:
            return 10
        if self.valueOf_ is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='docRefTextType'):
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
        if self.refid is not None:
            showIndent(outfile, level)
            outfile.write('refid = %s,\n' % (self.refid,))
        if self.kindref is not None:
            showIndent(outfile, level)
            outfile.write('kindref = "%s",\n' % (self.kindref,))
        if self.external is not None:
            showIndent(outfile, level)
            outfile.write('external = %s,\n' % (self.external,))

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            i = 10
            return i + 15
        showIndent(outfile, level)
        outfile.write('valueOf_ = "%s",\n' % (self.valueOf_,))

    def build(self, node_):
        if False:
            for i in range(10):
                print('nop')
        attrs = node_.attributes
        self.buildAttributes(attrs)
        self.valueOf_ = ''
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            return 10
        if attrs.get('refid'):
            self.refid = attrs.get('refid').value
        if attrs.get('kindref'):
            self.kindref = attrs.get('kindref').value
        if attrs.get('external'):
            self.external = attrs.get('external').value

    def buildChildren(self, child_, nodeName_):
        if False:
            i = 10
            return i + 15
        if child_.nodeType == Node.TEXT_NODE:
            obj_ = self.mixedclass_(MixedContainer.CategoryText, MixedContainer.TypeNone, '', child_.nodeValue)
            self.content_.append(obj_)
        if child_.nodeType == Node.TEXT_NODE:
            self.valueOf_ += child_.nodeValue
        elif child_.nodeType == Node.CDATA_SECTION_NODE:
            self.valueOf_ += '![CDATA[' + child_.nodeValue + ']]'

class docTableType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, rows=None, cols=None, row=None, caption=None):
        if False:
            return 10
        self.rows = rows
        self.cols = cols
        if row is None:
            self.row = []
        else:
            self.row = row
        self.caption = caption

    def factory(*args_, **kwargs_):
        if False:
            for i in range(10):
                print('nop')
        if docTableType.subclass:
            return docTableType.subclass(*args_, **kwargs_)
        else:
            return docTableType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_row(self):
        if False:
            for i in range(10):
                print('nop')
        return self.row

    def set_row(self, row):
        if False:
            i = 10
            return i + 15
        self.row = row

    def add_row(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.row.append(value)

    def insert_row(self, index, value):
        if False:
            for i in range(10):
                print('nop')
        self.row[index] = value

    def get_caption(self):
        if False:
            for i in range(10):
                print('nop')
        return self.caption

    def set_caption(self, caption):
        if False:
            for i in range(10):
                print('nop')
        self.caption = caption

    def get_rows(self):
        if False:
            for i in range(10):
                print('nop')
        return self.rows

    def set_rows(self, rows):
        if False:
            for i in range(10):
                print('nop')
        self.rows = rows

    def get_cols(self):
        if False:
            for i in range(10):
                print('nop')
        return self.cols

    def set_cols(self, cols):
        if False:
            for i in range(10):
                print('nop')
        self.cols = cols

    def export(self, outfile, level, namespace_='', name_='docTableType', namespacedef_=''):
        if False:
            print('Hello World!')
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='docTableType')
        if self.hasContent_():
            outfile.write('>\n')
            self.exportChildren(outfile, level + 1, namespace_, name_)
            showIndent(outfile, level)
            outfile.write('</%s%s>\n' % (namespace_, name_))
        else:
            outfile.write(' />\n')

    def exportAttributes(self, outfile, level, namespace_='', name_='docTableType'):
        if False:
            i = 10
            return i + 15
        if self.rows is not None:
            outfile.write(' rows="%s"' % self.format_integer(self.rows, input_name='rows'))
        if self.cols is not None:
            outfile.write(' cols="%s"' % self.format_integer(self.cols, input_name='cols'))

    def exportChildren(self, outfile, level, namespace_='', name_='docTableType'):
        if False:
            i = 10
            return i + 15
        for row_ in self.row:
            row_.export(outfile, level, namespace_, name_='row')
        if self.caption:
            self.caption.export(outfile, level, namespace_, name_='caption')

    def hasContent_(self):
        if False:
            i = 10
            return i + 15
        if self.row is not None or self.caption is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='docTableType'):
        if False:
            i = 10
            return i + 15
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            return 10
        if self.rows is not None:
            showIndent(outfile, level)
            outfile.write('rows = %s,\n' % (self.rows,))
        if self.cols is not None:
            showIndent(outfile, level)
            outfile.write('cols = %s,\n' % (self.cols,))

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            for i in range(10):
                print('nop')
        showIndent(outfile, level)
        outfile.write('row=[\n')
        level += 1
        for row in self.row:
            showIndent(outfile, level)
            outfile.write('model_.row(\n')
            row.exportLiteral(outfile, level, name_='row')
            showIndent(outfile, level)
            outfile.write('),\n')
        level -= 1
        showIndent(outfile, level)
        outfile.write('],\n')
        if self.caption:
            showIndent(outfile, level)
            outfile.write('caption=model_.docCaptionType(\n')
            self.caption.exportLiteral(outfile, level, name_='caption')
            showIndent(outfile, level)
            outfile.write('),\n')

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
            i = 10
            return i + 15
        if attrs.get('rows'):
            try:
                self.rows = int(attrs.get('rows').value)
            except ValueError as exp:
                raise ValueError('Bad integer attribute (rows): %s' % exp)
        if attrs.get('cols'):
            try:
                self.cols = int(attrs.get('cols').value)
            except ValueError as exp:
                raise ValueError('Bad integer attribute (cols): %s' % exp)

    def buildChildren(self, child_, nodeName_):
        if False:
            i = 10
            return i + 15
        if child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'row':
            obj_ = docRowType.factory()
            obj_.build(child_)
            self.row.append(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'caption':
            obj_ = docCaptionType.factory()
            obj_.build(child_)
            self.set_caption(obj_)

class docRowType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, entry=None):
        if False:
            print('Hello World!')
        if entry is None:
            self.entry = []
        else:
            self.entry = entry

    def factory(*args_, **kwargs_):
        if False:
            while True:
                i = 10
        if docRowType.subclass:
            return docRowType.subclass(*args_, **kwargs_)
        else:
            return docRowType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_entry(self):
        if False:
            for i in range(10):
                print('nop')
        return self.entry

    def set_entry(self, entry):
        if False:
            i = 10
            return i + 15
        self.entry = entry

    def add_entry(self, value):
        if False:
            print('Hello World!')
        self.entry.append(value)

    def insert_entry(self, index, value):
        if False:
            while True:
                i = 10
        self.entry[index] = value

    def export(self, outfile, level, namespace_='', name_='docRowType', namespacedef_=''):
        if False:
            print('Hello World!')
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='docRowType')
        if self.hasContent_():
            outfile.write('>\n')
            self.exportChildren(outfile, level + 1, namespace_, name_)
            showIndent(outfile, level)
            outfile.write('</%s%s>\n' % (namespace_, name_))
        else:
            outfile.write(' />\n')

    def exportAttributes(self, outfile, level, namespace_='', name_='docRowType'):
        if False:
            while True:
                i = 10
        pass

    def exportChildren(self, outfile, level, namespace_='', name_='docRowType'):
        if False:
            while True:
                i = 10
        for entry_ in self.entry:
            entry_.export(outfile, level, namespace_, name_='entry')

    def hasContent_(self):
        if False:
            return 10
        if self.entry is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='docRowType'):
        if False:
            i = 10
            return i + 15
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            i = 10
            return i + 15
        pass

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            while True:
                i = 10
        showIndent(outfile, level)
        outfile.write('entry=[\n')
        level += 1
        for entry in self.entry:
            showIndent(outfile, level)
            outfile.write('model_.entry(\n')
            entry.exportLiteral(outfile, level, name_='entry')
            showIndent(outfile, level)
            outfile.write('),\n')
        level -= 1
        showIndent(outfile, level)
        outfile.write('],\n')

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
            print('Hello World!')
        pass

    def buildChildren(self, child_, nodeName_):
        if False:
            i = 10
            return i + 15
        if child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'entry':
            obj_ = docEntryType.factory()
            obj_.build(child_)
            self.entry.append(obj_)

class docEntryType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, thead=None, para=None):
        if False:
            while True:
                i = 10
        self.thead = thead
        if para is None:
            self.para = []
        else:
            self.para = para

    def factory(*args_, **kwargs_):
        if False:
            for i in range(10):
                print('nop')
        if docEntryType.subclass:
            return docEntryType.subclass(*args_, **kwargs_)
        else:
            return docEntryType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_para(self):
        if False:
            for i in range(10):
                print('nop')
        return self.para

    def set_para(self, para):
        if False:
            for i in range(10):
                print('nop')
        self.para = para

    def add_para(self, value):
        if False:
            while True:
                i = 10
        self.para.append(value)

    def insert_para(self, index, value):
        if False:
            i = 10
            return i + 15
        self.para[index] = value

    def get_thead(self):
        if False:
            return 10
        return self.thead

    def set_thead(self, thead):
        if False:
            while True:
                i = 10
        self.thead = thead

    def export(self, outfile, level, namespace_='', name_='docEntryType', namespacedef_=''):
        if False:
            print('Hello World!')
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='docEntryType')
        if self.hasContent_():
            outfile.write('>\n')
            self.exportChildren(outfile, level + 1, namespace_, name_)
            showIndent(outfile, level)
            outfile.write('</%s%s>\n' % (namespace_, name_))
        else:
            outfile.write(' />\n')

    def exportAttributes(self, outfile, level, namespace_='', name_='docEntryType'):
        if False:
            while True:
                i = 10
        if self.thead is not None:
            outfile.write(' thead=%s' % (quote_attrib(self.thead),))

    def exportChildren(self, outfile, level, namespace_='', name_='docEntryType'):
        if False:
            while True:
                i = 10
        for para_ in self.para:
            para_.export(outfile, level, namespace_, name_='para')

    def hasContent_(self):
        if False:
            return 10
        if self.para is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='docEntryType'):
        if False:
            i = 10
            return i + 15
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            while True:
                i = 10
        if self.thead is not None:
            showIndent(outfile, level)
            outfile.write('thead = "%s",\n' % (self.thead,))

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            print('Hello World!')
        showIndent(outfile, level)
        outfile.write('para=[\n')
        level += 1
        for para in self.para:
            showIndent(outfile, level)
            outfile.write('model_.para(\n')
            para.exportLiteral(outfile, level, name_='para')
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
            i = 10
            return i + 15
        if attrs.get('thead'):
            self.thead = attrs.get('thead').value

    def buildChildren(self, child_, nodeName_):
        if False:
            while True:
                i = 10
        if child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'para':
            obj_ = docParaType.factory()
            obj_.build(child_)
            self.para.append(obj_)

class docCaptionType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, valueOf_='', mixedclass_=None, content_=None):
        if False:
            print('Hello World!')
        if mixedclass_ is None:
            self.mixedclass_ = MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        if content_ is None:
            self.content_ = []
        else:
            self.content_ = content_

    def factory(*args_, **kwargs_):
        if False:
            return 10
        if docCaptionType.subclass:
            return docCaptionType.subclass(*args_, **kwargs_)
        else:
            return docCaptionType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def getValueOf_(self):
        if False:
            while True:
                i = 10
        return self.valueOf_

    def setValueOf_(self, valueOf_):
        if False:
            print('Hello World!')
        self.valueOf_ = valueOf_

    def export(self, outfile, level, namespace_='', name_='docCaptionType', namespacedef_=''):
        if False:
            i = 10
            return i + 15
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='docCaptionType')
        outfile.write('>')
        self.exportChildren(outfile, level + 1, namespace_, name_)
        outfile.write('</%s%s>\n' % (namespace_, name_))

    def exportAttributes(self, outfile, level, namespace_='', name_='docCaptionType'):
        if False:
            while True:
                i = 10
        pass

    def exportChildren(self, outfile, level, namespace_='', name_='docCaptionType'):
        if False:
            while True:
                i = 10
        if self.valueOf_.find('![CDATA') > -1:
            value = quote_xml('%s' % self.valueOf_)
            value = value.replace('![CDATA', '<![CDATA')
            value = value.replace(']]', ']]>')
            outfile.write(value)
        else:
            outfile.write(quote_xml('%s' % self.valueOf_))

    def hasContent_(self):
        if False:
            i = 10
            return i + 15
        if self.valueOf_ is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='docCaptionType'):
        if False:
            for i in range(10):
                print('nop')
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            print('Hello World!')
        pass

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            return 10
        showIndent(outfile, level)
        outfile.write('valueOf_ = "%s",\n' % (self.valueOf_,))

    def build(self, node_):
        if False:
            for i in range(10):
                print('nop')
        attrs = node_.attributes
        self.buildAttributes(attrs)
        self.valueOf_ = ''
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            while True:
                i = 10
        pass

    def buildChildren(self, child_, nodeName_):
        if False:
            while True:
                i = 10
        if child_.nodeType == Node.TEXT_NODE:
            obj_ = self.mixedclass_(MixedContainer.CategoryText, MixedContainer.TypeNone, '', child_.nodeValue)
            self.content_.append(obj_)
        if child_.nodeType == Node.TEXT_NODE:
            self.valueOf_ += child_.nodeValue
        elif child_.nodeType == Node.CDATA_SECTION_NODE:
            self.valueOf_ += '![CDATA[' + child_.nodeValue + ']]'

class docHeadingType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, level=None, valueOf_='', mixedclass_=None, content_=None):
        if False:
            i = 10
            return i + 15
        self.level = level
        if mixedclass_ is None:
            self.mixedclass_ = MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        if content_ is None:
            self.content_ = []
        else:
            self.content_ = content_

    def factory(*args_, **kwargs_):
        if False:
            for i in range(10):
                print('nop')
        if docHeadingType.subclass:
            return docHeadingType.subclass(*args_, **kwargs_)
        else:
            return docHeadingType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_level(self):
        if False:
            while True:
                i = 10
        return self.level

    def set_level(self, level):
        if False:
            while True:
                i = 10
        self.level = level

    def getValueOf_(self):
        if False:
            for i in range(10):
                print('nop')
        return self.valueOf_

    def setValueOf_(self, valueOf_):
        if False:
            while True:
                i = 10
        self.valueOf_ = valueOf_

    def export(self, outfile, level, namespace_='', name_='docHeadingType', namespacedef_=''):
        if False:
            while True:
                i = 10
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='docHeadingType')
        outfile.write('>')
        self.exportChildren(outfile, level + 1, namespace_, name_)
        outfile.write('</%s%s>\n' % (namespace_, name_))

    def exportAttributes(self, outfile, level, namespace_='', name_='docHeadingType'):
        if False:
            while True:
                i = 10
        if self.level is not None:
            outfile.write(' level="%s"' % self.format_integer(self.level, input_name='level'))

    def exportChildren(self, outfile, level, namespace_='', name_='docHeadingType'):
        if False:
            print('Hello World!')
        if self.valueOf_.find('![CDATA') > -1:
            value = quote_xml('%s' % self.valueOf_)
            value = value.replace('![CDATA', '<![CDATA')
            value = value.replace(']]', ']]>')
            outfile.write(value)
        else:
            outfile.write(quote_xml('%s' % self.valueOf_))

    def hasContent_(self):
        if False:
            for i in range(10):
                print('nop')
        if self.valueOf_ is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='docHeadingType'):
        if False:
            i = 10
            return i + 15
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            while True:
                i = 10
        if self.level is not None:
            showIndent(outfile, level)
            outfile.write('level = %s,\n' % (self.level,))

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            return 10
        showIndent(outfile, level)
        outfile.write('valueOf_ = "%s",\n' % (self.valueOf_,))

    def build(self, node_):
        if False:
            for i in range(10):
                print('nop')
        attrs = node_.attributes
        self.buildAttributes(attrs)
        self.valueOf_ = ''
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            for i in range(10):
                print('nop')
        if attrs.get('level'):
            try:
                self.level = int(attrs.get('level').value)
            except ValueError as exp:
                raise ValueError('Bad integer attribute (level): %s' % exp)

    def buildChildren(self, child_, nodeName_):
        if False:
            i = 10
            return i + 15
        if child_.nodeType == Node.TEXT_NODE:
            obj_ = self.mixedclass_(MixedContainer.CategoryText, MixedContainer.TypeNone, '', child_.nodeValue)
            self.content_.append(obj_)
        if child_.nodeType == Node.TEXT_NODE:
            self.valueOf_ += child_.nodeValue
        elif child_.nodeType == Node.CDATA_SECTION_NODE:
            self.valueOf_ += '![CDATA[' + child_.nodeValue + ']]'

class docImageType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, width=None, type_=None, name=None, height=None, valueOf_='', mixedclass_=None, content_=None):
        if False:
            while True:
                i = 10
        self.width = width
        self.type_ = type_
        self.name = name
        self.height = height
        if mixedclass_ is None:
            self.mixedclass_ = MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        if content_ is None:
            self.content_ = []
        else:
            self.content_ = content_

    def factory(*args_, **kwargs_):
        if False:
            print('Hello World!')
        if docImageType.subclass:
            return docImageType.subclass(*args_, **kwargs_)
        else:
            return docImageType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_width(self):
        if False:
            for i in range(10):
                print('nop')
        return self.width

    def set_width(self, width):
        if False:
            for i in range(10):
                print('nop')
        self.width = width

    def get_type(self):
        if False:
            return 10
        return self.type_

    def set_type(self, type_):
        if False:
            i = 10
            return i + 15
        self.type_ = type_

    def get_name(self):
        if False:
            print('Hello World!')
        return self.name

    def set_name(self, name):
        if False:
            i = 10
            return i + 15
        self.name = name

    def get_height(self):
        if False:
            while True:
                i = 10
        return self.height

    def set_height(self, height):
        if False:
            print('Hello World!')
        self.height = height

    def getValueOf_(self):
        if False:
            print('Hello World!')
        return self.valueOf_

    def setValueOf_(self, valueOf_):
        if False:
            print('Hello World!')
        self.valueOf_ = valueOf_

    def export(self, outfile, level, namespace_='', name_='docImageType', namespacedef_=''):
        if False:
            while True:
                i = 10
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='docImageType')
        outfile.write('>')
        self.exportChildren(outfile, level + 1, namespace_, name_)
        outfile.write('</%s%s>\n' % (namespace_, name_))

    def exportAttributes(self, outfile, level, namespace_='', name_='docImageType'):
        if False:
            return 10
        if self.width is not None:
            outfile.write(' width=%s' % (self.format_string(quote_attrib(self.width).encode(ExternalEncoding), input_name='width'),))
        if self.type_ is not None:
            outfile.write(' type=%s' % (quote_attrib(self.type_),))
        if self.name is not None:
            outfile.write(' name=%s' % (self.format_string(quote_attrib(self.name).encode(ExternalEncoding), input_name='name'),))
        if self.height is not None:
            outfile.write(' height=%s' % (self.format_string(quote_attrib(self.height).encode(ExternalEncoding), input_name='height'),))

    def exportChildren(self, outfile, level, namespace_='', name_='docImageType'):
        if False:
            return 10
        if self.valueOf_.find('![CDATA') > -1:
            value = quote_xml('%s' % self.valueOf_)
            value = value.replace('![CDATA', '<![CDATA')
            value = value.replace(']]', ']]>')
            outfile.write(value)
        else:
            outfile.write(quote_xml('%s' % self.valueOf_))

    def hasContent_(self):
        if False:
            i = 10
            return i + 15
        if self.valueOf_ is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='docImageType'):
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
        if self.width is not None:
            showIndent(outfile, level)
            outfile.write('width = %s,\n' % (self.width,))
        if self.type_ is not None:
            showIndent(outfile, level)
            outfile.write('type_ = "%s",\n' % (self.type_,))
        if self.name is not None:
            showIndent(outfile, level)
            outfile.write('name = %s,\n' % (self.name,))
        if self.height is not None:
            showIndent(outfile, level)
            outfile.write('height = %s,\n' % (self.height,))

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            i = 10
            return i + 15
        showIndent(outfile, level)
        outfile.write('valueOf_ = "%s",\n' % (self.valueOf_,))

    def build(self, node_):
        if False:
            print('Hello World!')
        attrs = node_.attributes
        self.buildAttributes(attrs)
        self.valueOf_ = ''
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            print('Hello World!')
        if attrs.get('width'):
            self.width = attrs.get('width').value
        if attrs.get('type'):
            self.type_ = attrs.get('type').value
        if attrs.get('name'):
            self.name = attrs.get('name').value
        if attrs.get('height'):
            self.height = attrs.get('height').value

    def buildChildren(self, child_, nodeName_):
        if False:
            i = 10
            return i + 15
        if child_.nodeType == Node.TEXT_NODE:
            obj_ = self.mixedclass_(MixedContainer.CategoryText, MixedContainer.TypeNone, '', child_.nodeValue)
            self.content_.append(obj_)
        if child_.nodeType == Node.TEXT_NODE:
            self.valueOf_ += child_.nodeValue
        elif child_.nodeType == Node.CDATA_SECTION_NODE:
            self.valueOf_ += '![CDATA[' + child_.nodeValue + ']]'

class docDotFileType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, name=None, valueOf_='', mixedclass_=None, content_=None):
        if False:
            print('Hello World!')
        self.name = name
        if mixedclass_ is None:
            self.mixedclass_ = MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        if content_ is None:
            self.content_ = []
        else:
            self.content_ = content_

    def factory(*args_, **kwargs_):
        if False:
            for i in range(10):
                print('nop')
        if docDotFileType.subclass:
            return docDotFileType.subclass(*args_, **kwargs_)
        else:
            return docDotFileType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_name(self):
        if False:
            return 10
        return self.name

    def set_name(self, name):
        if False:
            return 10
        self.name = name

    def getValueOf_(self):
        if False:
            for i in range(10):
                print('nop')
        return self.valueOf_

    def setValueOf_(self, valueOf_):
        if False:
            while True:
                i = 10
        self.valueOf_ = valueOf_

    def export(self, outfile, level, namespace_='', name_='docDotFileType', namespacedef_=''):
        if False:
            while True:
                i = 10
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='docDotFileType')
        outfile.write('>')
        self.exportChildren(outfile, level + 1, namespace_, name_)
        outfile.write('</%s%s>\n' % (namespace_, name_))

    def exportAttributes(self, outfile, level, namespace_='', name_='docDotFileType'):
        if False:
            while True:
                i = 10
        if self.name is not None:
            outfile.write(' name=%s' % (self.format_string(quote_attrib(self.name).encode(ExternalEncoding), input_name='name'),))

    def exportChildren(self, outfile, level, namespace_='', name_='docDotFileType'):
        if False:
            print('Hello World!')
        if self.valueOf_.find('![CDATA') > -1:
            value = quote_xml('%s' % self.valueOf_)
            value = value.replace('![CDATA', '<![CDATA')
            value = value.replace(']]', ']]>')
            outfile.write(value)
        else:
            outfile.write(quote_xml('%s' % self.valueOf_))

    def hasContent_(self):
        if False:
            print('Hello World!')
        if self.valueOf_ is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='docDotFileType'):
        if False:
            return 10
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            return 10
        if self.name is not None:
            showIndent(outfile, level)
            outfile.write('name = %s,\n' % (self.name,))

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            print('Hello World!')
        showIndent(outfile, level)
        outfile.write('valueOf_ = "%s",\n' % (self.valueOf_,))

    def build(self, node_):
        if False:
            return 10
        attrs = node_.attributes
        self.buildAttributes(attrs)
        self.valueOf_ = ''
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            return 10
        if attrs.get('name'):
            self.name = attrs.get('name').value

    def buildChildren(self, child_, nodeName_):
        if False:
            for i in range(10):
                print('nop')
        if child_.nodeType == Node.TEXT_NODE:
            obj_ = self.mixedclass_(MixedContainer.CategoryText, MixedContainer.TypeNone, '', child_.nodeValue)
            self.content_.append(obj_)
        if child_.nodeType == Node.TEXT_NODE:
            self.valueOf_ += child_.nodeValue
        elif child_.nodeType == Node.CDATA_SECTION_NODE:
            self.valueOf_ += '![CDATA[' + child_.nodeValue + ']]'

class docTocItemType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, id=None, valueOf_='', mixedclass_=None, content_=None):
        if False:
            for i in range(10):
                print('nop')
        self.id = id
        if mixedclass_ is None:
            self.mixedclass_ = MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        if content_ is None:
            self.content_ = []
        else:
            self.content_ = content_

    def factory(*args_, **kwargs_):
        if False:
            i = 10
            return i + 15
        if docTocItemType.subclass:
            return docTocItemType.subclass(*args_, **kwargs_)
        else:
            return docTocItemType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_id(self):
        if False:
            while True:
                i = 10
        return self.id

    def set_id(self, id):
        if False:
            i = 10
            return i + 15
        self.id = id

    def getValueOf_(self):
        if False:
            i = 10
            return i + 15
        return self.valueOf_

    def setValueOf_(self, valueOf_):
        if False:
            while True:
                i = 10
        self.valueOf_ = valueOf_

    def export(self, outfile, level, namespace_='', name_='docTocItemType', namespacedef_=''):
        if False:
            while True:
                i = 10
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='docTocItemType')
        outfile.write('>')
        self.exportChildren(outfile, level + 1, namespace_, name_)
        outfile.write('</%s%s>\n' % (namespace_, name_))

    def exportAttributes(self, outfile, level, namespace_='', name_='docTocItemType'):
        if False:
            i = 10
            return i + 15
        if self.id is not None:
            outfile.write(' id=%s' % (self.format_string(quote_attrib(self.id).encode(ExternalEncoding), input_name='id'),))

    def exportChildren(self, outfile, level, namespace_='', name_='docTocItemType'):
        if False:
            print('Hello World!')
        if self.valueOf_.find('![CDATA') > -1:
            value = quote_xml('%s' % self.valueOf_)
            value = value.replace('![CDATA', '<![CDATA')
            value = value.replace(']]', ']]>')
            outfile.write(value)
        else:
            outfile.write(quote_xml('%s' % self.valueOf_))

    def hasContent_(self):
        if False:
            print('Hello World!')
        if self.valueOf_ is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='docTocItemType'):
        if False:
            for i in range(10):
                print('nop')
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            i = 10
            return i + 15
        if self.id is not None:
            showIndent(outfile, level)
            outfile.write('id = %s,\n' % (self.id,))

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            for i in range(10):
                print('nop')
        showIndent(outfile, level)
        outfile.write('valueOf_ = "%s",\n' % (self.valueOf_,))

    def build(self, node_):
        if False:
            print('Hello World!')
        attrs = node_.attributes
        self.buildAttributes(attrs)
        self.valueOf_ = ''
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            i = 10
            return i + 15
        if attrs.get('id'):
            self.id = attrs.get('id').value

    def buildChildren(self, child_, nodeName_):
        if False:
            while True:
                i = 10
        if child_.nodeType == Node.TEXT_NODE:
            obj_ = self.mixedclass_(MixedContainer.CategoryText, MixedContainer.TypeNone, '', child_.nodeValue)
            self.content_.append(obj_)
        if child_.nodeType == Node.TEXT_NODE:
            self.valueOf_ += child_.nodeValue
        elif child_.nodeType == Node.CDATA_SECTION_NODE:
            self.valueOf_ += '![CDATA[' + child_.nodeValue + ']]'

class docTocListType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, tocitem=None):
        if False:
            i = 10
            return i + 15
        if tocitem is None:
            self.tocitem = []
        else:
            self.tocitem = tocitem

    def factory(*args_, **kwargs_):
        if False:
            print('Hello World!')
        if docTocListType.subclass:
            return docTocListType.subclass(*args_, **kwargs_)
        else:
            return docTocListType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_tocitem(self):
        if False:
            print('Hello World!')
        return self.tocitem

    def set_tocitem(self, tocitem):
        if False:
            return 10
        self.tocitem = tocitem

    def add_tocitem(self, value):
        if False:
            print('Hello World!')
        self.tocitem.append(value)

    def insert_tocitem(self, index, value):
        if False:
            for i in range(10):
                print('nop')
        self.tocitem[index] = value

    def export(self, outfile, level, namespace_='', name_='docTocListType', namespacedef_=''):
        if False:
            i = 10
            return i + 15
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='docTocListType')
        if self.hasContent_():
            outfile.write('>\n')
            self.exportChildren(outfile, level + 1, namespace_, name_)
            showIndent(outfile, level)
            outfile.write('</%s%s>\n' % (namespace_, name_))
        else:
            outfile.write(' />\n')

    def exportAttributes(self, outfile, level, namespace_='', name_='docTocListType'):
        if False:
            i = 10
            return i + 15
        pass

    def exportChildren(self, outfile, level, namespace_='', name_='docTocListType'):
        if False:
            return 10
        for tocitem_ in self.tocitem:
            tocitem_.export(outfile, level, namespace_, name_='tocitem')

    def hasContent_(self):
        if False:
            i = 10
            return i + 15
        if self.tocitem is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='docTocListType'):
        if False:
            i = 10
            return i + 15
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            while True:
                i = 10
        pass

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            while True:
                i = 10
        showIndent(outfile, level)
        outfile.write('tocitem=[\n')
        level += 1
        for tocitem in self.tocitem:
            showIndent(outfile, level)
            outfile.write('model_.tocitem(\n')
            tocitem.exportLiteral(outfile, level, name_='tocitem')
            showIndent(outfile, level)
            outfile.write('),\n')
        level -= 1
        showIndent(outfile, level)
        outfile.write('],\n')

    def build(self, node_):
        if False:
            i = 10
            return i + 15
        attrs = node_.attributes
        self.buildAttributes(attrs)
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            return 10
        pass

    def buildChildren(self, child_, nodeName_):
        if False:
            print('Hello World!')
        if child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'tocitem':
            obj_ = docTocItemType.factory()
            obj_.build(child_)
            self.tocitem.append(obj_)

class docLanguageType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, langid=None, para=None):
        if False:
            i = 10
            return i + 15
        self.langid = langid
        if para is None:
            self.para = []
        else:
            self.para = para

    def factory(*args_, **kwargs_):
        if False:
            for i in range(10):
                print('nop')
        if docLanguageType.subclass:
            return docLanguageType.subclass(*args_, **kwargs_)
        else:
            return docLanguageType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_para(self):
        if False:
            print('Hello World!')
        return self.para

    def set_para(self, para):
        if False:
            while True:
                i = 10
        self.para = para

    def add_para(self, value):
        if False:
            return 10
        self.para.append(value)

    def insert_para(self, index, value):
        if False:
            for i in range(10):
                print('nop')
        self.para[index] = value

    def get_langid(self):
        if False:
            return 10
        return self.langid

    def set_langid(self, langid):
        if False:
            print('Hello World!')
        self.langid = langid

    def export(self, outfile, level, namespace_='', name_='docLanguageType', namespacedef_=''):
        if False:
            return 10
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='docLanguageType')
        if self.hasContent_():
            outfile.write('>\n')
            self.exportChildren(outfile, level + 1, namespace_, name_)
            showIndent(outfile, level)
            outfile.write('</%s%s>\n' % (namespace_, name_))
        else:
            outfile.write(' />\n')

    def exportAttributes(self, outfile, level, namespace_='', name_='docLanguageType'):
        if False:
            while True:
                i = 10
        if self.langid is not None:
            outfile.write(' langid=%s' % (self.format_string(quote_attrib(self.langid).encode(ExternalEncoding), input_name='langid'),))

    def exportChildren(self, outfile, level, namespace_='', name_='docLanguageType'):
        if False:
            return 10
        for para_ in self.para:
            para_.export(outfile, level, namespace_, name_='para')

    def hasContent_(self):
        if False:
            return 10
        if self.para is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='docLanguageType'):
        if False:
            while True:
                i = 10
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            i = 10
            return i + 15
        if self.langid is not None:
            showIndent(outfile, level)
            outfile.write('langid = %s,\n' % (self.langid,))

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            while True:
                i = 10
        showIndent(outfile, level)
        outfile.write('para=[\n')
        level += 1
        for para in self.para:
            showIndent(outfile, level)
            outfile.write('model_.para(\n')
            para.exportLiteral(outfile, level, name_='para')
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
            i = 10
            return i + 15
        if attrs.get('langid'):
            self.langid = attrs.get('langid').value

    def buildChildren(self, child_, nodeName_):
        if False:
            return 10
        if child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'para':
            obj_ = docParaType.factory()
            obj_.build(child_)
            self.para.append(obj_)

class docParamListType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, kind=None, parameteritem=None):
        if False:
            i = 10
            return i + 15
        self.kind = kind
        if parameteritem is None:
            self.parameteritem = []
        else:
            self.parameteritem = parameteritem

    def factory(*args_, **kwargs_):
        if False:
            while True:
                i = 10
        if docParamListType.subclass:
            return docParamListType.subclass(*args_, **kwargs_)
        else:
            return docParamListType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_parameteritem(self):
        if False:
            return 10
        return self.parameteritem

    def set_parameteritem(self, parameteritem):
        if False:
            i = 10
            return i + 15
        self.parameteritem = parameteritem

    def add_parameteritem(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.parameteritem.append(value)

    def insert_parameteritem(self, index, value):
        if False:
            for i in range(10):
                print('nop')
        self.parameteritem[index] = value

    def get_kind(self):
        if False:
            return 10
        return self.kind

    def set_kind(self, kind):
        if False:
            return 10
        self.kind = kind

    def export(self, outfile, level, namespace_='', name_='docParamListType', namespacedef_=''):
        if False:
            return 10
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='docParamListType')
        if self.hasContent_():
            outfile.write('>\n')
            self.exportChildren(outfile, level + 1, namespace_, name_)
            showIndent(outfile, level)
            outfile.write('</%s%s>\n' % (namespace_, name_))
        else:
            outfile.write(' />\n')

    def exportAttributes(self, outfile, level, namespace_='', name_='docParamListType'):
        if False:
            while True:
                i = 10
        if self.kind is not None:
            outfile.write(' kind=%s' % (quote_attrib(self.kind),))

    def exportChildren(self, outfile, level, namespace_='', name_='docParamListType'):
        if False:
            while True:
                i = 10
        for parameteritem_ in self.parameteritem:
            parameteritem_.export(outfile, level, namespace_, name_='parameteritem')

    def hasContent_(self):
        if False:
            print('Hello World!')
        if self.parameteritem is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='docParamListType'):
        if False:
            i = 10
            return i + 15
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

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            print('Hello World!')
        showIndent(outfile, level)
        outfile.write('parameteritem=[\n')
        level += 1
        for parameteritem in self.parameteritem:
            showIndent(outfile, level)
            outfile.write('model_.parameteritem(\n')
            parameteritem.exportLiteral(outfile, level, name_='parameteritem')
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
            for i in range(10):
                print('nop')
        if attrs.get('kind'):
            self.kind = attrs.get('kind').value

    def buildChildren(self, child_, nodeName_):
        if False:
            while True:
                i = 10
        if child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'parameteritem':
            obj_ = docParamListItem.factory()
            obj_.build(child_)
            self.parameteritem.append(obj_)

class docParamListItem(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, parameternamelist=None, parameterdescription=None):
        if False:
            for i in range(10):
                print('nop')
        if parameternamelist is None:
            self.parameternamelist = []
        else:
            self.parameternamelist = parameternamelist
        self.parameterdescription = parameterdescription

    def factory(*args_, **kwargs_):
        if False:
            print('Hello World!')
        if docParamListItem.subclass:
            return docParamListItem.subclass(*args_, **kwargs_)
        else:
            return docParamListItem(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_parameternamelist(self):
        if False:
            i = 10
            return i + 15
        return self.parameternamelist

    def set_parameternamelist(self, parameternamelist):
        if False:
            return 10
        self.parameternamelist = parameternamelist

    def add_parameternamelist(self, value):
        if False:
            i = 10
            return i + 15
        self.parameternamelist.append(value)

    def insert_parameternamelist(self, index, value):
        if False:
            while True:
                i = 10
        self.parameternamelist[index] = value

    def get_parameterdescription(self):
        if False:
            for i in range(10):
                print('nop')
        return self.parameterdescription

    def set_parameterdescription(self, parameterdescription):
        if False:
            i = 10
            return i + 15
        self.parameterdescription = parameterdescription

    def export(self, outfile, level, namespace_='', name_='docParamListItem', namespacedef_=''):
        if False:
            for i in range(10):
                print('nop')
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='docParamListItem')
        if self.hasContent_():
            outfile.write('>\n')
            self.exportChildren(outfile, level + 1, namespace_, name_)
            showIndent(outfile, level)
            outfile.write('</%s%s>\n' % (namespace_, name_))
        else:
            outfile.write(' />\n')

    def exportAttributes(self, outfile, level, namespace_='', name_='docParamListItem'):
        if False:
            while True:
                i = 10
        pass

    def exportChildren(self, outfile, level, namespace_='', name_='docParamListItem'):
        if False:
            while True:
                i = 10
        for parameternamelist_ in self.parameternamelist:
            parameternamelist_.export(outfile, level, namespace_, name_='parameternamelist')
        if self.parameterdescription:
            self.parameterdescription.export(outfile, level, namespace_, name_='parameterdescription')

    def hasContent_(self):
        if False:
            i = 10
            return i + 15
        if self.parameternamelist is not None or self.parameterdescription is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='docParamListItem'):
        if False:
            return 10
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            while True:
                i = 10
        pass

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            print('Hello World!')
        showIndent(outfile, level)
        outfile.write('parameternamelist=[\n')
        level += 1
        for parameternamelist in self.parameternamelist:
            showIndent(outfile, level)
            outfile.write('model_.parameternamelist(\n')
            parameternamelist.exportLiteral(outfile, level, name_='parameternamelist')
            showIndent(outfile, level)
            outfile.write('),\n')
        level -= 1
        showIndent(outfile, level)
        outfile.write('],\n')
        if self.parameterdescription:
            showIndent(outfile, level)
            outfile.write('parameterdescription=model_.descriptionType(\n')
            self.parameterdescription.exportLiteral(outfile, level, name_='parameterdescription')
            showIndent(outfile, level)
            outfile.write('),\n')

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
            i = 10
            return i + 15
        pass

    def buildChildren(self, child_, nodeName_):
        if False:
            return 10
        if child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'parameternamelist':
            obj_ = docParamNameList.factory()
            obj_.build(child_)
            self.parameternamelist.append(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'parameterdescription':
            obj_ = descriptionType.factory()
            obj_.build(child_)
            self.set_parameterdescription(obj_)

class docParamNameList(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, parametername=None):
        if False:
            for i in range(10):
                print('nop')
        if parametername is None:
            self.parametername = []
        else:
            self.parametername = parametername

    def factory(*args_, **kwargs_):
        if False:
            print('Hello World!')
        if docParamNameList.subclass:
            return docParamNameList.subclass(*args_, **kwargs_)
        else:
            return docParamNameList(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_parametername(self):
        if False:
            for i in range(10):
                print('nop')
        return self.parametername

    def set_parametername(self, parametername):
        if False:
            i = 10
            return i + 15
        self.parametername = parametername

    def add_parametername(self, value):
        if False:
            i = 10
            return i + 15
        self.parametername.append(value)

    def insert_parametername(self, index, value):
        if False:
            return 10
        self.parametername[index] = value

    def export(self, outfile, level, namespace_='', name_='docParamNameList', namespacedef_=''):
        if False:
            for i in range(10):
                print('nop')
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='docParamNameList')
        if self.hasContent_():
            outfile.write('>\n')
            self.exportChildren(outfile, level + 1, namespace_, name_)
            showIndent(outfile, level)
            outfile.write('</%s%s>\n' % (namespace_, name_))
        else:
            outfile.write(' />\n')

    def exportAttributes(self, outfile, level, namespace_='', name_='docParamNameList'):
        if False:
            print('Hello World!')
        pass

    def exportChildren(self, outfile, level, namespace_='', name_='docParamNameList'):
        if False:
            return 10
        for parametername_ in self.parametername:
            parametername_.export(outfile, level, namespace_, name_='parametername')

    def hasContent_(self):
        if False:
            for i in range(10):
                print('nop')
        if self.parametername is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='docParamNameList'):
        if False:
            return 10
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            return 10
        pass

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            i = 10
            return i + 15
        showIndent(outfile, level)
        outfile.write('parametername=[\n')
        level += 1
        for parametername in self.parametername:
            showIndent(outfile, level)
            outfile.write('model_.parametername(\n')
            parametername.exportLiteral(outfile, level, name_='parametername')
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
            for i in range(10):
                print('nop')
        pass

    def buildChildren(self, child_, nodeName_):
        if False:
            for i in range(10):
                print('nop')
        if child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'parametername':
            obj_ = docParamName.factory()
            obj_.build(child_)
            self.parametername.append(obj_)

class docParamName(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, direction=None, ref=None, mixedclass_=None, content_=None):
        if False:
            i = 10
            return i + 15
        self.direction = direction
        if mixedclass_ is None:
            self.mixedclass_ = MixedContainer
        else:
            self.mixedclass_ = mixedclass_
        if content_ is None:
            self.content_ = []
        else:
            self.content_ = content_

    def factory(*args_, **kwargs_):
        if False:
            print('Hello World!')
        if docParamName.subclass:
            return docParamName.subclass(*args_, **kwargs_)
        else:
            return docParamName(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_ref(self):
        if False:
            while True:
                i = 10
        return self.ref

    def set_ref(self, ref):
        if False:
            i = 10
            return i + 15
        self.ref = ref

    def get_direction(self):
        if False:
            i = 10
            return i + 15
        return self.direction

    def set_direction(self, direction):
        if False:
            i = 10
            return i + 15
        self.direction = direction

    def export(self, outfile, level, namespace_='', name_='docParamName', namespacedef_=''):
        if False:
            print('Hello World!')
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='docParamName')
        outfile.write('>')
        self.exportChildren(outfile, level + 1, namespace_, name_)
        outfile.write('</%s%s>\n' % (namespace_, name_))

    def exportAttributes(self, outfile, level, namespace_='', name_='docParamName'):
        if False:
            while True:
                i = 10
        if self.direction is not None:
            outfile.write(' direction=%s' % (quote_attrib(self.direction),))

    def exportChildren(self, outfile, level, namespace_='', name_='docParamName'):
        if False:
            for i in range(10):
                print('nop')
        for item_ in self.content_:
            item_.export(outfile, level, item_.name, namespace_)

    def hasContent_(self):
        if False:
            i = 10
            return i + 15
        if self.ref is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='docParamName'):
        if False:
            return 10
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            i = 10
            return i + 15
        if self.direction is not None:
            showIndent(outfile, level)
            outfile.write('direction = "%s",\n' % (self.direction,))

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            for i in range(10):
                print('nop')
        showIndent(outfile, level)
        outfile.write('content_ = [\n')
        for item_ in self.content_:
            item_.exportLiteral(outfile, level, name_)
        showIndent(outfile, level)
        outfile.write('],\n')

    def build(self, node_):
        if False:
            return 10
        attrs = node_.attributes
        self.buildAttributes(attrs)
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            i = 10
            return i + 15
        if attrs.get('direction'):
            self.direction = attrs.get('direction').value

    def buildChildren(self, child_, nodeName_):
        if False:
            for i in range(10):
                print('nop')
        if child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'ref':
            childobj_ = docRefTextType.factory()
            childobj_.build(child_)
            obj_ = self.mixedclass_(MixedContainer.CategoryComplex, MixedContainer.TypeNone, 'ref', childobj_)
            self.content_.append(obj_)
        elif child_.nodeType == Node.TEXT_NODE:
            obj_ = self.mixedclass_(MixedContainer.CategoryText, MixedContainer.TypeNone, '', child_.nodeValue)
            self.content_.append(obj_)

class docXRefSectType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, id=None, xreftitle=None, xrefdescription=None):
        if False:
            print('Hello World!')
        self.id = id
        if xreftitle is None:
            self.xreftitle = []
        else:
            self.xreftitle = xreftitle
        self.xrefdescription = xrefdescription

    def factory(*args_, **kwargs_):
        if False:
            for i in range(10):
                print('nop')
        if docXRefSectType.subclass:
            return docXRefSectType.subclass(*args_, **kwargs_)
        else:
            return docXRefSectType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_xreftitle(self):
        if False:
            return 10
        return self.xreftitle

    def set_xreftitle(self, xreftitle):
        if False:
            for i in range(10):
                print('nop')
        self.xreftitle = xreftitle

    def add_xreftitle(self, value):
        if False:
            return 10
        self.xreftitle.append(value)

    def insert_xreftitle(self, index, value):
        if False:
            for i in range(10):
                print('nop')
        self.xreftitle[index] = value

    def get_xrefdescription(self):
        if False:
            i = 10
            return i + 15
        return self.xrefdescription

    def set_xrefdescription(self, xrefdescription):
        if False:
            for i in range(10):
                print('nop')
        self.xrefdescription = xrefdescription

    def get_id(self):
        if False:
            while True:
                i = 10
        return self.id

    def set_id(self, id):
        if False:
            print('Hello World!')
        self.id = id

    def export(self, outfile, level, namespace_='', name_='docXRefSectType', namespacedef_=''):
        if False:
            return 10
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='docXRefSectType')
        if self.hasContent_():
            outfile.write('>\n')
            self.exportChildren(outfile, level + 1, namespace_, name_)
            showIndent(outfile, level)
            outfile.write('</%s%s>\n' % (namespace_, name_))
        else:
            outfile.write(' />\n')

    def exportAttributes(self, outfile, level, namespace_='', name_='docXRefSectType'):
        if False:
            while True:
                i = 10
        if self.id is not None:
            outfile.write(' id=%s' % (self.format_string(quote_attrib(self.id).encode(ExternalEncoding), input_name='id'),))

    def exportChildren(self, outfile, level, namespace_='', name_='docXRefSectType'):
        if False:
            i = 10
            return i + 15
        for xreftitle_ in self.xreftitle:
            showIndent(outfile, level)
            outfile.write('<%sxreftitle>%s</%sxreftitle>\n' % (namespace_, self.format_string(quote_xml(xreftitle_).encode(ExternalEncoding), input_name='xreftitle'), namespace_))
        if self.xrefdescription:
            self.xrefdescription.export(outfile, level, namespace_, name_='xrefdescription')

    def hasContent_(self):
        if False:
            while True:
                i = 10
        if self.xreftitle is not None or self.xrefdescription is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='docXRefSectType'):
        if False:
            return 10
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            while True:
                i = 10
        if self.id is not None:
            showIndent(outfile, level)
            outfile.write('id = %s,\n' % (self.id,))

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            for i in range(10):
                print('nop')
        showIndent(outfile, level)
        outfile.write('xreftitle=[\n')
        level += 1
        for xreftitle in self.xreftitle:
            showIndent(outfile, level)
            outfile.write('%s,\n' % quote_python(xreftitle).encode(ExternalEncoding))
        level -= 1
        showIndent(outfile, level)
        outfile.write('],\n')
        if self.xrefdescription:
            showIndent(outfile, level)
            outfile.write('xrefdescription=model_.descriptionType(\n')
            self.xrefdescription.exportLiteral(outfile, level, name_='xrefdescription')
            showIndent(outfile, level)
            outfile.write('),\n')

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
            print('Hello World!')
        if attrs.get('id'):
            self.id = attrs.get('id').value

    def buildChildren(self, child_, nodeName_):
        if False:
            return 10
        if child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'xreftitle':
            xreftitle_ = ''
            for text__content_ in child_.childNodes:
                xreftitle_ += text__content_.nodeValue
            self.xreftitle.append(xreftitle_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'xrefdescription':
            obj_ = descriptionType.factory()
            obj_.build(child_)
            self.set_xrefdescription(obj_)

class docCopyType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, link=None, para=None, sect1=None, internal=None):
        if False:
            while True:
                i = 10
        self.link = link
        if para is None:
            self.para = []
        else:
            self.para = para
        if sect1 is None:
            self.sect1 = []
        else:
            self.sect1 = sect1
        self.internal = internal

    def factory(*args_, **kwargs_):
        if False:
            i = 10
            return i + 15
        if docCopyType.subclass:
            return docCopyType.subclass(*args_, **kwargs_)
        else:
            return docCopyType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_para(self):
        if False:
            for i in range(10):
                print('nop')
        return self.para

    def set_para(self, para):
        if False:
            i = 10
            return i + 15
        self.para = para

    def add_para(self, value):
        if False:
            return 10
        self.para.append(value)

    def insert_para(self, index, value):
        if False:
            for i in range(10):
                print('nop')
        self.para[index] = value

    def get_sect1(self):
        if False:
            return 10
        return self.sect1

    def set_sect1(self, sect1):
        if False:
            while True:
                i = 10
        self.sect1 = sect1

    def add_sect1(self, value):
        if False:
            i = 10
            return i + 15
        self.sect1.append(value)

    def insert_sect1(self, index, value):
        if False:
            while True:
                i = 10
        self.sect1[index] = value

    def get_internal(self):
        if False:
            while True:
                i = 10
        return self.internal

    def set_internal(self, internal):
        if False:
            i = 10
            return i + 15
        self.internal = internal

    def get_link(self):
        if False:
            return 10
        return self.link

    def set_link(self, link):
        if False:
            while True:
                i = 10
        self.link = link

    def export(self, outfile, level, namespace_='', name_='docCopyType', namespacedef_=''):
        if False:
            print('Hello World!')
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='docCopyType')
        if self.hasContent_():
            outfile.write('>\n')
            self.exportChildren(outfile, level + 1, namespace_, name_)
            showIndent(outfile, level)
            outfile.write('</%s%s>\n' % (namespace_, name_))
        else:
            outfile.write(' />\n')

    def exportAttributes(self, outfile, level, namespace_='', name_='docCopyType'):
        if False:
            i = 10
            return i + 15
        if self.link is not None:
            outfile.write(' link=%s' % (self.format_string(quote_attrib(self.link).encode(ExternalEncoding), input_name='link'),))

    def exportChildren(self, outfile, level, namespace_='', name_='docCopyType'):
        if False:
            print('Hello World!')
        for para_ in self.para:
            para_.export(outfile, level, namespace_, name_='para')
        for sect1_ in self.sect1:
            sect1_.export(outfile, level, namespace_, name_='sect1')
        if self.internal:
            self.internal.export(outfile, level, namespace_, name_='internal')

    def hasContent_(self):
        if False:
            i = 10
            return i + 15
        if self.para is not None or self.sect1 is not None or self.internal is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='docCopyType'):
        if False:
            print('Hello World!')
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            return 10
        if self.link is not None:
            showIndent(outfile, level)
            outfile.write('link = %s,\n' % (self.link,))

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            print('Hello World!')
        showIndent(outfile, level)
        outfile.write('para=[\n')
        level += 1
        for para in self.para:
            showIndent(outfile, level)
            outfile.write('model_.para(\n')
            para.exportLiteral(outfile, level, name_='para')
            showIndent(outfile, level)
            outfile.write('),\n')
        level -= 1
        showIndent(outfile, level)
        outfile.write('],\n')
        showIndent(outfile, level)
        outfile.write('sect1=[\n')
        level += 1
        for sect1 in self.sect1:
            showIndent(outfile, level)
            outfile.write('model_.sect1(\n')
            sect1.exportLiteral(outfile, level, name_='sect1')
            showIndent(outfile, level)
            outfile.write('),\n')
        level -= 1
        showIndent(outfile, level)
        outfile.write('],\n')
        if self.internal:
            showIndent(outfile, level)
            outfile.write('internal=model_.docInternalType(\n')
            self.internal.exportLiteral(outfile, level, name_='internal')
            showIndent(outfile, level)
            outfile.write('),\n')

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
        if attrs.get('link'):
            self.link = attrs.get('link').value

    def buildChildren(self, child_, nodeName_):
        if False:
            return 10
        if child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'para':
            obj_ = docParaType.factory()
            obj_.build(child_)
            self.para.append(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'sect1':
            obj_ = docSect1Type.factory()
            obj_.build(child_)
            self.sect1.append(obj_)
        elif child_.nodeType == Node.ELEMENT_NODE and nodeName_ == 'internal':
            obj_ = docInternalType.factory()
            obj_.build(child_)
            self.set_internal(obj_)

class docCharType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, char=None, valueOf_=''):
        if False:
            i = 10
            return i + 15
        self.char = char
        self.valueOf_ = valueOf_

    def factory(*args_, **kwargs_):
        if False:
            print('Hello World!')
        if docCharType.subclass:
            return docCharType.subclass(*args_, **kwargs_)
        else:
            return docCharType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def get_char(self):
        if False:
            print('Hello World!')
        return self.char

    def set_char(self, char):
        if False:
            print('Hello World!')
        self.char = char

    def getValueOf_(self):
        if False:
            i = 10
            return i + 15
        return self.valueOf_

    def setValueOf_(self, valueOf_):
        if False:
            while True:
                i = 10
        self.valueOf_ = valueOf_

    def export(self, outfile, level, namespace_='', name_='docCharType', namespacedef_=''):
        if False:
            print('Hello World!')
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='docCharType')
        if self.hasContent_():
            outfile.write('>\n')
            self.exportChildren(outfile, level + 1, namespace_, name_)
            showIndent(outfile, level)
            outfile.write('</%s%s>\n' % (namespace_, name_))
        else:
            outfile.write(' />\n')

    def exportAttributes(self, outfile, level, namespace_='', name_='docCharType'):
        if False:
            i = 10
            return i + 15
        if self.char is not None:
            outfile.write(' char=%s' % (quote_attrib(self.char),))

    def exportChildren(self, outfile, level, namespace_='', name_='docCharType'):
        if False:
            i = 10
            return i + 15
        if self.valueOf_.find('![CDATA') > -1:
            value = quote_xml('%s' % self.valueOf_)
            value = value.replace('![CDATA', '<![CDATA')
            value = value.replace(']]', ']]>')
            outfile.write(value)
        else:
            outfile.write(quote_xml('%s' % self.valueOf_))

    def hasContent_(self):
        if False:
            i = 10
            return i + 15
        if self.valueOf_ is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='docCharType'):
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
        if self.char is not None:
            showIndent(outfile, level)
            outfile.write('char = "%s",\n' % (self.char,))

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            return 10
        showIndent(outfile, level)
        outfile.write('valueOf_ = "%s",\n' % (self.valueOf_,))

    def build(self, node_):
        if False:
            for i in range(10):
                print('nop')
        attrs = node_.attributes
        self.buildAttributes(attrs)
        self.valueOf_ = ''
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            i = 10
            return i + 15
        if attrs.get('char'):
            self.char = attrs.get('char').value

    def buildChildren(self, child_, nodeName_):
        if False:
            for i in range(10):
                print('nop')
        if child_.nodeType == Node.TEXT_NODE:
            self.valueOf_ += child_.nodeValue
        elif child_.nodeType == Node.CDATA_SECTION_NODE:
            self.valueOf_ += '![CDATA[' + child_.nodeValue + ']]'

class docEmptyType(GeneratedsSuper):
    subclass = None
    superclass = None

    def __init__(self, valueOf_=''):
        if False:
            for i in range(10):
                print('nop')
        self.valueOf_ = valueOf_

    def factory(*args_, **kwargs_):
        if False:
            while True:
                i = 10
        if docEmptyType.subclass:
            return docEmptyType.subclass(*args_, **kwargs_)
        else:
            return docEmptyType(*args_, **kwargs_)
    factory = staticmethod(factory)

    def getValueOf_(self):
        if False:
            i = 10
            return i + 15
        return self.valueOf_

    def setValueOf_(self, valueOf_):
        if False:
            for i in range(10):
                print('nop')
        self.valueOf_ = valueOf_

    def export(self, outfile, level, namespace_='', name_='docEmptyType', namespacedef_=''):
        if False:
            while True:
                i = 10
        showIndent(outfile, level)
        outfile.write('<%s%s %s' % (namespace_, name_, namespacedef_))
        self.exportAttributes(outfile, level, namespace_, name_='docEmptyType')
        if self.hasContent_():
            outfile.write('>\n')
            self.exportChildren(outfile, level + 1, namespace_, name_)
            showIndent(outfile, level)
            outfile.write('</%s%s>\n' % (namespace_, name_))
        else:
            outfile.write(' />\n')

    def exportAttributes(self, outfile, level, namespace_='', name_='docEmptyType'):
        if False:
            print('Hello World!')
        pass

    def exportChildren(self, outfile, level, namespace_='', name_='docEmptyType'):
        if False:
            for i in range(10):
                print('nop')
        if self.valueOf_.find('![CDATA') > -1:
            value = quote_xml('%s' % self.valueOf_)
            value = value.replace('![CDATA', '<![CDATA')
            value = value.replace(']]', ']]>')
            outfile.write(value)
        else:
            outfile.write(quote_xml('%s' % self.valueOf_))

    def hasContent_(self):
        if False:
            return 10
        if self.valueOf_ is not None:
            return True
        else:
            return False

    def exportLiteral(self, outfile, level, name_='docEmptyType'):
        if False:
            i = 10
            return i + 15
        level += 1
        self.exportLiteralAttributes(outfile, level, name_)
        if self.hasContent_():
            self.exportLiteralChildren(outfile, level, name_)

    def exportLiteralAttributes(self, outfile, level, name_):
        if False:
            for i in range(10):
                print('nop')
        pass

    def exportLiteralChildren(self, outfile, level, name_):
        if False:
            print('Hello World!')
        showIndent(outfile, level)
        outfile.write('valueOf_ = "%s",\n' % (self.valueOf_,))

    def build(self, node_):
        if False:
            while True:
                i = 10
        attrs = node_.attributes
        self.buildAttributes(attrs)
        self.valueOf_ = ''
        for child_ in node_.childNodes:
            nodeName_ = child_.nodeName.split(':')[-1]
            self.buildChildren(child_, nodeName_)

    def buildAttributes(self, attrs):
        if False:
            return 10
        pass

    def buildChildren(self, child_, nodeName_):
        if False:
            for i in range(10):
                print('nop')
        if child_.nodeType == Node.TEXT_NODE:
            self.valueOf_ += child_.nodeValue
        elif child_.nodeType == Node.CDATA_SECTION_NODE:
            self.valueOf_ += '![CDATA[' + child_.nodeValue + ']]'
USAGE_TEXT = '\nUsage: python <Parser>.py [ -s ] <in_xml_file>\nOptions:\n    -s        Use the SAX parser, not the minidom parser.\n'

def usage():
    if False:
        for i in range(10):
            print('nop')
    print(USAGE_TEXT)
    sys.exit(1)

def parse(inFileName):
    if False:
        for i in range(10):
            print('nop')
    doc = minidom.parse(inFileName)
    rootNode = doc.documentElement
    rootObj = DoxygenType.factory()
    rootObj.build(rootNode)
    doc = None
    sys.stdout.write('<?xml version="1.0" ?>\n')
    rootObj.export(sys.stdout, 0, name_='doxygen', namespacedef_='')
    return rootObj

def parseString(inString):
    if False:
        i = 10
        return i + 15
    doc = minidom.parseString(inString)
    rootNode = doc.documentElement
    rootObj = DoxygenType.factory()
    rootObj.build(rootNode)
    doc = None
    sys.stdout.write('<?xml version="1.0" ?>\n')
    rootObj.export(sys.stdout, 0, name_='doxygen', namespacedef_='')
    return rootObj

def parseLiteral(inFileName):
    if False:
        return 10
    doc = minidom.parse(inFileName)
    rootNode = doc.documentElement
    rootObj = DoxygenType.factory()
    rootObj.build(rootNode)
    doc = None
    sys.stdout.write('from compound import *\n\n')
    sys.stdout.write('rootObj = doxygen(\n')
    rootObj.exportLiteral(sys.stdout, 0, name_='doxygen')
    sys.stdout.write(')\n')
    return rootObj

def main():
    if False:
        i = 10
        return i + 15
    args = sys.argv[1:]
    if len(args) == 1:
        parse(args[0])
    else:
        usage()
if __name__ == '__main__':
    main()