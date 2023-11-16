from xml.dom import minidom
from xml.dom import Node

def format(text):
    if False:
        while True:
            i = 10
    doc = minidom.parseString(text)
    root = doc.childNodes[0]
    return root.toprettyxml(indent='  ')

def formatXML(doc, encoding=None):
    if False:
        return 10
    root = doc.childNodes[0]
    return root.toprettyxml(indent='  ', encoding=encoding)

def _patch_minidom():
    if False:
        i = 10
        return i + 15
    minidom.Text.writexml = _writexml_text
    minidom.Element.writexml = _writexml_element
    minidom.Node.toprettyxml = _toprettyxml_node

def _collapse(node):
    if False:
        while True:
            i = 10
    for child in node.childNodes:
        if child.nodeType == Node.TEXT_NODE and len(child.data.strip()) == 0:
            child.data = ''
        else:
            _collapse(child)

def _writexml_text(self, writer, indent='', addindent='', newl=''):
    if False:
        i = 10
        return i + 15
    minidom._write_data(writer, '%s' % self.data.strip())

def _writexml_element(self, writer, indent='', addindent='', newl=''):
    if False:
        for i in range(10):
            print('nop')
    writer.write(indent + '<' + self.tagName)
    attrs = self._get_attributes()
    a_names = attrs.keys()
    a_names.sort()
    for a_name in a_names:
        writer.write(' %s="' % a_name)
        minidom._write_data(writer, attrs[a_name].value)
        writer.write('"')
    if self.childNodes:
        if self.childNodes[0].nodeType == Node.TEXT_NODE and len(self.childNodes[0].data) > 0:
            writer.write('>')
        else:
            writer.write('>%s' % newl)
        for node in self.childNodes:
            node.writexml(writer, indent + addindent, addindent, newl)
        if self.childNodes[-1].nodeType == Node.TEXT_NODE and len(self.childNodes[0].data) > 0:
            writer.write('</%s>%s' % (self.tagName, newl))
        else:
            writer.write('%s</%s>%s' % (indent, self.tagName, newl))
    else:
        writer.write('/>%s' % newl)

def _toprettyxml_node(self, indent='\t', newl='\n', encoding=None):
    if False:
        for i in range(10):
            print('nop')
    _collapse(self)
    writer = minidom._get_StringIO()
    if encoding is not None:
        import codecs
        writer = codecs.lookup(encoding)[3](writer)
    if self.nodeType == Node.DOCUMENT_NODE:
        self.writexml(writer, '', indent, newl, encoding)
    else:
        self.writexml(writer, '', indent, newl)
    return writer.getvalue()
_patch_minidom()