""" XML node tree handling

Means to create XML elements from Nuitka tree nodes and to convert the
XML tree to ASCII or output it.
"""
from nuitka.__past__ import StringIO

def _indent(elem, level=0, more_sibs=False):
    if False:
        while True:
            i = 10
    i = '\n'
    if level:
        i += (level - 1) * '  '
    num_kids = len(elem)
    if num_kids:
        if not elem.text or not elem.text.strip():
            elem.text = i + '  '
            if level:
                elem.text += '  '
        count = 0
        for kid in elem:
            _indent(kid, level + 1, count < num_kids - 1)
            count += 1
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
            if more_sibs:
                elem.tail += '  '
    elif level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i
        if more_sibs:
            elem.tail += '  '
    return elem

def _dedent(elem, level=0):
    if False:
        while True:
            i = 10
    if not elem.text or not elem.text.strip():
        elem.text = ''
    for child in elem:
        _dedent(child, level + 1)
    if not elem.tail or not elem.tail.strip():
        elem.tail = ''
    return elem
try:
    import xml.etree.ElementTree
    xml_module = xml.etree.ElementTree
    Element = xml.etree.ElementTree.Element

    def xml_tostring(tree, indent=True):
        if False:
            for i in range(10):
                print('nop')
        if indent:
            _indent(tree)
        elif not indent:
            _dedent(tree)
        return xml_module.tostring(tree)
except ImportError:
    xml_module = None
    Element = None
    xml_tostring = None

def toBytes(tree, indent=True):
    if False:
        while True:
            i = 10
    return xml_tostring(tree, indent=indent)

def toString(tree):
    if False:
        print('Hello World!')
    result = toBytes(tree)
    if str is not bytes:
        result = result.decode('utf8')
    return result

def fromString(text):
    if False:
        for i in range(10):
            print('nop')
    return fromFile(StringIO(text))

def fromFile(file_handle, use_lxml=False):
    if False:
        for i in range(10):
            print('nop')
    if use_lxml:
        from lxml import etree
        return etree.parse(file_handle).getroot()
    else:
        return xml_module.parse(file_handle).getroot()

def appendTreeElement(parent, *args, **kwargs):
    if False:
        return 10
    element = Element(*args, **kwargs)
    parent.append(element)
    return element

def dumpTreeXMLToFile(tree, output_file):
    if False:
        for i in range(10):
            print('nop')
    'Write an XML node tree to a file.'
    value = toString(tree).rstrip()
    output_file.write(value)