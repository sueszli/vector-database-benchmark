"""Module containing operations for reading and writing dialogs as XML"""
from __future__ import print_function
from __future__ import unicode_literals
from xml.etree.cElementTree import Element
from xml.etree.cElementTree import SubElement
from xml.etree.cElementTree import ElementTree
import six
import ctypes
import re
import bz2, base64
try:
    import PIL.Image
    PIL_imported = True
except ImportError:
    PIL_imported = False
from . import controls
from .windows.win32structures import LOGFONTW, RECT

class XMLParsingError(RuntimeError):
    """Wrap parsing Exceptions"""
    pass

def _set_node_props(element, name, value):
    if False:
        while True:
            i = 10
    'Set the properties of the node based on the type of object'
    if isinstance(value, ctypes.Structure):
        struct_elem = SubElement(element, name)
        cls_name = value.__class__.__name__
        struct_elem.set('__type__', '{0}'.format(cls_name))
        for prop_name in value._fields_:
            prop_name = prop_name[0]
            item_val = getattr(value, prop_name)
            if isinstance(item_val, six.integer_types):
                prop_name += '_LONG'
                item_val = six.text_type(item_val)
            struct_elem.set(prop_name, _escape_specials(item_val))
    elif hasattr(value, 'tobytes') and hasattr(value, 'size'):
        try:
            if value.size[0] * value.size[1] > 5000 * 5000:
                raise MemoryError
            image_data = base64.encodestring(bz2.compress(value.tobytes())).decode('utf-8')
            _set_node_props(element, name + '_IMG', {'mode': value.mode, 'size_x': value.size[0], 'size_y': value.size[1], 'data': image_data})
        except (SystemError, MemoryError):
            pass
    elif isinstance(value, (list, tuple)):
        listelem = SubElement(element, name + '_LIST')
        for (i, attrval) in enumerate(value):
            _set_node_props(listelem, '%s_%05d' % (name, i), attrval)
    elif isinstance(value, dict):
        dict_elem = SubElement(element, name)
        for (item_name, val) in value.items():
            _set_node_props(dict_elem, item_name, val)
    else:
        if isinstance(value, bool):
            value = six.integer_types[-1](value)
        if isinstance(value, six.integer_types):
            name += '_LONG'
        element.set(name, _escape_specials(value))

def WriteDialogToFile(filename, props):
    if False:
        for i in range(10):
            print('nop')
    '\n    Write the props to the file\n\n    props can be either a dialog or a dictionary\n    '
    try:
        props[0].keys()
    except (TypeError, AttributeError):
        props = controls.get_dialog_props_from_handle(props)
    root = Element('DIALOG')
    root.set('_version_', '2.0')
    for ctrl in props:
        ctrlelem = SubElement(root, 'CONTROL')
        for (name, value) in sorted(ctrl.items()):
            _set_node_props(ctrlelem, name, value)
    tree = ElementTree(root)
    tree.write(filename, encoding='utf-8')

def _escape_specials(string):
    if False:
        while True:
            i = 10
    'Ensure that some characters are escaped before writing to XML'
    string = six.text_type(string)
    string = string.replace('\\', '\\\\')
    for i in range(0, 32):
        string = string.replace(six.unichr(i), '\\%02d' % i)
    return string

def _un_escape_specials(string):
    if False:
        for i in range(10):
            print('nop')
    'Replace escaped characters with real character'
    for i in range(0, 32):
        string = string.replace('\\%02d' % i, six.unichr(i))
    string = string.replace('\\\\', '\\')
    return six.text_type(string)

def _xml_to_struct(element, struct_type=None):
    if False:
        print('Hello World!')
    "\n    Convert an ElementTree to a ctypes Struct\n\n    If struct_type is not specified then element['__type__']\n    will be used for the ctypes struct type\n    "
    try:
        attribs = element.attrib
    except AttributeError:
        attribs = element
    if not struct_type:
        struct = globals()[attribs['__type__']]()
    else:
        struct = globals()[struct_type]()
    struct_attribs = dict(((at.upper(), at) for at in dir(struct)))
    for prop_name in attribs:
        val = attribs[prop_name]
        if prop_name.endswith('_LONG'):
            val = six.integer_types[-1](val)
            prop_name = prop_name[:-5]
        elif isinstance(val, six.string_types):
            val = six.text_type(val)
        if prop_name.upper() in struct_attribs:
            prop_name = struct_attribs[prop_name.upper()]
            setattr(struct, prop_name, val)
    return struct

def _old_xml_to_titles(element):
    if False:
        i = 10
        return i + 15
    'For OLD XML files convert the titles as a list'
    title_names = element.keys()
    title_names.sort()
    titles = []
    for name in title_names:
        val = element[name]
        val = val.replace('\\n', '\n')
        val = val.replace('\\x12', '\x12')
        val = val.replace('\\\\', '\\')
        titles.append(six.text_type(val))
    return titles

def _extract_properties(properties, prop_name, prop_value):
    if False:
        return 10
    "\n    Hmmm - confusing - can't remember exactly how\n    all these similar functions call each other\n    "
    (prop_name, reqd_index) = _split_number(prop_name)
    if reqd_index is None:
        if prop_name in properties:
            try:
                properties[prop_name].append(prop_value)
            except AttributeError:
                new_val = [properties[prop_name], prop_value]
                properties[prop_name] = new_val
        else:
            properties[prop_name] = prop_value
    else:
        properties.setdefault(prop_name, [])
        while 1:
            if len(properties[prop_name]) <= reqd_index:
                properties[prop_name].append('')
            else:
                break
        properties[prop_name][reqd_index] = prop_value

def _get_attributes(element):
    if False:
        for i in range(10):
            print('nop')
    'Get the attributes from an element'
    properties = {}
    for (attrib_name, val) in element.attrib.items():
        if attrib_name.endswith('_LONG'):
            val = six.integer_types[-1](val)
            attrib_name = attrib_name[:-5]
        else:
            val = _un_escape_specials(val)
        _extract_properties(properties, attrib_name, val)
    return properties
number = re.compile('^(.*)_(\\d{5})$')

def _split_number(prop_name):
    if False:
        while True:
            i = 10
    '\n    Return (string, number) for a prop_name in the format string_number\n\n    The number part has to be 5 digits long\n    None is returned if there is no _number part\n\n    e.g.\n    >>> _split_number("NoNumber")\n    (\'NoNumber\', None)\n    >>> _split_number("Anumber_00003")\n    (\'Anumber\', 3)\n    >>> _split_number("notEnoughDigits_0003")\n    (\'notEnoughDigits_0003\', None)\n    '
    found = number.search(prop_name)
    if not found:
        return (prop_name, None)
    return (found.group(1), int(found.group(2)))

def _read_xml_structure(control_element):
    if False:
        i = 10
        return i + 15
    '\n    Convert an element into nested Python objects\n\n    The values will be returned in a dictionary as following:\n\n     - the attributes will be items of the dictionary\n       for each subelement\n\n       + if it has a __type__ attribute then it is converted to a\n         ctypes structure\n       + if the element tag ends with _IMG then it is converted to\n         a PIL image\n\n     - If there are elements with the same name or attributes with\n       ordering e.g. texts_00001, texts_00002 they will be put into a\n       list (in the correct order)\n    '
    properties = _get_attributes(control_element)
    for elem in control_element:
        if '__type__' in elem.attrib:
            propval = _xml_to_struct(elem)
        elif elem.tag.endswith('_IMG'):
            elem.tag = elem.tag[:-4]
            img = _get_attributes(elem)
            data = bz2.decompress(base64.decodestring(img['data'].encode('utf-8')))
            if PIL_imported is False:
                raise RuntimeError('PIL is not installed!')
            propval = PIL.Image.frombytes(img['mode'], (img['size_x'], img['size_y']), data)
        elif elem.tag.endswith('_LIST'):
            elem.tag = elem.tag[:-5]
            propval = _read_xml_structure(elem)
            if propval == {}:
                propval = list()
            else:
                propval = propval[elem.tag]
        else:
            propval = _read_xml_structure(elem)
        _extract_properties(properties, elem.tag, propval)
    return properties

def ReadPropertiesFromFile(filename):
    if False:
        i = 10
        return i + 15
    'Return a list of controls from XML file filename'
    parsed = ElementTree().parse(filename)
    props = _read_xml_structure(parsed)['CONTROL']
    if not isinstance(props, list):
        props = [props]
    if not '_version_' in parsed.attrib.keys():
        for ctrl_prop in props:
            ctrl_prop['fonts'] = [_xml_to_struct(ctrl_prop['FONT'], 'LOGFONTW')]
            ctrl_prop['rectangle'] = _xml_to_struct(ctrl_prop['RECTANGLE'], 'RECT')
            ctrl_prop['client_rects'] = [_xml_to_struct(ctrl_prop['CLIENTRECT'], 'RECT')]
            ctrl_prop['texts'] = _old_xml_to_titles(ctrl_prop['TITLES'])
            ctrl_prop['class_name'] = ctrl_prop['CLASS']
            ctrl_prop['context_help_id'] = ctrl_prop['HELPID']
            ctrl_prop['control_id'] = ctrl_prop['CTRLID']
            ctrl_prop['exstyle'] = ctrl_prop['EXSTYLE']
            ctrl_prop['friendly_class_name'] = ctrl_prop['FRIENDLYCLASS']
            ctrl_prop['is_unicode'] = ctrl_prop['ISUNICODE']
            ctrl_prop['is_visible'] = ctrl_prop['ISVISIBLE']
            ctrl_prop['style'] = ctrl_prop['STYLE']
            ctrl_prop['user_data'] = ctrl_prop['USERDATA']
            for prop_name in ['CLASS', 'CLIENTRECT', 'CTRLID', 'EXSTYLE', 'FONT', 'FRIENDLYCLASS', 'HELPID', 'ISUNICODE', 'ISVISIBLE', 'RECTANGLE', 'STYLE', 'TITLES', 'USERDATA']:
                del ctrl_prop[prop_name]
    return props