from lxml import etree
import odoo
import odoo.tools as tools
from . import print_fnc
from odoo.models import BaseModel
from odoo.tools.safe_eval import safe_eval

class InheritDict(dict):

    def __init__(self, parent=None):
        if False:
            return 10
        self.parent = parent

    def __getitem__(self, name):
        if False:
            while True:
                i = 10
        if name in self:
            return super(InheritDict, self).__getitem__(name)
        elif not self.parent:
            raise KeyError
        else:
            return self.parent[name]

def tounicode(val):
    if False:
        while True:
            i = 10
    if isinstance(val, str):
        unicode_val = unicode(val, 'utf-8')
    elif isinstance(val, unicode):
        unicode_val = val
    else:
        unicode_val = unicode(val)
    return unicode_val

class document(object):

    def __init__(self, cr, uid, datas, func=False):
        if False:
            i = 10
            return i + 15
        self.cr = cr
        self.uid = uid
        self.datas = datas
        self.func = func or {}
        self.bin_datas = {}

    def node_attrs_get(self, node):
        if False:
            for i in range(10):
                print('nop')
        if len(node.attrib):
            return node.attrib
        return {}

    def get_value(self, browser, field_path):
        if False:
            print('Hello World!')
        fields = field_path.split('.')
        if not len(fields):
            return ''
        value = browser
        for f in fields:
            if isinstance(value, (BaseModel, list)):
                if not value:
                    return ''
                value = value[0]
            value = value[f]
        return value or ''

    def get_value2(self, browser, field_path):
        if False:
            i = 10
            return i + 15
        value = self.get_value(browser, field_path)
        if isinstance(value, BaseModel):
            return value.id
        else:
            return value

    def eval(self, record, expr):
        if False:
            return 10
        return safe_eval(expr, {}, {'obj': record})

    def parse_node(self, node, parent, browser, datas=None):
        if False:
            while True:
                i = 10
        env = odoo.api.Environment(self.cr, self.uid, {})
        attrs = self.node_attrs_get(node)
        if 'type' in attrs:
            if attrs['type'] == 'field':
                value = self.get_value(browser, attrs['name'])
                if value == '' and 'default' in attrs:
                    value = attrs['default']
                el = etree.SubElement(parent, node.tag)
                el.text = tounicode(value)
                for (key, value) in attrs.iteritems():
                    if key not in ('type', 'name', 'default'):
                        el.set(key, value)
            elif attrs['type'] == 'attachment':
                model = browser._name
                value = self.get_value(browser, attrs['name'])
                atts = env['ir.attachment'].search([('res_model', '=', model), ('res_id', '=', int(value))])
                datas = atts.read()
                if len(datas):
                    datas = datas[0]
                    fname = str(datas['datas_fname'])
                    ext = fname.split('.')[-1].lower()
                    if ext in ('jpg', 'jpeg', 'png'):
                        import base64
                        from StringIO import StringIO
                        dt = base64.decodestring(datas['datas'])
                        fp = StringIO()
                        fp.write(dt)
                        i = str(len(self.bin_datas))
                        self.bin_datas[i] = fp
                        el = etree.SubElement(parent, node.tag)
                        el.text = i
            elif attrs['type'] == 'data':
                txt = self.datas.get('form', {}).get(attrs['name'], '')
                el = etree.SubElement(parent, node.tag)
                el.text = txt
            elif attrs['type'] == 'function':
                if attrs['name'] in self.func:
                    txt = self.func[attrs['name']](node)
                else:
                    txt = print_fnc.print_fnc(attrs['name'], node)
                el = etree.SubElement(parent, node.tag)
                el.text = txt
            elif attrs['type'] == 'eval':
                value = self.eval(browser, attrs['expr'])
                el = etree.SubElement(parent, node.tag)
                el.text = str(value)
            elif attrs['type'] == 'fields':
                fields = attrs['name'].split(',')
                vals = {}
                for b in browser:
                    value = tuple([self.get_value2(b, f) for f in fields])
                    if not value in vals:
                        vals[value] = []
                    vals[value].append(b)
                keys = vals.keys()
                keys.sort()
                if 'order' in attrs and attrs['order'] == 'desc':
                    keys.reverse()
                v_list = [vals[k] for k in keys]
                for v in v_list:
                    el = etree.SubElement(parent, node.tag)
                    for el_cld in node:
                        self.parse_node(el_cld, el, v)
            elif attrs['type'] == 'call':
                if len(attrs['args']):
                    args = [self.eval(browser, arg) for arg in attrs['args'].split(',')]
                else:
                    args = []
                if 'model' in attrs:
                    obj = env[attrs['model']]
                else:
                    obj = browser
                if 'ids' in attrs:
                    ids = self.eval(browser, attrs['ids'])
                else:
                    ids = browser.ids
                newdatas = getattr(obj, attrs['name'])(*args)

                def parse_result_tree(node, parent, datas):
                    if False:
                        i = 10
                        return i + 15
                    if not node.tag == etree.Comment:
                        el = etree.SubElement(parent, node.tag)
                        atr = self.node_attrs_get(node)
                        if 'value' in atr:
                            if not isinstance(datas[atr['value']], (str, unicode)):
                                txt = str(datas[atr['value']])
                            else:
                                txt = datas[atr['value']]
                            el.text = txt
                        else:
                            for el_cld in node:
                                parse_result_tree(el_cld, el, datas)
                if not isinstance(newdatas, (BaseModel, list)):
                    newdatas = [newdatas]
                for newdata in newdatas:
                    parse_result_tree(node, parent, newdata)
            elif attrs['type'] == 'zoom':
                value = self.get_value(browser, attrs['name'])
                if value:
                    if not isinstance(value, (BaseModel, list)):
                        v_list = [value]
                    else:
                        v_list = value
                    for v in v_list:
                        el = etree.SubElement(parent, node.tag)
                        for el_cld in node:
                            self.parse_node(el_cld, el, v)
        elif not node.tag == etree.Comment:
            if node.tag == parent.tag:
                el = parent
            else:
                el = etree.SubElement(parent, node.tag)
            for el_cld in node:
                self.parse_node(el_cld, el, browser)

    def xml_get(self):
        if False:
            i = 10
            return i + 15
        return etree.tostring(self.doc, encoding='utf-8', xml_declaration=True, pretty_print=True)

    def parse_tree(self, ids, model, context=None):
        if False:
            print('Hello World!')
        env = odoo.api.Environment(self.cr, self.uid, context or {})
        browser = env[model].browse(ids)
        self.parse_node(self.dom, self.doc, browser)

    def parse_string(self, xml, ids, model, context=None):
        if False:
            print('Hello World!')
        self.dom = etree.XML(xml)
        self.parse_tree(ids, model, context)

    def parse(self, filename, ids, model, context=None):
        if False:
            for i in range(10):
                print('nop')
        src_file = tools.file_open(filename)
        try:
            self.dom = etree.XML(src_file.read())
            self.doc = etree.Element(self.dom.tag)
            self.parse_tree(ids, model, context)
        finally:
            src_file.close()

    def close(self):
        if False:
            print('Hello World!')
        self.doc = None
        self.dom = None