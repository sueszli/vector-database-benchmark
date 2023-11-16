import json
from lxml import etree
from ..exceptions import except_orm
from ..models import MetaModel, BaseModel, Model, TransientModel, AbstractModel, MAGIC_COLUMNS, LOG_ACCESS_COLUMNS
from odoo.tools.safe_eval import safe_eval
browse_record_list = BaseModel

class browse_record(object):
    """ Pseudo-class for testing record instances """

    class __metaclass__(type):

        def __instancecheck__(self, inst):
            if False:
                for i in range(10):
                    print('nop')
            return isinstance(inst, BaseModel) and len(inst) <= 1

class browse_null(object):
    """ Pseudo-class for testing null instances """

    class __metaclass__(type):

        def __instancecheck__(self, inst):
            if False:
                for i in range(10):
                    print('nop')
            return isinstance(inst, BaseModel) and (not inst)

def transfer_field_to_modifiers(field, modifiers):
    if False:
        print('Hello World!')
    default_values = {}
    state_exceptions = {}
    for attr in ('invisible', 'readonly', 'required'):
        state_exceptions[attr] = []
        default_values[attr] = bool(field.get(attr))
    for (state, modifs) in field.get('states', {}).items():
        for modif in modifs:
            if default_values[modif[0]] != modif[1]:
                state_exceptions[modif[0]].append(state)
    for (attr, default_value) in default_values.items():
        if state_exceptions[attr]:
            modifiers[attr] = [('state', 'not in' if default_value else 'in', state_exceptions[attr])]
        else:
            modifiers[attr] = default_value

def transfer_node_to_modifiers(node, modifiers, context=None, in_tree_view=False):
    if False:
        while True:
            i = 10
    if node.get('attrs'):
        modifiers.update(safe_eval(node.get('attrs')))
    if node.get('states'):
        if 'invisible' in modifiers and isinstance(modifiers['invisible'], list):
            modifiers['invisible'].append(('state', 'not in', node.get('states').split(',')))
        else:
            modifiers['invisible'] = [('state', 'not in', node.get('states').split(','))]
    for a in ('invisible', 'readonly', 'required'):
        if node.get(a):
            v = bool(safe_eval(node.get(a), {'context': context or {}}))
            if in_tree_view and a == 'invisible':
                modifiers['tree_invisible'] = v
            elif v or (a not in modifiers or not isinstance(modifiers[a], list)):
                modifiers[a] = v

def simplify_modifiers(modifiers):
    if False:
        while True:
            i = 10
    for a in ('invisible', 'readonly', 'required'):
        if a in modifiers and (not modifiers[a]):
            del modifiers[a]

def transfer_modifiers_to_node(modifiers, node):
    if False:
        while True:
            i = 10
    if modifiers:
        simplify_modifiers(modifiers)
        node.set('modifiers', json.dumps(modifiers))

def setup_modifiers(node, field=None, context=None, in_tree_view=False):
    if False:
        print('Hello World!')
    ' Processes node attributes and field descriptors to generate\n    the ``modifiers`` node attribute and set it on the provided node.\n\n    Alters its first argument in-place.\n\n    :param node: ``field`` node from an OpenERP view\n    :type node: lxml.etree._Element\n    :param dict field: field descriptor corresponding to the provided node\n    :param dict context: execution context used to evaluate node attributes\n    :param bool in_tree_view: triggers the ``tree_invisible`` code\n                              path (separate from ``invisible``): in\n                              tree view there are two levels of\n                              invisibility, cell content (a column is\n                              present but the cell itself is not\n                              displayed) with ``invisible`` and column\n                              invisibility (the whole column is\n                              hidden) with ``tree_invisible``.\n    :returns: nothing\n    '
    modifiers = {}
    if field is not None:
        transfer_field_to_modifiers(field, modifiers)
    transfer_node_to_modifiers(node, modifiers, context=context, in_tree_view=in_tree_view)
    transfer_modifiers_to_node(modifiers, node)

def test_modifiers(what, expected):
    if False:
        print('Hello World!')
    modifiers = {}
    if isinstance(what, basestring):
        node = etree.fromstring(what)
        transfer_node_to_modifiers(node, modifiers)
        simplify_modifiers(modifiers)
        dump = json.dumps(modifiers)
        assert dump == expected, '%s != %s' % (dump, expected)
    elif isinstance(what, dict):
        transfer_field_to_modifiers(what, modifiers)
        simplify_modifiers(modifiers)
        dump = json.dumps(modifiers)
        assert dump == expected, '%s != %s' % (dump, expected)

def modifiers_tests():
    if False:
        while True:
            i = 10
    test_modifiers('<field name="a"/>', '{}')
    test_modifiers('<field name="a" invisible="1"/>', '{"invisible": true}')
    test_modifiers('<field name="a" readonly="1"/>', '{"readonly": true}')
    test_modifiers('<field name="a" required="1"/>', '{"required": true}')
    test_modifiers('<field name="a" invisible="0"/>', '{}')
    test_modifiers('<field name="a" readonly="0"/>', '{}')
    test_modifiers('<field name="a" required="0"/>', '{}')
    test_modifiers('<field name="a" invisible="1" required="1"/>', '{"invisible": true, "required": true}')
    test_modifiers('<field name="a" invisible="1" required="0"/>', '{"invisible": true}')
    test_modifiers('<field name="a" invisible="0" required="1"/>', '{"required": true}')
    test_modifiers('<field name="a" attrs="{\'invisible\': [(\'b\', \'=\', \'c\')]}"/>', '{"invisible": [["b", "=", "c"]]}')
    test_modifiers({}, '{}')
    test_modifiers({'invisible': True}, '{"invisible": true}')
    test_modifiers({'invisible': False}, '{}')