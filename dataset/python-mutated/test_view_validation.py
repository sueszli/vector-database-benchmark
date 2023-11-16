from lxml import etree
from StringIO import StringIO
import unittest
from odoo.tools.view_validation import valid_page_in_book, valid_att_in_form, valid_type_in_colspan, valid_type_in_col, valid_att_in_field, valid_att_in_label, valid_field_in_graph, valid_field_in_tree
invalid_form = etree.parse(StringIO('<form>\n    <label></label>\n    <group>\n        <div>\n            <page></page>\n            <label colspan="True"></label>\n            <field></field>\n        </div>\n    </group>\n    <notebook>\n        <page>\n            <group col="Two">\n            <div>\n                <label></label>\n                <field colspan="Five"> </field>\n                </div>\n            </group>\n        </page>\n    </notebook>\n</form>\n')).getroot()
valid_form = etree.parse(StringIO('<form string="">\n    <field name=""></field>\n    <field name=""></field>\n    <notebook>\n        <page>\n            <field name=""></field>\n            <label string=""></label>\n            <field name=""></field>\n        </page>\n        <page>\n            <group colspan="5" col="2">\n                <label for=""></label>\n                <label string="" colspan="5"></label>\n            </group>\n        </page>\n    </notebook>\n</form>\n')).getroot()
invalid_graph = etree.parse(StringIO('<graph>\n    <label/>\n    <group>\n        <div>\n            <field></field>\n            <field></field>\n        </div>\n    </group>\n</graph>\n')).getroot()
valid_graph = etree.parse(StringIO('<graph string="">\n    <field name=""></field>\n    <field name=""></field>\n</graph>\n')).getroot()
invalid_tree = etree.parse(StringIO('<tree>\n  <group>\n    <div>\n      <field></field>\n      <field></field>\n    </div>\n  </group>\n</tree>\n')).getroot()
valid_tree = etree.parse(StringIO('<tree string="">\n    <field name=""></field>\n    <field name=""></field>\n    <button/>\n    <field name=""></field>\n</tree>\n')).getroot()

class TestViewValidation(unittest.TestCase):
    """ Test the view validation code (but not the views themselves). """

    def test_page_validation(self):
        if False:
            i = 10
            return i + 15
        assert not valid_page_in_book(invalid_form)
        assert valid_page_in_book(valid_form)

    def test_all_field_validation(self):
        if False:
            for i in range(10):
                print('nop')
        assert not valid_att_in_field(invalid_form)
        assert valid_att_in_field(valid_form)

    def test_all_label_validation(self):
        if False:
            print('Hello World!')
        assert not valid_att_in_label(invalid_form)
        assert valid_att_in_label(valid_form)

    def test_form_string_validation(self):
        if False:
            for i in range(10):
                print('nop')
        assert valid_att_in_form(valid_form)

    def test_graph_validation(self):
        if False:
            while True:
                i = 10
        assert not valid_field_in_graph(invalid_graph)
        assert valid_field_in_graph(valid_graph)

    def test_tree_validation(self):
        if False:
            print('Hello World!')
        assert not valid_field_in_tree(invalid_tree)
        assert valid_field_in_tree(valid_tree)

    def test_colspan_datatype_validation(self):
        if False:
            i = 10
            return i + 15
        assert not valid_type_in_colspan(invalid_form)
        assert valid_type_in_colspan(valid_form)

    def test_col_datatype_validation(self):
        if False:
            return 10
        assert not valid_type_in_col(invalid_form)
        assert valid_type_in_col(valid_form)