import json
import frappe
from frappe.core.doctype.doctype.doctype import InvalidFieldNameError
from frappe.core.doctype.doctype.test_doctype import new_doctype
from frappe.test_runner import make_test_records_for_doctype
from frappe.tests.utils import FrappeTestCase
test_dependencies = ['Custom Field', 'Property Setter']

class TestCustomizeForm(FrappeTestCase):

    def insert_custom_field(self):
        if False:
            while True:
                i = 10
        frappe.delete_doc_if_exists('Custom Field', 'Event-custom_test_field')
        self.field = frappe.get_doc({'doctype': 'Custom Field', 'fieldname': 'custom_test_field', 'dt': 'Event', 'label': 'Test Custom Field', 'description': 'A Custom Field for Testing', 'fieldtype': 'Select', 'in_list_view': 1, 'options': '\nCustom 1\nCustom 2\nCustom 3', 'default': 'Custom 3', 'insert_after': frappe.get_meta('Event').fields[-1].fieldname}).insert()

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.insert_custom_field()
        frappe.db.delete('Property Setter', dict(doc_type='Event'))
        frappe.db.commit()
        frappe.clear_cache(doctype='Event')

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        frappe.delete_doc('Custom Field', self.field.name)
        frappe.db.commit()
        frappe.clear_cache(doctype='Event')

    def get_customize_form(self, doctype=None):
        if False:
            while True:
                i = 10
        d = frappe.get_doc('Customize Form')
        if doctype:
            d.doc_type = doctype
        d.run_method('fetch_to_customize')
        return d

    def test_fetch_to_customize(self):
        if False:
            print('Hello World!')
        d = self.get_customize_form()
        self.assertEqual(d.doc_type, None)
        self.assertEqual(len(d.get('fields')), 0)
        d = self.get_customize_form('Event')
        self.assertEqual(d.doc_type, 'Event')
        self.assertEqual(len(d.get('fields')), 38)
        d = self.get_customize_form('Event')
        self.assertEqual(d.doc_type, 'Event')
        self.assertEqual(len(d.get('fields')), len(frappe.get_doc('DocType', d.doc_type).fields) + 1)
        self.assertEqual(d.get('fields')[-1].fieldname, self.field.fieldname)
        self.assertEqual(d.get('fields', {'fieldname': 'event_type'})[0].in_list_view, 1)
        return d

    def test_save_customization_property(self):
        if False:
            i = 10
            return i + 15
        d = self.get_customize_form('Event')
        self.assertEqual(frappe.db.get_value('Property Setter', {'doc_type': 'Event', 'property': 'allow_copy'}, 'value'), None)
        d.allow_copy = 1
        d.run_method('save_customization')
        self.assertEqual(frappe.db.get_value('Property Setter', {'doc_type': 'Event', 'property': 'allow_copy'}, 'value'), '1')
        d.allow_copy = 0
        d.run_method('save_customization')
        self.assertEqual(frappe.db.get_value('Property Setter', {'doc_type': 'Event', 'property': 'allow_copy'}, 'value'), None)

    def test_save_customization_field_property(self):
        if False:
            return 10
        d = self.get_customize_form('Event')
        self.assertEqual(frappe.db.get_value('Property Setter', {'doc_type': 'Event', 'property': 'reqd', 'field_name': 'repeat_this_event'}, 'value'), None)
        repeat_this_event_field = d.get('fields', {'fieldname': 'repeat_this_event'})[0]
        repeat_this_event_field.reqd = 1
        d.run_method('save_customization')
        self.assertEqual(frappe.db.get_value('Property Setter', {'doc_type': 'Event', 'property': 'reqd', 'field_name': 'repeat_this_event'}, 'value'), '1')
        repeat_this_event_field = d.get('fields', {'fieldname': 'repeat_this_event'})[0]
        repeat_this_event_field.reqd = 0
        d.run_method('save_customization')
        self.assertEqual(frappe.db.get_value('Property Setter', {'doc_type': 'Event', 'property': 'reqd', 'field_name': 'repeat_this_event'}, 'value'), None)

    def test_save_customization_custom_field_property(self):
        if False:
            return 10
        d = self.get_customize_form('Event')
        self.assertEqual(frappe.db.get_value('Custom Field', self.field.name, 'reqd'), 0)
        custom_field = d.get('fields', {'fieldname': self.field.fieldname})[0]
        custom_field.reqd = 1
        custom_field.no_copy = 1
        d.run_method('save_customization')
        self.assertEqual(frappe.db.get_value('Custom Field', self.field.name, 'reqd'), 1)
        self.assertEqual(frappe.db.get_value('Custom Field', self.field.name, 'no_copy'), 1)
        custom_field = d.get('fields', {'is_custom_field': True})[0]
        custom_field.reqd = 0
        custom_field.no_copy = 0
        d.run_method('save_customization')
        self.assertEqual(frappe.db.get_value('Custom Field', self.field.name, 'reqd'), 0)
        self.assertEqual(frappe.db.get_value('Custom Field', self.field.name, 'no_copy'), 0)

    def test_save_customization_new_field(self):
        if False:
            print('Hello World!')
        d = self.get_customize_form('Event')
        last_fieldname = d.fields[-1].fieldname
        d.append('fields', {'label': 'Test Add Custom Field Via Customize Form', 'fieldtype': 'Data', 'is_custom_field': 1})
        d.run_method('save_customization')
        custom_field_name = 'Event-custom_test_add_custom_field_via_customize_form'
        self.assertEqual(frappe.db.get_value('Custom Field', custom_field_name, 'fieldtype'), 'Data')
        self.assertEqual(frappe.db.get_value('Custom Field', custom_field_name, 'insert_after'), last_fieldname)
        frappe.delete_doc('Custom Field', custom_field_name)
        self.assertEqual(frappe.db.get_value('Custom Field', custom_field_name), None)

    def test_save_customization_remove_field(self):
        if False:
            while True:
                i = 10
        d = self.get_customize_form('Event')
        custom_field = d.get('fields', {'fieldname': self.field.fieldname})[0]
        d.get('fields').remove(custom_field)
        d.run_method('save_customization')
        self.assertEqual(frappe.db.get_value('Custom Field', custom_field.name), None)
        frappe.local.test_objects['Custom Field'] = []
        make_test_records_for_doctype('Custom Field')

    def test_reset_to_defaults(self):
        if False:
            i = 10
            return i + 15
        d = frappe.get_doc('Customize Form')
        d.doc_type = 'Event'
        d.run_method('reset_to_defaults')
        self.assertEqual(d.get('fields', {'fieldname': 'repeat_this_event'})[0].in_list_view, 0)
        frappe.local.test_objects['Property Setter'] = []
        make_test_records_for_doctype('Property Setter')

    def test_set_allow_on_submit(self):
        if False:
            print('Hello World!')
        d = self.get_customize_form('Event')
        d.get('fields', {'fieldname': 'subject'})[0].allow_on_submit = 1
        d.get('fields', {'fieldname': 'custom_test_field'})[0].allow_on_submit = 1
        d.run_method('save_customization')
        d = self.get_customize_form('Event')
        self.assertEqual(d.get('fields', {'fieldname': 'subject'})[0].allow_on_submit or 0, 0)
        self.assertEqual(d.get('fields', {'fieldname': 'custom_test_field'})[0].allow_on_submit, 1)

    def test_title_field_pattern(self):
        if False:
            while True:
                i = 10
        d = self.get_customize_form('Web Form')
        df = d.get('fields', {'fieldname': 'title'})[0]
        df.default = '{doc_type} - {introduction_test}'
        self.assertRaises(InvalidFieldNameError, d.run_method, 'save_customization')
        df.default = '{doc_type} - {introduction text}'
        self.assertRaises(InvalidFieldNameError, d.run_method, 'save_customization')
        df.default = '{doc_type} - {introduction_text}'
        d.run_method('save_customization')
        df.default = '{{ {doc_type} }} - {introduction_text}'
        d.run_method('save_customization')
        df.default = None
        d.run_method('save_customization')

    def test_core_doctype_customization(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(frappe.ValidationError, self.get_customize_form, 'User')

    def test_save_customization_length_field_property(self):
        if False:
            print('Hello World!')
        d = self.get_customize_form('Notification Log')
        document_name = d.get('fields', {'fieldname': 'document_name'})[0]
        document_name.length = 255
        d.run_method('save_customization')
        self.assertEqual(frappe.db.get_value('Property Setter', {'doc_type': 'Notification Log', 'property': 'length', 'field_name': 'document_name'}, 'value'), '255')
        self.assertTrue(d.flags.update_db)
        length = frappe.db.sql("SELECT character_maximum_length\n\t\t\tFROM information_schema.columns\n\t\t\tWHERE table_name = 'tabNotification Log'\n\t\t\tAND column_name = 'document_name'")[0][0]
        self.assertEqual(length, 255)

    def test_custom_link(self):
        if False:
            while True:
                i = 10
        try:
            testdt_name = 'Test Link for Event'
            testdt = new_doctype(testdt_name, fields=[dict(fieldtype='Link', fieldname='event', options='Event')]).insert()
            testdt_name1 = 'Test Link for Event 1'
            testdt1 = new_doctype(testdt_name1, fields=[dict(fieldtype='Link', fieldname='event', options='Event')]).insert()
            d = self.get_customize_form('Event')
            d.append('links', dict(link_doctype=testdt_name, link_fieldname='event', group='Tests'))
            d.append('links', dict(link_doctype=testdt_name1, link_fieldname='event', group='Tests'))
            d.run_method('save_customization')
            frappe.clear_cache()
            event = frappe.get_meta('Event')
            self.assertTrue([d.name for d in event.links if d.link_doctype == testdt_name])
            self.assertTrue([d.name for d in event.links if d.link_doctype == testdt_name1])
            order = json.loads(event.links_order)
            self.assertListEqual(order, [d.name for d in event.links])
            d = self.get_customize_form('Event')
            d.links = []
            d.run_method('save_customization')
            frappe.clear_cache()
            event = frappe.get_meta('Event')
            self.assertFalse([d.name for d in event.links or [] if d.link_doctype == testdt_name])
        finally:
            testdt.delete()
            testdt1.delete()

    def test_custom_internal_links(self):
        if False:
            for i in range(10):
                print('nop')
        frappe.clear_cache()
        d = self.get_customize_form('User Group')
        d.append('links', dict(link_doctype='User Group Member', parent_doctype='User Group', link_fieldname='user', table_fieldname='user_group_members', group='Tests', custom=1))
        d.run_method('save_customization')
        frappe.clear_cache()
        user_group = frappe.get_meta('User Group')
        self.assertTrue([d.name for d in user_group.links if d.link_doctype == 'User Group Member'])
        self.assertTrue([d.name for d in user_group.links if d.parent_doctype == 'User Group'])
        d = self.get_customize_form('User Group')
        d.links = []
        d.run_method('save_customization')
        frappe.clear_cache()
        user_group = frappe.get_meta('Event')
        self.assertFalse([d.name for d in user_group.links or [] if d.link_doctype == 'User Group Member'])

    def test_custom_action(self):
        if False:
            return 10
        test_route = '/app/List/DocType'
        d = self.get_customize_form('Event')
        d.append('actions', dict(label='Test Action', action_type='Route', action=test_route))
        d.run_method('save_customization')
        frappe.clear_cache()
        event = frappe.get_meta('Event')
        action = [d for d in event.actions if d.label == 'Test Action']
        self.assertEqual(len(action), 1)
        self.assertEqual(action[0].action, test_route)
        d = self.get_customize_form('Event')
        d.actions = []
        d.run_method('save_customization')
        frappe.clear_cache()
        event = frappe.get_meta('Event')
        action = [d for d in event.actions if d.label == 'Test Action']
        self.assertEqual(len(action), 0)

    def test_custom_label(self):
        if False:
            return 10
        d = self.get_customize_form('Event')
        d.label = 'Test Rename'
        d.run_method('save_customization')
        self.assertEqual(d.label, 'Test Rename')
        d.label = 'Test Rename 2'
        d.run_method('save_customization')
        self.assertEqual(d.label, 'Test Rename 2')
        d.run_method('save_customization')
        self.assertEqual(d.label, 'Test Rename 2')
        d.label = ''
        d.run_method('save_customization')
        self.assertEqual(d.label, '')

    def test_change_to_autoincrement_autoname(self):
        if False:
            for i in range(10):
                print('nop')
        d = self.get_customize_form('Event')
        d.autoname = 'autoincrement'
        with self.assertRaises(frappe.ValidationError):
            d.run_method('save_customization')

    def test_system_generated_fields(self):
        if False:
            return 10
        doctype = 'Event'
        custom_field_name = 'custom_test_field'
        custom_field = frappe.get_doc('Custom Field', {'dt': doctype, 'fieldname': custom_field_name})
        custom_field.is_system_generated = 1
        custom_field.save()
        d = self.get_customize_form(doctype)
        custom_field = d.getone('fields', {'fieldname': custom_field_name})
        custom_field.description = 'Test Description'
        d.run_method('save_customization')
        property_setter_filters = {'doc_type': doctype, 'field_name': custom_field_name, 'property': 'description'}
        self.assertEqual(frappe.db.get_value('Property Setter', property_setter_filters, 'value'), 'Test Description')

    def test_custom_field_order(self):
        if False:
            for i in range(10):
                print('nop')
        customize_form = self.get_customize_form(doctype='ToDo')
        customize_form.fields.insert(0, customize_form.fields.pop())
        customize_form.save_customization()
        field_order_property = json.loads(frappe.db.get_value('Property Setter', {'doc_type': 'ToDo', 'property': 'field_order'}, 'value'))
        self.assertEqual(field_order_property, [df.fieldname for df in frappe.get_meta('ToDo').fields])