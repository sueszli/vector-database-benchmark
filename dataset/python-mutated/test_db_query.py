import datetime
from contextlib import contextmanager
from unittest.mock import MagicMock, patch
import frappe
from frappe.core.doctype.doctype.test_doctype import new_doctype
from frappe.core.page.permission_manager.permission_manager import add, reset, update
from frappe.custom.doctype.property_setter.property_setter import make_property_setter
from frappe.database.utils import DefaultOrderBy
from frappe.desk.reportview import get_filters_cond
from frappe.handler import execute_cmd
from frappe.model.db_query import DatabaseQuery, get_between_date_filter
from frappe.permissions import add_user_permission, clear_user_permissions_for_doctype
from frappe.query_builder import Column
from frappe.tests.utils import FrappeTestCase
from frappe.utils.testutils import add_custom_field, clear_custom_fields
test_dependencies = ['User', 'Blog Post', 'Blog Category', 'Blogger']

@contextmanager
def setup_test_user(set_user=False):
    if False:
        print('Hello World!')
    test_user = frappe.get_doc('User', 'test@example.com')
    user_roles = frappe.get_roles()
    test_user.remove_roles(*user_roles)
    test_user.add_roles('Blogger')
    if set_user:
        frappe.set_user(test_user.name)
    yield test_user
    test_user.remove_roles('Blogger')
    test_user.add_roles(*user_roles)

@contextmanager
def setup_patched_blog_post():
    if False:
        for i in range(10):
            print('nop')
    add_child_table_to_blog_post()
    make_property_setter('Blog Post', 'published', 'permlevel', 1, 'Int')
    reset('Blog Post')
    add('Blog Post', 'Website Manager', 1)
    update('Blog Post', 'Website Manager', 1, 'write', 1)
    yield

class TestDBQuery(FrappeTestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        frappe.set_user('Administrator')

    def test_basic(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue({'name': 'DocType'} in DatabaseQuery('DocType').execute(limit_page_length=None))

    def test_extract_tables(self):
        if False:
            print('Hello World!')
        db_query = DatabaseQuery('DocType')
        add_custom_field('DocType', 'test_tab_field', 'Data')
        db_query.fields = ['tabNote.creation', 'test_tab_field', 'tabDocType.test_tab_field']
        db_query.extract_tables()
        self.assertIn('`tabNote`', db_query.tables)
        self.assertIn('`tabDocType`', db_query.tables)
        self.assertNotIn('test_tab_field', db_query.tables)
        clear_custom_fields('DocType')

    def test_child_table_field_syntax(self):
        if False:
            return 10
        note = frappe.get_doc(doctype='Note', title=f'Test {frappe.utils.random_string(8)}', content='test', seen_by=[{'user': 'Administrator'}]).insert()
        result = frappe.get_all('Note', filters={'name': note.name}, fields=['name', 'seen_by.user as seen_by'], limit=1)
        self.assertEqual(result[0].seen_by, 'Administrator')
        note.delete()

    def test_child_table_join(self):
        if False:
            i = 10
            return i + 15
        frappe.delete_doc_if_exists('DocType', 'Parent DocType 1')
        frappe.delete_doc_if_exists('DocType', 'Parent DocType 2')
        frappe.delete_doc_if_exists('DocType', 'Child DocType')
        frappe.get_doc({'doctype': 'DocType', 'name': 'Child DocType', 'module': 'Custom', 'custom': 1, 'istable': 1, 'fields': [{'label': 'Title', 'fieldname': 'title', 'fieldtype': 'Data'}]}).insert()
        frappe.get_doc({'doctype': 'DocType', 'name': 'Parent DocType 1', 'module': 'Custom', 'custom': 1, 'fields': [{'label': 'Title', 'fieldname': 'title', 'fieldtype': 'Data'}, {'label': 'Table Field 1', 'fieldname': 'child', 'fieldtype': 'Table', 'options': 'Child DocType'}], 'permissions': [{'role': 'System Manager'}]}).insert()
        frappe.get_doc({'doctype': 'DocType', 'name': 'Parent DocType 2', 'module': 'Custom', 'custom': 1, 'fields': [{'label': 'Title', 'fieldname': 'title', 'fieldtype': 'Data'}, {'label': 'Table Field 1', 'fieldname': 'child', 'fieldtype': 'Table', 'options': 'Child DocType'}], 'permissions': [{'role': 'System Manager'}]}).insert()
        frappe.db.delete('Parent DocType 1')
        frappe.db.delete('Parent DocType 2')
        frappe.db.delete('Child DocType')
        frappe.get_doc(doctype='Parent DocType 1', title='test', child=[{'title': 'parent 1 child record 1'}, {'title': 'parent 1 child record 2'}], __newname='test_parent').insert(ignore_if_duplicate=True)
        frappe.get_doc(doctype='Parent DocType 2', title='test', child=[{'title': 'parent 2 child record 1'}], __newname='test_parent').insert(ignore_if_duplicate=True)
        results1 = frappe.get_all('Parent DocType 1', fields=['name', 'child.title as child_title'])
        results2 = frappe.get_all('Parent DocType 2', fields=['name', 'child.title as child_title'])
        self.assertEqual(results1[0].name, results2[0].name)
        self.assertEqual(len(results1), 2)
        self.assertEqual(len(results2), 1)
        parent1_children = [result.child_title for result in results1]
        self.assertIn('parent 1 child record 1', parent1_children)
        self.assertIn('parent 1 child record 2', parent1_children)
        self.assertEqual(results2[0].child_title, 'parent 2 child record 1')

    def test_link_field_syntax(self):
        if False:
            return 10
        todo = frappe.get_doc(doctype='ToDo', description='Test ToDo', allocated_to='Administrator').insert()
        result = frappe.get_all('ToDo', filters={'name': todo.name}, fields=['name', 'allocated_to.email as allocated_user_email'], limit=1)
        self.assertEqual(result[0].allocated_user_email, 'admin@example.com')
        todo.delete()

    def test_build_match_conditions(self):
        if False:
            return 10
        clear_user_permissions_for_doctype('Blog Post', 'test2@example.com')
        test2user = frappe.get_doc('User', 'test2@example.com')
        test2user.add_roles('Blogger')
        frappe.set_user('test2@example.com')
        build_match_conditions = DatabaseQuery('Blog Post').build_match_conditions
        self.assertEqual(build_match_conditions(as_condition=False), [])
        self.assertEqual(build_match_conditions(as_condition=True), '')
        add_user_permission('Blog Post', '-test-blog-post', 'test2@example.com', True)
        add_user_permission('Blog Post', '-test-blog-post-1', 'test2@example.com', True)
        self.assertTrue({'Blog Post': ['-test-blog-post-1', '-test-blog-post']} in build_match_conditions(as_condition=False))
        if frappe.db.db_type == 'mariadb':
            assertion_string = "(((ifnull(`tabBlog Post`.`name`, '')='' or `tabBlog Post`.`name` in ('-test-blog-post-1', '-test-blog-post'))))"
        else:
            assertion_string = "(((ifnull(cast(`tabBlog Post`.`name` as varchar), '')='' or cast(`tabBlog Post`.`name` as varchar) in ('-test-blog-post-1', '-test-blog-post'))))"
        self.assertEqual(build_match_conditions(as_condition=True), assertion_string)
        frappe.set_user('Administrator')

    def test_fields(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue({'name': 'DocType', 'issingle': 0} in DatabaseQuery('DocType').execute(fields=['name', 'issingle'], limit_page_length=None))

    def test_filters_1(self):
        if False:
            return 10
        self.assertFalse({'name': 'DocType'} in DatabaseQuery('DocType').execute(filters=[['DocType', 'name', 'like', 'J%']]))

    def test_filters_2(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertFalse({'name': 'DocType'} in DatabaseQuery('DocType').execute(filters=[{'name': ['like', 'J%']}]))

    def test_filters_3(self):
        if False:
            i = 10
            return i + 15
        self.assertFalse({'name': 'DocType'} in DatabaseQuery('DocType').execute(filters={'name': ['like', 'J%']}))

    def test_filters_4(self):
        if False:
            return 10
        self.assertTrue({'name': 'DocField'} in DatabaseQuery('DocType').execute(filters={'name': 'DocField'}))

    def test_in_not_in_filters(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertFalse(DatabaseQuery('DocType').execute(filters={'name': ['in', None]}))
        self.assertTrue({'name': 'DocType'} in DatabaseQuery('DocType').execute(filters={'name': ['not in', None]}))
        for result in [{'name': 'DocType'}, {'name': 'DocField'}]:
            self.assertTrue(result in DatabaseQuery('DocType').execute(filters={'name': ['in', 'DocType,DocField']}))
        for result in [{'name': 'DocType'}, {'name': 'DocField'}]:
            self.assertFalse(result in DatabaseQuery('DocType').execute(filters={'name': ['not in', 'DocType,DocField']}))

    def test_none_filter(self):
        if False:
            return 10
        query = frappe.qb.get_query('DocType', fields='name', filters={'restrict_to_domain': None})
        sql = str(query).replace('`', '').replace('"', '')
        condition = 'restrict_to_domain IS NULL'
        self.assertIn(condition, sql)

    def test_or_filters(self):
        if False:
            for i in range(10):
                print('nop')
        data = DatabaseQuery('DocField').execute(filters={'parent': 'DocType'}, fields=['fieldname', 'fieldtype'], or_filters=[{'fieldtype': 'Table'}, {'fieldtype': 'Select'}])
        self.assertTrue({'fieldtype': 'Table', 'fieldname': 'fields'} in data)
        self.assertTrue({'fieldtype': 'Select', 'fieldname': 'document_type'} in data)
        self.assertFalse({'fieldtype': 'Check', 'fieldname': 'issingle'} in data)

    def test_between_filters(self):
        if False:
            while True:
                i = 10
        'test case to check between filter for date fields'
        frappe.db.delete('Event')
        todays_event = create_event()
        event1 = create_event(starts_on='2016-07-05 23:59:59')
        event2 = create_event(starts_on='2016-07-06 00:00:00')
        event3 = create_event(starts_on='2016-07-07 23:59:59')
        event4 = create_event(starts_on='2016-07-08 00:00:00')
        data = DatabaseQuery('Event').execute(filters={'starts_on': ['between', None]}, fields=['name'])
        self.assertTrue({'name': event1.name} not in data)
        data = DatabaseQuery('Event').execute(filters={'starts_on': ['between', ['2016-07-06', '2016-07-07']]}, fields=['name'])
        self.assertIn({'name': event2.name}, data)
        self.assertIn({'name': event3.name}, data)
        self.assertNotIn({'name': event1.name}, data)
        self.assertNotIn({'name': event4.name}, data)
        data = DatabaseQuery('Event').execute(filters={'starts_on': ['between', ['2016-07-07']]}, fields=['name'])
        self.assertIn({'name': event3.name}, data)
        self.assertIn({'name': event4.name}, data)
        self.assertIn({'name': todays_event.name}, data)
        self.assertNotIn({'name': event1.name}, data)
        self.assertNotIn({'name': event2.name}, data)
        data = DatabaseQuery('Event').execute(filters={'creation': ['between', ['2016-07-06', '2016-07-07']]}, fields=['name'])

    def test_between_filters_date_bounds(self):
        if False:
            while True:
                i = 10
        date_df = frappe._dict(fieldtype='Date')
        datetime_df = frappe._dict(fieldtype='Datetime')
        today = frappe.utils.nowdate()
        cond = get_between_date_filter('', date_df)
        self.assertQueryEqual(cond, f"'{today}' AND '{today}'")
        start = '2021-01-01'
        cond = get_between_date_filter([start], date_df)
        self.assertQueryEqual(cond, f"'{start}' AND '{today}'")
        start = '2021-01-01'
        end = '2022-01-02'
        cond = get_between_date_filter([start, end], date_df)
        self.assertQueryEqual(cond, f"'{start}' AND '{end}'")
        start = '2021-01-01'
        cond = get_between_date_filter([start, start], datetime_df)
        self.assertQueryEqual(cond, f"'{start} 00:00:00.000000' AND '{start} 23:59:59.999999'")
        start = '2021-01-01 01:01:00'
        end = '2022-01-02 12:23:43'
        cond = get_between_date_filter([start, end], datetime_df)
        self.assertQueryEqual(cond, f"'{start}.000000' AND '{end}.000000'")

    def test_ignore_permissions_for_get_filters_cond(self):
        if False:
            i = 10
            return i + 15
        frappe.set_user('test2@example.com')
        self.assertRaises(frappe.PermissionError, get_filters_cond, 'DocType', dict(istable=1), [])
        self.assertTrue(get_filters_cond('DocType', dict(istable=1), [], ignore_permissions=True))
        frappe.set_user('Administrator')

    def test_query_fields_sanitizer(self):
        if False:
            print('Hello World!')
        self.assertRaises(frappe.DataError, DatabaseQuery('DocType').execute, fields=['name', 'issingle, version()'], limit_start=0, limit_page_length=1)
        self.assertRaises(frappe.DataError, DatabaseQuery('DocType').execute, fields=['name', 'issingle, IF(issingle=1, (select name from tabUser), count(name))'], limit_start=0, limit_page_length=1)
        self.assertRaises(frappe.DataError, DatabaseQuery('DocType').execute, fields=['name', 'issingle, (select count(*) from tabSessions)'], limit_start=0, limit_page_length=1)
        self.assertRaises(frappe.DataError, DatabaseQuery('DocType').execute, fields=['name', "issingle, SELECT LOCATE('', `tabUser`.`user`) AS user;"], limit_start=0, limit_page_length=1)
        self.assertRaises(frappe.DataError, DatabaseQuery('DocType').execute, fields=['name', 'issingle, IF(issingle=1, (SELECT name from tabUser), count(*))'], limit_start=0, limit_page_length=1)
        self.assertRaises(frappe.DataError, DatabaseQuery('DocType').execute, fields=['name', "issingle ''"], limit_start=0, limit_page_length=1)
        self.assertRaises(frappe.DataError, DatabaseQuery('DocType').execute, fields=['name', "issingle,'"], limit_start=0, limit_page_length=1)
        self.assertRaises(frappe.DataError, DatabaseQuery('DocType').execute, fields=['name', 'select * from tabSessions'], limit_start=0, limit_page_length=1)
        self.assertRaises(frappe.DataError, DatabaseQuery('DocType').execute, fields=['name', 'issingle from --'], limit_start=0, limit_page_length=1)
        self.assertRaises(frappe.DataError, DatabaseQuery('DocType').execute, fields=['name', 'issingle from tabDocType order by 2 --'], limit_start=0, limit_page_length=1)
        self.assertRaises(frappe.DataError, DatabaseQuery('DocType').execute, fields=['name', "1' UNION SELECT * FROM __Auth --"], limit_start=0, limit_page_length=1)
        self.assertRaises(frappe.DataError, DatabaseQuery('DocType').execute, fields=['@@version'], limit_start=0, limit_page_length=1)
        data = DatabaseQuery('DocType').execute(fields=['count(`name`) as count'], limit_start=0, limit_page_length=1)
        self.assertTrue('count' in data[0])
        data = DatabaseQuery('DocType').execute(fields=['name', 'issingle', "locate('', name) as _relevance"], limit_start=0, limit_page_length=1)
        self.assertTrue('_relevance' in data[0])
        data = DatabaseQuery('DocType').execute(fields=['name', 'issingle', 'date(creation) as creation'], limit_start=0, limit_page_length=1)
        self.assertTrue('creation' in data[0])
        if frappe.db.db_type != 'postgres':
            data = DatabaseQuery('DocType').execute(fields=['name', 'issingle', 'datediff(modified, creation) as date_diff'], limit_start=0, limit_page_length=1)
            self.assertTrue('date_diff' in data[0])
        with self.assertRaises(frappe.DataError):
            DatabaseQuery('DocType').execute(fields=['name', 'issingle', 'if (issingle=1, (select name from tabUser), count(name))'], limit_start=0, limit_page_length=1)
        with self.assertRaises(frappe.DataError):
            DatabaseQuery('DocType').execute(fields=['name', 'issingle', 'if(issingle=1, (select name from tabUser), count(name))'], limit_start=0, limit_page_length=1)
        with self.assertRaises(frappe.DataError):
            DatabaseQuery('DocType').execute(fields=['name', 'issingle', '( select name from `tabUser` where `tabDocType`.owner = `tabUser`.name )'], limit_start=0, limit_page_length=1, ignore_permissions=True)
        with self.assertRaises(frappe.DataError):
            DatabaseQuery('DocType').execute(fields=['name', 'issingle', '(select name from `tabUser` where `tabDocType`.owner = `tabUser`.name )'], limit_start=0, limit_page_length=1)

    def test_nested_permission(self):
        if False:
            return 10
        frappe.set_user('Administrator')
        create_nested_doctype()
        create_nested_doctype_records()
        clear_user_permissions_for_doctype('Nested DocType')
        add_user_permission('Nested DocType', 'Level 1 A', 'test2@example.com')
        from frappe.core.page.permission_manager.permission_manager import update
        update('Nested DocType', 'All', 0, 'if_owner', 0)
        frappe.set_user('test2@example.com')
        data = DatabaseQuery('Nested DocType').execute()
        self.assertTrue({'name': 'Level 2 A'} in data)
        self.assertTrue({'name': 'Level 2 A'} in data)
        self.assertFalse({'name': 'Level 1 B'} in data)
        self.assertFalse({'name': 'Level 2 B'} in data)
        update('Nested DocType', 'All', 0, 'if_owner', 1)
        frappe.set_user('Administrator')

    def test_filter_sanitizer(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(frappe.DataError, DatabaseQuery('DocType').execute, fields=['name'], filters={'istable,': 1}, limit_start=0, limit_page_length=1)
        self.assertRaises(frappe.DataError, DatabaseQuery('DocType').execute, fields=['name'], filters={'editable_grid,': 1}, or_filters={'istable,': 1}, limit_start=0, limit_page_length=1)
        self.assertRaises(frappe.DataError, DatabaseQuery('DocType').execute, fields=['name'], filters={'editable_grid,': 1}, or_filters=[['DocType', 'istable,', '=', 1]], limit_start=0, limit_page_length=1)
        self.assertRaises(frappe.DataError, DatabaseQuery('DocType').execute, fields=['name'], filters={'editable_grid,': 1}, or_filters=[['DocType', 'istable', '=', 1], ['DocType', 'beta and 1=1', '=', 0]], limit_start=0, limit_page_length=1)
        out = DatabaseQuery('DocType').execute(fields=['name'], filters={'editable_grid': 1, 'module': 'Core'}, or_filters=[['DocType', 'istable', '=', 1]], order_by='creation')
        self.assertTrue('DocField' in [d['name'] for d in out])
        out = DatabaseQuery('DocType').execute(fields=['name'], filters={'issingle': 1}, or_filters=[['DocType', 'module', '=', 'Core']], order_by='creation')
        self.assertTrue('Role Permission for Page and Report' in [d['name'] for d in out])
        out = DatabaseQuery('DocType').execute(fields=['name'], filters={'track_changes': 1, 'module': 'Core'}, order_by='creation')
        self.assertTrue('File' in [d['name'] for d in out])
        out = DatabaseQuery('DocType').execute(fields=['name'], filters=[['DocType', 'ifnull(track_changes, 0)', '=', 0], ['DocType', 'module', '=', 'Core']], order_by='creation')
        self.assertTrue('DefaultValue' in [d['name'] for d in out])

    def test_order_by_group_by_sanitizer(self):
        if False:
            print('Hello World!')
        with self.assertRaises(frappe.ValidationError):
            DatabaseQuery('DocType').execute(fields=['name'], order_by='sleep (1) asc')
        with self.assertRaises(frappe.ValidationError):
            DatabaseQuery('DocType').execute(fields=['name'], group_by='SLEEP(0)')
        with self.assertRaises(frappe.ValidationError):
            DatabaseQuery('DocType').execute(fields=['name'], order_by='(select rank from tabRankedDocTypes where tabRankedDocTypes.name = tabDocType.name) asc')
        DatabaseQuery('DocType').execute(fields=['name'], order_by='name asc')
        DatabaseQuery('DocType').execute(fields=['name'], order_by='name asc', group_by='name')
        if frappe.db.db_type == 'mariadb':
            DatabaseQuery('DocType').execute(fields=['name'], order_by='timestamp(modified)')

    def test_of_not_of_descendant_ancestors(self):
        if False:
            while True:
                i = 10
        frappe.set_user('Administrator')
        clear_user_permissions_for_doctype('Nested DocType')
        data = frappe.get_all('Nested DocType', {'name': ('descendants of', 'Level 2 A')})
        self.assertTrue({'name': 'Level 3 A'} in data)
        data = frappe.get_all('Nested DocType', {'name': ('descendants of', 'Level 1 A')})
        self.assertTrue({'name': 'Level 3 A'} in data)
        self.assertTrue({'name': 'Level 2 A'} in data)
        self.assertFalse({'name': 'Level 2 B'} in data)
        self.assertFalse({'name': 'Level 1 B'} in data)
        self.assertFalse({'name': 'Level 1 A'} in data)
        self.assertFalse({'name': 'Root'} in data)
        data = frappe.get_all('Nested DocType', {'name': ('ancestors of', 'Level 2 A')})
        self.assertFalse({'name': 'Level 3 A'} in data)
        self.assertFalse({'name': 'Level 2 A'} in data)
        self.assertFalse({'name': 'Level 2 B'} in data)
        self.assertFalse({'name': 'Level 1 B'} in data)
        self.assertTrue({'name': 'Level 1 A'} in data)
        self.assertTrue({'name': 'Root'} in data)
        data = frappe.get_all('Nested DocType', {'name': ('ancestors of', 'Level 1 A')})
        self.assertFalse({'name': 'Level 3 A'} in data)
        self.assertFalse({'name': 'Level 2 A'} in data)
        self.assertFalse({'name': 'Level 2 B'} in data)
        self.assertFalse({'name': 'Level 1 B'} in data)
        self.assertFalse({'name': 'Level 1 A'} in data)
        self.assertTrue({'name': 'Root'} in data)
        data = frappe.get_all('Nested DocType', {'name': ('not descendants of', 'Level 2 A')})
        self.assertFalse({'name': 'Level 3 A'} in data)
        self.assertTrue({'name': 'Level 2 A'} in data)
        self.assertTrue({'name': 'Level 2 B'} in data)
        self.assertTrue({'name': 'Level 1 A'} in data)
        self.assertTrue({'name': 'Root'} in data)
        data = frappe.get_all('Nested DocType', {'name': ('not descendants of', 'Level 1 A')})
        self.assertFalse({'name': 'Level 3 A'} in data)
        self.assertFalse({'name': 'Level 2 A'} in data)
        self.assertTrue({'name': 'Level 2 B'} in data)
        self.assertTrue({'name': 'Level 1 B'} in data)
        self.assertTrue({'name': 'Level 1 A'} in data)
        self.assertTrue({'name': 'Root'} in data)
        data = frappe.get_all('Nested DocType', {'name': ('not ancestors of', 'Level 2 A')})
        self.assertTrue({'name': 'Level 3 A'} in data)
        self.assertTrue({'name': 'Level 2 A'} in data)
        self.assertTrue({'name': 'Level 2 B'} in data)
        self.assertTrue({'name': 'Level 1 B'} in data)
        self.assertTrue({'name': 'Level 1 A'} not in data)
        self.assertTrue({'name': 'Root'} not in data)
        data = frappe.get_all('Nested DocType', {'name': ('not ancestors of', 'Level 1 A')})
        self.assertTrue({'name': 'Level 3 A'} in data)
        self.assertTrue({'name': 'Level 2 A'} in data)
        self.assertTrue({'name': 'Level 2 B'} in data)
        self.assertTrue({'name': 'Level 1 B'} in data)
        self.assertTrue({'name': 'Level 1 A'} in data)
        self.assertFalse({'name': 'Root'} in data)
        data = frappe.get_all('Nested DocType', {'name': ('ancestors of', 'Root')})
        self.assertTrue(len(data) == 0)
        self.assertTrue(len(frappe.get_all('Nested DocType', {'name': ('not ancestors of', 'Root')})) == len(frappe.get_all('Nested DocType')))

    def test_is_set_is_not_set(self):
        if False:
            print('Hello World!')
        res = DatabaseQuery('DocType').execute(filters={'autoname': ['is', 'not set']})
        self.assertTrue({'name': 'Integration Request'} in res)
        self.assertTrue({'name': 'User'} in res)
        self.assertFalse({'name': 'Blogger'} in res)
        res = DatabaseQuery('DocType').execute(filters={'autoname': ['is', 'set']})
        self.assertTrue({'name': 'DocField'} in res)
        self.assertTrue({'name': 'Prepared Report'} in res)
        self.assertFalse({'name': 'Property Setter'} in res)
        frappe.db.set_value('DocType', 'Property Setter', 'autoname', None, update_modified=False)
        res = DatabaseQuery('DocType').execute(filters={'autoname': ['is', 'set']})
        self.assertFalse({'name': 'Property Setter'} in res)

    def test_set_field_tables(self):
        if False:
            print('Hello World!')
        data = frappe.db.get_list('Web Form', filters=[['Web Form Field', 'reqd', '=', 1]], fields=['count(*) as count'], order_by='count desc', limit=50)

    def test_pluck_name(self):
        if False:
            i = 10
            return i + 15
        names = DatabaseQuery('DocType').execute(filters={'name': 'DocType'}, pluck='name')
        self.assertEqual(names, ['DocType'])

    def test_pluck_any_field(self):
        if False:
            while True:
                i = 10
        owners = DatabaseQuery('DocType').execute(filters={'name': 'DocType'}, pluck='owner')
        self.assertEqual(owners, ['Administrator'])

    def test_prepare_select_args(self):
        if False:
            i = 10
            return i + 15
        doctypes = frappe.get_all('DocType', filters={'docstatus': 0, 'document_type': ('!=', '')}, group_by='document_type', fields=['document_type', 'sum(is_submittable) as is_submittable'], limit=1, as_list=True)
        if frappe.conf.db_type == 'mariadb':
            self.assertTrue(len(doctypes[0]) == 2)
        else:
            self.assertTrue(len(doctypes[0]) == 3)
            self.assertTrue(isinstance(doctypes[0][2], datetime.datetime))

    def test_column_comparison(self):
        if False:
            for i in range(10):
                print('nop')
        'Test DatabaseQuery.execute to test column comparison'
        users_unedited = frappe.get_all('User', filters={'creation': Column('modified')}, fields=['name', 'creation', 'modified'], limit=1)
        users_edited = frappe.get_all('User', filters={'creation': ('!=', Column('modified'))}, fields=['name', 'creation', 'modified'], limit=1)
        self.assertEqual(users_unedited[0].modified, users_unedited[0].creation)
        self.assertNotEqual(users_edited[0].modified, users_edited[0].creation)

    def test_permlevel_fields(self):
        if False:
            for i in range(10):
                print('nop')
        with setup_patched_blog_post(), setup_test_user(set_user=True):
            data = frappe.get_list('Blog Post', filters={'published': 1}, fields=['name', 'published'], limit=1)
            self.assertFalse('published' in data[0])
            self.assertTrue('name' in data[0])
            self.assertEqual(len(data[0]), 1)
            data = frappe.get_list('Blog Post', filters={'published': 1}, fields=['name', '`published`'], limit=1)
            self.assertFalse('published' in data[0])
            self.assertTrue('name' in data[0])
            self.assertEqual(len(data[0]), 1)
            data = frappe.get_list('Blog Post', filters={'published': 1}, fields=['name', '`tabBlog Post`.`published`'], limit=1)
            self.assertFalse('published' in data[0])
            self.assertTrue('name' in data[0])
            self.assertEqual(len(data[0]), 1)
            data = frappe.get_list('Blog Post', filters={'published': 1}, fields=['name', '`tabTest Child`.`test_field`'], limit=1)
            self.assertFalse('test_field' in data[0])
            self.assertTrue('name' in data[0])
            self.assertEqual(len(data[0]), 1)
            data = frappe.get_list('Blog Post', filters={'published': 1}, fields=['name', 'MAX(`published`)'], limit=1)
            self.assertTrue('name' in data[0])
            self.assertEqual(len(data[0]), 1)
            data = frappe.get_list('Blog Post', filters={'published': 1}, fields=['name', 'LAST(published)'], limit=1)
            self.assertTrue('name' in data[0])
            self.assertEqual(len(data[0]), 1)
            data = frappe.get_list('Blog Post', filters={'published': 1}, fields=['name', 'MAX(`modified`)'], limit=1, order_by=None, group_by='name')
            self.assertEqual(len(data[0]), 2)
            data = frappe.get_list('Blog Post', filters={'published': 1}, fields=['name', 'now() abhi'], limit=1)
            self.assertIsInstance(data[0]['abhi'], datetime.datetime)
            self.assertEqual(len(data[0]), 2)
            data = frappe.get_list('Blog Post', filters={'published': 1}, fields=['name', "'LABEL'"], limit=1)
            self.assertTrue('name' in data[0])
            self.assertTrue('LABEL' in data[0].values())
            self.assertEqual(len(data[0]), 2)
            data = frappe.get_list('Blog Post', filters={'published': 1}, fields=['name', 'COUNT(*) as count'], limit=1, order_by=None, group_by='name')
            self.assertTrue('count' in data[0])
            self.assertEqual(len(data[0]), 2)
            data = frappe.get_list('Blog Post', filters={'published': 1}, fields=['name', 'COUNT(*) count'], limit=1, order_by=None, group_by='name')
            self.assertTrue('count' in data[0])
            self.assertEqual(len(data[0]), 2)
            data = frappe.get_list('Blog Post', fields=['name', 'blogger.full_name as blogger_full_name', 'blog_category.description'], limit=1)
            self.assertTrue('name' in data[0])
            self.assertTrue('blogger_full_name' in data[0])
            self.assertTrue('description' in data[0])

    def test_cast_name(self):
        if False:
            print('Hello World!')
        from frappe.core.doctype.doctype.test_doctype import new_doctype
        frappe.delete_doc_if_exists('DocType', 'autoinc_dt_test')
        dt = new_doctype('autoinc_dt_test', autoname='autoincrement').insert(ignore_permissions=True)
        query = DatabaseQuery('autoinc_dt_test').execute(fields=["locate('1', `tabautoinc_dt_test`.`name`)", 'name', "locate('1', name)"], filters={'name': 1}, run=False)
        if frappe.db.db_type == 'postgres':
            self.assertTrue('strpos( cast("tabautoinc_dt_test"."name" as varchar), \'1\')' in query)
            self.assertTrue("strpos( cast(name as varchar), '1')" in query)
            self.assertTrue('where cast("tabautoinc_dt_test"."name" as varchar) = \'1\'' in query)
        else:
            self.assertTrue("locate('1', `tabautoinc_dt_test`.`name`)" in query)
            self.assertTrue("locate('1', name)" in query)
            self.assertTrue('where `tabautoinc_dt_test`.`name` = 1' in query)
        dt.delete(ignore_permissions=True)

    def test_fieldname_starting_with_int(self):
        if False:
            i = 10
            return i + 15
        from frappe.core.doctype.doctype.test_doctype import new_doctype
        frappe.delete_doc_if_exists('DocType', 'dt_with_int_named_fieldname')
        frappe.delete_doc_if_exists('DocType', 'table_dt')
        table_dt = new_doctype('table_dt', istable=1, fields=[{'label': '1field', 'fieldname': '2field', 'fieldtype': 'Data'}]).insert()
        dt = new_doctype('dt_with_int_named_fieldname', fields=[{'label': '1field', 'fieldname': '1field', 'fieldtype': 'Data'}, {'label': '2table_field', 'fieldname': '2table_field', 'fieldtype': 'Table', 'options': table_dt.name}]).insert(ignore_permissions=True)
        dt_data = frappe.get_doc({'doctype': 'dt_with_int_named_fieldname', '1field': '10'}).insert(ignore_permissions=True)
        query = DatabaseQuery('dt_with_int_named_fieldname')
        self.assertTrue(query.execute(filters={'1field': '10'}))
        self.assertTrue(query.execute(filters={'1field': ['like', '1%']}))
        self.assertTrue(query.execute(filters={'1field': ['in', '1,2,10']}))
        self.assertTrue(query.execute(filters={'1field': ['is', 'set']}))
        self.assertFalse(query.execute(filters={'1field': ['not like', '1%']}))
        self.assertTrue(query.execute(filters=[['table_dt', '2field', 'is', 'not set']]))
        frappe.get_doc({'doctype': table_dt.name, '2field': '10', 'parent': dt_data.name, 'parenttype': dt_data.doctype, 'parentfield': '2table_field'}).insert(ignore_permissions=True)
        self.assertTrue(query.execute(filters=[['table_dt', '2field', 'is', 'set']]))
        dt.delete()
        table_dt.delete()

    def test_permission_query_condition(self):
        if False:
            for i in range(10):
                print('nop')
        from frappe.desk.doctype.dashboard_settings.dashboard_settings import create_dashboard_settings
        self.doctype = 'Dashboard Settings'
        self.user = "test'5@example.com"
        permission_query_conditions = DatabaseQuery.get_permission_query_conditions(self)
        create_dashboard_settings(self.user)
        dashboard_settings = frappe.db.sql('\n\t\t\t\tSELECT name\n\t\t\t\tFROM `tabDashboard Settings`\n\t\t\t\tWHERE {condition}\n\t\t\t'.format(condition=permission_query_conditions), as_dict=1)[0]
        self.assertTrue(dashboard_settings)

    def test_virtual_doctype(self):
        if False:
            return 10
        'Test that virtual doctypes can be queried using get_all'
        virtual_doctype = new_doctype('Virtual DocType')
        virtual_doctype.is_virtual = 1
        virtual_doctype.insert(ignore_if_duplicate=True)

        class VirtualDocType:

            @staticmethod
            def get_list(args):
                if False:
                    for i in range(10):
                        print('nop')
                ...
        with patch('frappe.controllers', new={frappe.local.site: {'Virtual DocType': VirtualDocType}}):
            VirtualDocType.get_list = MagicMock()
            frappe.get_all('Virtual DocType', filters={'name': 'test'}, fields=['name'], limit=1)
            call_args = VirtualDocType.get_list.call_args[0][0]
            VirtualDocType.get_list.assert_called_once()
            self.assertIsInstance(call_args, dict)
            self.assertEqual(call_args['doctype'], 'Virtual DocType')
            self.assertEqual(call_args['filters'], [['Virtual DocType', 'name', '=', 'test']])
            self.assertEqual(call_args['fields'], ['name'])
            self.assertEqual(call_args['limit_page_length'], 1)
            self.assertEqual(call_args['limit_start'], 0)
            self.assertEqual(call_args['order_by'], DefaultOrderBy)

    def test_coalesce_with_in_ops(self):
        if False:
            i = 10
            return i + 15
        self.assertNotIn('ifnull', frappe.get_all('User', {'name': ('in', ['a', 'b'])}, run=0))
        self.assertIn('ifnull', frappe.get_all('User', {'name': ('in', ['a', None])}, run=0))
        self.assertIn('ifnull', frappe.get_all('User', {'name': ('in', ['a', ''])}, run=0))
        self.assertIn('ifnull', frappe.get_all('User', {'name': ('in', [])}, run=0))
        self.assertIn('ifnull', frappe.get_all('User', {'name': ('not in', ['a'])}, run=0))
        self.assertIn('ifnull', frappe.get_all('User', {'name': ('not in', [])}, run=0))
        self.assertIn('ifnull', frappe.get_all('User', {'name': ('not in', [''])}, run=0))

    def test_ambiguous_linked_tables(self):
        if False:
            print('Hello World!')
        from frappe.desk.reportview import get
        if not frappe.db.exists('DocType', 'Related Todos'):
            frappe.get_doc({'doctype': 'DocType', 'custom': 1, 'module': 'Custom', 'name': 'Related Todos', 'naming_rule': 'Random', 'autoname': 'hash', 'fields': [{'label': 'Todo One', 'fieldname': 'todo_one', 'fieldtype': 'Link', 'options': 'ToDo', 'reqd': 1}, {'label': 'Todo Two', 'fieldname': 'todo_two', 'fieldtype': 'Link', 'options': 'ToDo', 'reqd': 1}]}).insert()
        else:
            frappe.db.delete('Related Todos')
        todo_one = frappe.get_doc({'doctype': 'ToDo', 'description': 'Todo One'}).insert()
        todo_two = frappe.get_doc({'doctype': 'ToDo', 'description': 'Todo Two'}).insert()
        frappe.get_doc({'doctype': 'Related Todos', 'todo_one': todo_one.name, 'todo_two': todo_two.name}).insert()
        frappe.form_dict.doctype = 'Related Todos'
        frappe.form_dict.fields = ['`tabRelated Todos`.`name`', '`tabRelated Todos`.`todo_one`', '`tabRelated Todos`.`todo_two`', 'todo_one.description as todo_one_description', 'todo_two.description as todo_two_description']
        data = get()
        self.assertEqual(len(data['values']), 1)

class TestReportView(FrappeTestCase):

    def test_get_count(self):
        if False:
            while True:
                i = 10
        frappe.local.request = frappe._dict()
        frappe.local.request.method = 'GET'
        frappe.local.form_dict = frappe._dict({'doctype': 'DocType', 'filters': [['DocType', 'show_title_field_in_link', '=', 1]], 'fields': [], 'distinct': 'false'})
        list_filter_response = execute_cmd('frappe.desk.reportview.get_count')
        frappe.local.form_dict = frappe._dict({'doctype': 'DocType', 'filters': {'show_title_field_in_link': 1}, 'distinct': 'true'})
        dict_filter_response = execute_cmd('frappe.desk.reportview.get_count')
        self.assertIsInstance(list_filter_response, int)
        self.assertEqual(list_filter_response, dict_filter_response)
        frappe.local.form_dict = frappe._dict({'doctype': 'DocType', 'filters': [['DocField', 'fieldtype', '=', 'Data']], 'fields': [], 'distinct': 'true'})
        child_filter_response = execute_cmd('frappe.desk.reportview.get_count')
        current_value = frappe.db.sql("select distinct count(distinct `tabDocType`.name) as total_count from `tabDocType` left join `tabDocField` on (`tabDocField`.parenttype = 'DocType' and `tabDocField`.parent = `tabDocType`.name) where `tabDocField`.`fieldtype` = 'Data'")[0][0]
        self.assertEqual(child_filter_response, current_value)

    def test_reportview_get(self):
        if False:
            while True:
                i = 10
        user = frappe.get_doc('User', 'test@example.com')
        add_child_table_to_blog_post()
        user_roles = frappe.get_roles()
        user.remove_roles(*user_roles)
        user.add_roles('Blogger')
        make_property_setter('Blog Post', 'published', 'permlevel', 1, 'Int')
        reset('Blog Post')
        add('Blog Post', 'Website Manager', 1)
        update('Blog Post', 'Website Manager', 1, 'write', 1)
        frappe.set_user(user.name)
        frappe.local.request = frappe._dict()
        frappe.local.request.method = 'POST'
        frappe.local.form_dict = frappe._dict({'doctype': 'Blog Post', 'fields': ['published', 'title', '`tabTest Child`.`test_field`']})
        response = execute_cmd('frappe.desk.reportview.get')
        self.assertListEqual(response['keys'], ['title'])
        frappe.local.form_dict = frappe._dict({'doctype': 'Blog Post', 'fields': ['*']})
        response = execute_cmd('frappe.desk.reportview.get')
        self.assertNotIn('published', response['keys'])
        frappe.set_user('Administrator')
        user.add_roles('Website Manager')
        frappe.set_user(user.name)
        frappe.set_user('Administrator')
        frappe.local.form_dict = frappe._dict({'doctype': 'Blog Post', 'fields': ['published', 'title', '`tabTest Child`.`test_field`']})
        response = execute_cmd('frappe.desk.reportview.get')
        self.assertListEqual(response['keys'], ['published', 'title', 'test_field'])
        user.remove_roles('Blogger', 'Website Manager')
        user.add_roles(*user_roles)

    def test_reportview_get_aggregation(self):
        if False:
            for i in range(10):
                print('nop')
        frappe.local.request = frappe._dict()
        frappe.local.request.method = 'POST'
        frappe.local.form_dict = frappe._dict({'doctype': 'DocType', 'fields': '["`tabDocField`.`label` as field_label","`tabDocField`.`name` as field_name"]', 'filters': '[]', 'order_by': '_aggregate_column desc', 'start': 0, 'page_length': 20, 'view': 'Report', 'with_comment_count': 0, 'group_by': 'field_label, field_name', 'aggregate_on_field': 'columns', 'aggregate_on_doctype': 'DocField', 'aggregate_function': 'sum'})
        response = execute_cmd('frappe.desk.reportview.get')
        self.assertListEqual(response['keys'], ['field_label', 'field_name', '_aggregate_column'])

    def test_reportview_get_permlevel_system_users(self):
        if False:
            for i in range(10):
                print('nop')
        with setup_patched_blog_post(), setup_test_user(set_user=True):
            frappe.local.request = frappe._dict()
            frappe.local.request.method = 'POST'
            frappe.local.form_dict = frappe._dict({'doctype': 'Blog Post', 'fields': ['published', 'title', '`tabTest Child`.`test_field`']})
            response = execute_cmd('frappe.desk.reportview.get')
            self.assertListEqual(response['keys'], ['title'])
            frappe.local.form_dict = frappe._dict({'doctype': 'Blog Post', 'fields': ['*']})
            response = execute_cmd('frappe.desk.reportview.get')
            self.assertNotIn('published', response['keys'])

    def test_reportview_get_admin(self):
        if False:
            print('Hello World!')
        with setup_patched_blog_post():
            frappe.local.request = frappe._dict()
            frappe.local.request.method = 'POST'
            frappe.local.form_dict = frappe._dict({'doctype': 'Blog Post', 'fields': ['published', 'title', '`tabTest Child`.`test_field`']})
            response = execute_cmd('frappe.desk.reportview.get')
            self.assertListEqual(response['keys'], ['published', 'title', 'test_field'])

def add_child_table_to_blog_post():
    if False:
        i = 10
        return i + 15
    child_table = frappe.get_doc({'doctype': 'DocType', 'istable': 1, 'custom': 1, 'name': 'Test Child', 'module': 'Custom', 'autoname': 'Prompt', 'fields': [{'fieldname': 'test_field', 'fieldtype': 'Data', 'permlevel': 1}]})
    child_table.insert(ignore_permissions=True, ignore_if_duplicate=True)
    clear_custom_fields('Blog Post')
    add_custom_field('Blog Post', 'child_table', 'Table', child_table.name)

def create_event(subject='_Test Event', starts_on=None):
    if False:
        print('Hello World!')
    'create a test event'
    from frappe.utils import get_datetime
    return frappe.get_doc({'doctype': 'Event', 'subject': subject, 'event_type': 'Public', 'starts_on': get_datetime(starts_on)}).insert(ignore_permissions=True)

def create_nested_doctype():
    if False:
        for i in range(10):
            print('nop')
    if frappe.db.exists('DocType', 'Nested DocType'):
        return
    frappe.get_doc({'doctype': 'DocType', 'name': 'Nested DocType', 'module': 'Custom', 'is_tree': 1, 'custom': 1, 'autoname': 'Prompt', 'fields': [{'label': 'Description', 'fieldname': 'description'}], 'permissions': [{'role': 'Blogger'}]}).insert()

def create_nested_doctype_records():
    if False:
        i = 10
        return i + 15
    '\n\tCreate a structure like:\n\t- Root\n\t        - Level 1 A\n\t                - Level 2 A\n\t                        - Level 3 A\n\t        - Level 1 B\n\t                - Level 2 B\n\t'
    records = [{'name': 'Root', 'is_group': 1}, {'name': 'Level 1 A', 'parent_nested_doctype': 'Root', 'is_group': 1}, {'name': 'Level 2 A', 'parent_nested_doctype': 'Level 1 A', 'is_group': 1}, {'name': 'Level 3 A', 'parent_nested_doctype': 'Level 2 A'}, {'name': 'Level 1 B', 'parent_nested_doctype': 'Root', 'is_group': 1}, {'name': 'Level 2 B', 'parent_nested_doctype': 'Level 1 B'}]
    for r in records:
        d = frappe.new_doc('Nested DocType')
        d.update(r)
        d.insert(ignore_permissions=True, ignore_if_duplicate=True)