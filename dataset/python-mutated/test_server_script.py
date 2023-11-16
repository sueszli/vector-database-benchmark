import requests
import frappe
from frappe.core.doctype.scheduled_job_type.scheduled_job_type import sync_jobs
from frappe.frappeclient import FrappeClient, FrappeException
from frappe.tests.utils import FrappeTestCase
from frappe.utils import get_site_url
scripts = [dict(name='test_todo', script_type='DocType Event', doctype_event='Before Insert', reference_doctype='ToDo', script='\nif "test" in doc.description:\n\tdoc.status = \'Closed\'\n'), dict(name='test_todo_validate', script_type='DocType Event', doctype_event='Before Insert', reference_doctype='ToDo', script='\nif "validate" in doc.description:\n\traise frappe.ValidationError\n'), dict(name='test_api', script_type='API', api_method='test_server_script', allow_guest=1, script="\nfrappe.response['message'] = 'hello'\n"), dict(name='test_return_value', script_type='API', api_method='test_return_value', allow_guest=1, script="\nfrappe.flags = 'hello'\n"), dict(name='test_permission_query', script_type='Permission Query', reference_doctype='ToDo', script="\nconditions = '1 = 1'\n"), dict(name='test_invalid_namespace_method', script_type='DocType Event', doctype_event='Before Insert', reference_doctype='Note', script='\nfrappe.method_that_doesnt_exist("do some magic")\n'), dict(name='test_todo_commit', script_type='DocType Event', doctype_event='Before Save', reference_doctype='ToDo', disabled=1, script='\nfrappe.db.commit()\n'), dict(name='test_add_index', script_type='DocType Event', doctype_event='Before Save', reference_doctype='ToDo', disabled=1, script='\nfrappe.db.add_index("Todo", ["color", "date"])\n')]

class TestServerScript(FrappeTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        super().setUpClass()
        frappe.db.truncate('Server Script')
        frappe.get_doc('User', 'Administrator').add_roles('Script Manager')
        for script in scripts:
            script_doc = frappe.get_doc(doctype='Server Script')
            script_doc.update(script)
            script_doc.insert()
        cls.enable_safe_exec()
        frappe.db.commit()
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        if False:
            print('Hello World!')
        frappe.db.commit()
        frappe.db.truncate('Server Script')
        frappe.cache.delete_value('server_script_map')

    def setUp(self):
        if False:
            while True:
                i = 10
        frappe.cache.delete_value('server_script_map')

    def test_doctype_event(self):
        if False:
            while True:
                i = 10
        todo = frappe.get_doc(dict(doctype='ToDo', description='hello')).insert()
        self.assertEqual(todo.status, 'Open')
        todo = frappe.get_doc(dict(doctype='ToDo', description='test todo')).insert()
        self.assertEqual(todo.status, 'Closed')
        self.assertRaises(frappe.ValidationError, frappe.get_doc(dict(doctype='ToDo', description='validate me')).insert)

    def test_api(self):
        if False:
            return 10
        response = requests.post(get_site_url(frappe.local.site) + '/api/method/test_server_script')
        self.assertEqual(response.status_code, 200)
        self.assertEqual('hello', response.json()['message'])

    def test_api_return(self):
        if False:
            while True:
                i = 10
        self.assertEqual(frappe.get_doc('Server Script', 'test_return_value').execute_method(), 'hello')

    def test_permission_query(self):
        if False:
            print('Hello World!')
        if frappe.conf.db_type == 'mariadb':
            self.assertTrue('where (1 = 1)' in frappe.db.get_list('ToDo', run=False))
        else:
            self.assertTrue("where (1 = '1')" in frappe.db.get_list('ToDo', run=False))
        self.assertTrue(isinstance(frappe.db.get_list('ToDo'), list))

    def test_attribute_error(self):
        if False:
            while True:
                i = 10
        'Raise AttributeError if method not found in Namespace'
        note = frappe.get_doc({'doctype': 'Note', 'title': 'Test Note: Server Script'})
        self.assertRaises(AttributeError, note.insert)

    def test_syntax_validation(self):
        if False:
            for i in range(10):
                print('nop')
        server_script = scripts[0]
        server_script['script'] = 'js || code.?'
        with self.assertRaises(frappe.ValidationError) as se:
            frappe.get_doc(doctype='Server Script', **server_script).insert()
        self.assertTrue('invalid python code' in str(se.exception).lower(), msg='Python code validation not working')

    def test_commit_in_doctype_event(self):
        if False:
            return 10
        server_script = frappe.get_doc('Server Script', 'test_todo_commit')
        server_script.disabled = 0
        server_script.save()
        self.assertRaises(AttributeError, frappe.get_doc(dict(doctype='ToDo', description='test me')).insert)
        server_script.disabled = 1
        server_script.save()

    def test_add_index_in_doctype_event(self):
        if False:
            for i in range(10):
                print('nop')
        server_script = frappe.get_doc('Server Script', 'test_add_index')
        server_script.disabled = 0
        server_script.save()
        self.assertRaises(AttributeError, frappe.get_doc(dict(doctype='ToDo', description='test me')).insert)
        server_script.disabled = 1
        server_script.save()

    def test_restricted_qb(self):
        if False:
            for i in range(10):
                print('nop')
        todo = frappe.get_doc(doctype='ToDo', description='QbScriptTestNote')
        todo.insert()
        script = frappe.get_doc(doctype='Server Script', name='test_qb_restrictions', script_type='API', api_method='test_qb_restrictions', allow_guest=1, script=f'\nfrappe.db.set_value("ToDo", "{todo.name}", "description", "safe")\n')
        script.insert()
        script.execute_method()
        todo.reload()
        self.assertEqual(todo.description, 'safe')
        script.script = f'\ntodo = frappe.qb.DocType("ToDo")\nfrappe.qb.update(todo).set(todo.description, "unsafe").where(todo.name == "{todo.name}").run()\n'
        script.save()
        self.assertRaises(frappe.PermissionError, script.execute_method)
        todo.reload()
        self.assertEqual(todo.description, 'safe')
        script.script = f'\ntodo = frappe.qb.DocType("ToDo")\nfrappe.qb.from_(todo).select(todo.name).where(todo.name == "{todo.name}").run()\n'
        script.save()
        script.execute_method()

    def test_scripts_all_the_way_down(self):
        if False:
            while True:
                i = 10
        script = frappe.get_doc(doctype='Server Script', name='test_nested_scripts_1', script_type='API', api_method='test_nested_scripts_1', script=f'log("nothing")')
        script.insert()
        script.execute_method()
        script = frappe.get_doc(doctype='Server Script', name='test_nested_scripts_2', script_type='API', api_method='test_nested_scripts_2', script=f'frappe.call("test_nested_scripts_1")')
        script.insert()
        script.execute_method()

    def test_server_script_rate_limiting(self):
        if False:
            print('Hello World!')
        script1 = frappe.get_doc(doctype='Server Script', name='rate_limited_server_script', script_type='API', enable_rate_limit=1, allow_guest=1, rate_limit_count=5, api_method='rate_limited_endpoint', script='frappe.flags = {"test": True}')
        script1.insert()
        script2 = frappe.get_doc(doctype='Server Script', name='rate_limited_server_script2', script_type='API', enable_rate_limit=1, allow_guest=1, rate_limit_count=5, api_method='rate_limited_endpoint2', script='frappe.flags = {"test": False}')
        script2.insert()
        frappe.db.commit()
        site = frappe.utils.get_site_url(frappe.local.site)
        client = FrappeClient(site)
        for _ in range(5):
            client.get_api(script1.api_method)
        self.assertRaises(FrappeException, client.get_api, script1.api_method)
        for _ in range(5):
            client.get_api(script2.api_method)
        self.assertRaises(FrappeException, client.get_api, script2.api_method)
        script1.delete()
        script2.delete()
        frappe.db.commit()

    def test_server_script_scheduled(self):
        if False:
            print('Hello World!')
        scheduled_script = frappe.get_doc(doctype='Server Script', name='scheduled_script_wo_cron', script_type='Scheduler Event', script='frappe.flags = {"test": True}', event_frequency='Hourly').insert()
        cron_script = frappe.get_doc(doctype='Server Script', name='scheduled_script_w_cron', script_type='Scheduler Event', script='frappe.flags = {"test": True}', event_frequency='Cron', cron_format='0 0 1 1 *').insert()
        sync_jobs()
        self.assertTrue(frappe.db.exists('Scheduled Job Type', {'server_script': scheduled_script.name}))
        cron_job_name = frappe.db.get_value('Scheduled Job Type', {'server_script': cron_script.name})
        self.assertTrue(cron_job_name)
        cron_job = frappe.get_doc('Scheduled Job Type', cron_job_name)
        self.assertEqual(cron_job.next_execution.day, 1)
        self.assertEqual(cron_job.next_execution.month, 1)
        cron_script.cron_format = '0 0 2 1 *'
        cron_script.save()
        cron_job.reload()
        self.assertEqual(cron_job.next_execution.day, 2)