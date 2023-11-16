import frappe
from frappe.tests.utils import FrappeTestCase

class TestSystemConsole(FrappeTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        if False:
            for i in range(10):
                print('nop')
        cls.enable_safe_exec()
        return super().setUpClass()

    def test_system_console(self):
        if False:
            return 10
        system_console = frappe.get_doc('System Console')
        system_console.console = 'log("hello")'
        system_console.run()
        self.assertEqual(system_console.output, 'hello')
        system_console.console = 'log(frappe.db.get_value("DocType", "DocType", "module"))'
        system_console.run()
        self.assertEqual(system_console.output, 'Core')

    def test_system_console_sql(self):
        if False:
            print('Hello World!')
        system_console = frappe.get_doc('System Console')
        system_console.type = 'SQL'
        system_console.console = "select 'test'"
        system_console.run()
        self.assertIn('test', system_console.output)
        system_console.console = "update `tabDocType` set is_virtual = 1 where name = 'xyz'"
        system_console.run()
        self.assertIn('PermissionError', system_console.output)