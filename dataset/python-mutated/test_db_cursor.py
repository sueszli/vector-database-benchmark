import unittest
import odoo
from odoo.tests import common
from odoo.tools.misc import mute_logger
ADMIN_USER_ID = common.ADMIN_USER_ID

def registry():
    if False:
        return 10
    return odoo.registry(common.get_db_name())

class TestExecute(unittest.TestCase):
    """ Try cr.execute with wrong parameters """

    @mute_logger('odoo.sql_db')
    def test_execute_bad_params(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Try to use iterable but non-list or int params in query parameters.\n        '
        with registry().cursor() as cr:
            with self.assertRaises(ValueError):
                cr.execute('SELECT id FROM res_users WHERE login=%s', 'admin')
            with self.assertRaises(ValueError):
                cr.execute('SELECT id FROM res_users WHERE id=%s', 1)
            with self.assertRaises(ValueError):
                cr.execute('SELECT id FROM res_users WHERE id=%s', '1')