from odoo.tests import common

class TestSingleTransactionCase(common.SingleTransactionCase):
    """
    Check the whole-class transaction behavior of SingleTransactionCase.
    """

    def test_00(self):
        if False:
            print('Hello World!')
        ' Create a partner. '
        self.env['res.partner'].create({'name': 'test_per_class_teardown_partner'})
        partners = self.env['res.partner'].search([('name', '=', 'test_per_class_teardown_partner')])
        self.assertEqual(1, len(partners), 'Test partner not found.')

    def test_01(self):
        if False:
            while True:
                i = 10
        ' Find the created partner. '
        partners = self.env['res.partner'].search([('name', '=', 'test_per_class_teardown_partner')])
        self.assertEqual(1, len(partners), 'Test partner not found.')

    def test_20a(self):
        if False:
            return 10
        ' Create a partner with a XML ID '
        (pid, _) = self.env['res.partner'].name_create('Mr Blue')
        self.env['ir.model.data'].create({'name': 'test_partner_blue', 'module': 'base', 'model': 'res.partner', 'res_id': pid})

    def test_20b(self):
        if False:
            return 10
        ' Resolve xml id with ref() and browse_ref() '
        xid = 'base.test_partner_blue'
        partner = self.env.ref(xid)
        pid = self.ref(xid)
        self.assertTrue(pid, 'ref() should resolve xid to database ID')
        self.assertEqual(pid, partner.id, 'ref() is not consistent with env.ref()')
        partner2 = self.browse_ref(xid)
        self.assertEqual(partner, partner2, 'browse_ref() should resolve xid to browse records')

class TestTransactionCase(common.TransactionCase):
    """
    Check the per-method transaction behavior of TransactionCase.
    """

    def test_00(self):
        if False:
            while True:
                i = 10
        ' Create a partner. '
        partners = self.env['res.partner'].search([('name', '=', 'test_per_class_teardown_partner')])
        self.assertEqual(0, len(partners), 'Test partner found.')
        self.env['res.partner'].create({'name': 'test_per_class_teardown_partner'})
        partners = self.env['res.partner'].search([('name', '=', 'test_per_class_teardown_partner')])
        self.assertEqual(1, len(partners), 'Test partner not found.')

    def test_01(self):
        if False:
            print('Hello World!')
        " Don't find the created partner. "
        partners = self.env['res.partner'].search([('name', '=', 'test_per_class_teardown_partner')])
        self.assertEqual(0, len(partners), 'Test partner found.')

    def test_20a(self):
        if False:
            i = 10
            return i + 15
        ' Create a partner with a XML ID then resolve xml id with ref() and browse_ref() '
        (pid, _) = self.env['res.partner'].name_create('Mr Yellow')
        self.env['ir.model.data'].create({'name': 'test_partner_yellow', 'module': 'base', 'model': 'res.partner', 'res_id': pid})
        xid = 'base.test_partner_yellow'
        partner = self.env.ref(xid)
        pid = self.ref(xid)
        self.assertEquals(pid, partner.id, 'ref() should resolve xid to database ID')
        partner2 = self.browse_ref(xid)
        self.assertEqual(partner, partner2, 'browse_ref() should resolve xid to browse records')