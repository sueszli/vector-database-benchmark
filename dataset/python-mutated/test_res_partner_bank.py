from odoo.tests.common import TransactionCase

class TestResPartnerBank(TransactionCase):
    """Tests acc_number
    """

    def test_sanitized_acc_number(self):
        if False:
            while True:
                i = 10
        partner_bank_model = self.env['res.partner.bank']
        acc_number = ' BE-001 2518823 03 '
        vals = partner_bank_model.search([('acc_number', '=', acc_number)])
        self.assertEquals(0, len(vals))
        partner_bank = partner_bank_model.create({'acc_number': acc_number, 'partner_id': self.ref('base.res_partner_2'), 'acc_type': 'bank'})
        vals = partner_bank_model.search([('acc_number', '=', acc_number)])
        self.assertEquals(1, len(vals))
        self.assertEquals(partner_bank, vals[0])
        vals = partner_bank_model.search([('acc_number', 'in', [acc_number])])
        self.assertEquals(1, len(vals))
        self.assertEquals(partner_bank, vals[0])
        self.assertEqual(partner_bank.acc_number, acc_number)
        sanitized_acc_number = 'BE001251882303'
        vals = partner_bank_model.search([('acc_number', '=', sanitized_acc_number)])
        self.assertEquals(1, len(vals))
        self.assertEquals(partner_bank, vals[0])
        vals = partner_bank_model.search([('acc_number', 'in', [sanitized_acc_number])])
        self.assertEquals(1, len(vals))
        self.assertEquals(partner_bank, vals[0])
        self.assertEqual(partner_bank.sanitized_acc_number, sanitized_acc_number)
        vals = partner_bank_model.search([('acc_number', '=', sanitized_acc_number.lower())])
        self.assertEquals(1, len(vals))
        vals = partner_bank_model.search([('acc_number', '=', acc_number.lower())])
        self.assertEquals(1, len(vals))