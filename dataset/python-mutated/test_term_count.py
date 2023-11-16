import odoo
from odoo.tests import common

class TestTermCount(common.TransactionCase):

    def test_count_term(self):
        if False:
            return 10
        '\n        Just make sure we have as many translation entries as we wanted.\n        '
        odoo.tools.trans_load(self.cr, 'test_translation_import/i18n/fr.po', 'fr_FR', verbose=False)
        ids = self.env['ir.translation'].search([('src', '=', '1XBUO5PUYH2RYZSA1FTLRYS8SPCNU1UYXMEYMM25ASV7JC2KTJZQESZYRV9L8CGB')])
        self.assertEqual(len(ids), 2)

    def test_noupdate(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Make sure no update do not overwrite translations\n        '
        menu = self.env.ref('test_translation_import.menu_test_translation_import')
        menu.name = 'New Name'
        odoo.tools.trans_load(self.cr, 'test_translation_import/i18n/fr.po', 'fr_FR', verbose=False)
        menu.with_context(lang='fr_FR').name = 'Nouveau nom'
        odoo.tools.trans_load(self.cr, 'test_translation_import/i18n/fr.po', 'fr_FR', verbose=False, context={'overwrite': True})
        menu.refresh()
        self.assertEqual(menu.name, 'New Name')
        self.assertEqual(menu.with_context(lang='fr_FR').name, 'Nouveau nom')