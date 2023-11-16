from odoo.tests.common import TransactionCase

class TestMenu(TransactionCase):

    def test_00_menu_deletion(self):
        if False:
            return 10
        'Verify that menu deletion works properly when there are child menus, and those\n           are indeed made orphans'
        Menu = self.env['ir.ui.menu']
        root = Menu.create({'name': 'Test root'})
        child1 = Menu.create({'name': 'Test child 1', 'parent_id': root.id})
        child2 = Menu.create({'name': 'Test child 2', 'parent_id': root.id})
        child21 = Menu.create({'name': 'Test child 2-1', 'parent_id': child2.id})
        all_ids = [root.id, child1.id, child2.id, child21.id]
        root.unlink()
        Menu = self.env['ir.ui.menu'].with_context({'ir.ui.menu.full_list': True})
        remaining = Menu.search([('id', 'in', all_ids)], order='id')
        self.assertEqual([child1.id, child2.id, child21.id], remaining.ids)
        orphans = Menu.search([('id', 'in', all_ids), ('parent_id', '=', False)], order='id')
        self.assertEqual([child1.id, child2.id], orphans.ids)