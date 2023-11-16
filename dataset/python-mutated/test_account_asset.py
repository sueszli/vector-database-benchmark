from odoo import tools
from odoo.tests import common
from odoo.modules.module import get_resource_path

class TestAccountAsset(common.TransactionCase):

    def _load(self, module, *args):
        if False:
            i = 10
            return i + 15
        tools.convert_file(self.cr, 'account_asset', get_resource_path(module, *args), {}, 'init', False, 'test', self.registry._assertion_report)

    def test_00_account_asset_asset(self):
        if False:
            for i in range(10):
                print('nop')
        self._load('account', 'test', 'account_minimal_test.xml')
        self._load('account_asset', 'test', 'account_asset_demo_test.xml')
        self.browse_ref('account_asset.account_asset_asset_vehicles_test0').validate()
        self.assertEqual(self.browse_ref('account_asset.account_asset_asset_vehicles_test0').state, 'open', 'Asset should be in Open state')
        self.browse_ref('account_asset.account_asset_asset_vehicles_test0').compute_depreciation_board()
        value = self.browse_ref('account_asset.account_asset_asset_vehicles_test0')
        self.assertEqual(value.method_number, len(value.depreciation_line_ids), 'Depreciation lines not created correctly')
        ids = self.env['account.asset.depreciation.line'].search([('asset_id', '=', self.ref('account_asset.account_asset_asset_vehicles_test0'))])
        for line in ids:
            line.create_move()
        asset = self.env['account.asset.asset'].browse([self.ref('account_asset.account_asset_asset_vehicles_test0')])[0]
        self.assertEqual(len(asset.depreciation_line_ids), asset.entry_count, 'Move lines not created correctly')
        self.assertEqual(self.browse_ref('account_asset.account_asset_asset_vehicles_test0').state, 'close', 'State of asset should be close')
        account_asset_asset_office0 = self.browse_ref('account_asset.account_asset_asset_office_test0')
        asset_modify_number_0 = self.env['asset.modify'].create({'name': 'Test reason', 'method_number': 10.0}).with_context({'active_id': account_asset_asset_office0.id})
        asset_modify_number_0.with_context({'active_id': account_asset_asset_office0.id}).modify()
        self.assertEqual(account_asset_asset_office0.method_number, len(account_asset_asset_office0.depreciation_line_ids))
        context = {'active_ids': [self.ref('account_asset.menu_asset_depreciation_confirmation_wizard')], 'active_id': self.ref('account_asset.menu_asset_depreciation_confirmation_wizard'), 'type': 'sale'}
        asset_compute_period_0 = self.env['asset.depreciation.confirmation.wizard'].create({})
        asset_compute_period_0.with_context(context).asset_compute()