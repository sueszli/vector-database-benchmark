from odoo import api, fields, models, tools

class AssetAssetReport(models.Model):
    _name = 'asset.asset.report'
    _description = 'Assets Analysis'
    _auto = False
    name = fields.Char(string='Year', required=False, readonly=True)
    date = fields.Date(readonly=True)
    depreciation_date = fields.Date(string='Depreciation Date', readonly=True)
    asset_id = fields.Many2one('account.asset.asset', string='Asset', readonly=True)
    asset_category_id = fields.Many2one('account.asset.category', string='Asset category', readonly=True)
    partner_id = fields.Many2one('res.partner', string='Partner', readonly=True)
    state = fields.Selection([('draft', 'Draft'), ('open', 'Running'), ('close', 'Close')], string='Status', readonly=True)
    depreciation_value = fields.Float(string='Amount of Depreciation Lines', readonly=True)
    installment_value = fields.Float(string='Amount of Installment Lines', readonly=True)
    move_check = fields.Boolean(string='Posted', readonly=True)
    installment_nbr = fields.Integer(string='# of Installment Lines', readonly=True)
    depreciation_nbr = fields.Integer(string='# of Depreciation Lines', readonly=True)
    gross_value = fields.Float(string='Gross Amount', readonly=True)
    posted_value = fields.Float(string='Posted Amount', readonly=True)
    unposted_value = fields.Float(string='Unposted Amount', readonly=True)
    company_id = fields.Many2one('res.company', string='Company', readonly=True)

    @api.model_cr
    def init(self):
        if False:
            i = 10
            return i + 15
        tools.drop_view_if_exists(self._cr, 'asset_asset_report')
        self._cr.execute('\n            create or replace view asset_asset_report as (\n                select\n                    min(dl.id) as id,\n                    dl.name as name,\n                    dl.depreciation_date as depreciation_date,\n                    a.date as date,\n                    (CASE WHEN dlmin.id = min(dl.id)\n                      THEN a.value\n                      ELSE 0\n                      END) as gross_value,\n                    dl.amount as depreciation_value,\n                    dl.amount as installment_value,\n                    (CASE WHEN dl.move_check\n                      THEN dl.amount\n                      ELSE 0\n                      END) as posted_value,\n                    (CASE WHEN NOT dl.move_check\n                      THEN dl.amount\n                      ELSE 0\n                      END) as unposted_value,\n                    dl.asset_id as asset_id,\n                    dl.move_check as move_check,\n                    a.category_id as asset_category_id,\n                    a.partner_id as partner_id,\n                    a.state as state,\n                    count(dl.*) as installment_nbr,\n                    count(dl.*) as depreciation_nbr,\n                    a.company_id as company_id\n                from account_asset_depreciation_line dl\n                    left join account_asset_asset a on (dl.asset_id=a.id)\n                    left join (select min(d.id) as id,ac.id as ac_id from account_asset_depreciation_line as d inner join account_asset_asset as ac ON (ac.id=d.asset_id) group by ac_id) as dlmin on dlmin.ac_id=a.id\n                group by\n                    dl.amount,dl.asset_id,dl.depreciation_date,dl.name,\n                    a.date, dl.move_check, a.state, a.category_id, a.partner_id, a.company_id,\n                    a.value, a.id, a.salvage_value, dlmin.id\n        )')