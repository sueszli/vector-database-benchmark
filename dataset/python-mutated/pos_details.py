from odoo import api, fields, models
from odoo.exceptions import UserError

class PosDetails(models.TransientModel):
    _name = 'pos.details.wizard'
    _description = 'Open Sale Details Report'

    def _default_start_date(self):
        if False:
            print('Hello World!')
        ' Find the earliest start_date of the latests sessions '
        config_ids = self.env['pos.config'].search([]).ids
        self.env.cr.execute("\n            SELECT\n            max(start_at) as start,\n            config_id\n            FROM pos_session\n            WHERE config_id = ANY(%s)\n            AND start_at > (NOW() - INTERVAL '2 DAYS')\n            GROUP BY config_id\n        ", (config_ids,))
        latest_start_dates = [res['start'] for res in self.env.cr.dictfetchall()]
        return latest_start_dates and min(latest_start_dates) or fields.Datetime.now()
    start_date = fields.Datetime(required=True, default=_default_start_date)
    end_date = fields.Datetime(required=True, default=fields.Datetime.now)
    pos_config_ids = fields.Many2many('pos.config', 'pos_detail_configs', default=lambda s: s.env['pos.config'].search([]))

    @api.onchange('start_date')
    def _onchange_start_date(self):
        if False:
            while True:
                i = 10
        if self.start_date and self.end_date and (self.end_date < self.start_date):
            self.end_date = self.start_date

    @api.onchange('end_date')
    def _onchange_end_date(self):
        if False:
            i = 10
            return i + 15
        if self.end_date and self.end_date < self.start_date:
            self.start_date = self.end_date

    @api.multi
    def generate_report(self):
        if False:
            while True:
                i = 10
        data = {'date_start': self.start_date, 'date_stop': self.end_date, 'config_ids': self.pos_config_ids.ids}
        return self.env['report'].get_action([], 'point_of_sale.report_saledetails', data=data)