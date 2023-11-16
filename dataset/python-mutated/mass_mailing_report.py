from odoo import api, fields, models, tools

class MassMailingReport(models.Model):
    _name = 'mail.statistics.report'
    _auto = False
    _description = 'Mass Mailing Statistics'
    scheduled_date = fields.Datetime(stirng='Scheduled Date', readonly=True)
    name = fields.Char(string='Mass Mail', readonly=True)
    campaign = fields.Char(string='Mass Mail Campaign', readonly=True)
    sent = fields.Integer(readonly=True)
    delivered = fields.Integer(readonly=True)
    opened = fields.Integer(readonly=True)
    bounced = fields.Integer(readonly=True)
    replied = fields.Integer(readonly=True)
    state = fields.Selection([('draft', 'Draft'), ('test', 'Tested'), ('done', 'Sent')], string='Status', readonly=True)
    email_from = fields.Char('From', readonly=True)

    @api.model_cr
    def init(self):
        if False:
            i = 10
            return i + 15
        'Mass Mail Statistical Report: based on mail.mail.statistics that models the various\n        statistics collected for each mailing, and mail.mass_mailing model that models the\n        various mailing performed. '
        tools.drop_view_if_exists(self.env.cr, 'mail_statistics_report')
        self.env.cr.execute('\n            CREATE OR REPLACE VIEW mail_statistics_report AS (\n                SELECT\n                    min(ms.id) as id,\n                    ms.scheduled as scheduled_date,\n                    utm_source.name as name,\n                    utm_campaign.name as campaign,\n                    count(ms.bounced) as bounced,\n                    count(ms.sent) as sent,\n                    (count(ms.sent) - count(ms.bounced)) as delivered,\n                    count(ms.opened) as opened,\n                    count(ms.replied) as replied,\n                    mm.state,\n                    mm.email_from\n                FROM\n                    mail_mail_statistics as ms\n                    left join mail_mass_mailing as mm ON (ms.mass_mailing_id=mm.id)\n                    left join mail_mass_mailing_campaign as mc ON (ms.mass_mailing_campaign_id=mc.id)\n                    left join utm_campaign as utm_campaign ON (mc.campaign_id = utm_campaign.id)\n                    left join utm_source as utm_source ON (mm.source_id = utm_source.id)\n                GROUP BY ms.scheduled, utm_source.name, utm_campaign.name, mm.state, mm.email_from\n            )')