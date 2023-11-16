from odoo import api, fields, models, tools

class ImLivechatReportChannel(models.Model):
    """ Livechat Support Report on the Channels """
    _name = 'im_livechat.report.channel'
    _description = 'Livechat Support Report'
    _order = 'start_date, technical_name'
    _auto = False
    uuid = fields.Char('UUID', readonly=True)
    channel_id = fields.Many2one('mail.channel', 'Conversation', readonly=True)
    channel_name = fields.Char('Channel Name', readonly=True)
    technical_name = fields.Char('Code', readonly=True)
    livechat_channel_id = fields.Many2one('im_livechat.channel', 'Channel', readonly=True)
    start_date = fields.Datetime('Start Date of session', readonly=True, help='Start date of the conversation')
    start_date_hour = fields.Char('Hour of start Date of session', readonly=True)
    duration = fields.Float('Average duration', digits=(16, 2), readonly=True, group_operator='avg', help='Duration of the conversation (in seconds)')
    nbr_speaker = fields.Integer('# of speakers', readonly=True, group_operator='avg', help='Number of different speakers')
    nbr_message = fields.Integer('Average message', readonly=True, group_operator='avg', help='Number of message in the conversation')
    partner_id = fields.Many2one('res.partner', 'Operator', readonly=True)

    @api.model_cr
    def init(self):
        if False:
            while True:
                i = 10
        tools.drop_view_if_exists(self.env.cr, 'im_livechat_report_channel')
        self.env.cr.execute("\n            CREATE OR REPLACE VIEW im_livechat_report_channel AS (\n                SELECT\n                    C.id as id,\n                    C.uuid as uuid,\n                    C.id as channel_id,\n                    C.name as channel_name,\n                    CONCAT(L.name, ' / ', C.id) as technical_name,\n                    C.livechat_channel_id as livechat_channel_id,\n                    C.create_date as start_date,\n                    to_char(date_trunc('hour', C.create_date), 'YYYY-MM-DD HH24:MI:SS') as start_date_hour,\n                    EXTRACT('epoch' FROM (max((SELECT (max(M.create_date)) FROM mail_message M JOIN mail_message_mail_channel_rel R ON (R.mail_message_id = M.id) WHERE R.mail_channel_id = C.id))-C.create_date)) as duration,\n                    count(distinct P.id) as nbr_speaker,\n                    count(distinct M.id) as nbr_message,\n                    MAX(S.partner_id) as partner_id\n                FROM mail_channel C\n                    JOIN mail_message_mail_channel_rel R ON (C.id = R.mail_channel_id)\n                    JOIN mail_message M ON (M.id = R.mail_message_id)\n                    JOIN mail_channel_partner S ON (S.channel_id = C.id)\n                    JOIN im_livechat_channel L ON (L.id = C.livechat_channel_id)\n                    LEFT JOIN res_partner P ON (M.author_id = P.id)\n                GROUP BY C.id, C.name, C.livechat_channel_id, L.name, C.create_date, C.uuid\n            )\n        ")