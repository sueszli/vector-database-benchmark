from odoo import api, models, fields
from odoo import tools

class ReportEventRegistration(models.Model):
    """Events Analysis"""
    _name = 'report.event.registration'
    _order = 'event_date desc'
    _auto = False
    create_date = fields.Datetime('Creation Date', readonly=True)
    event_date = fields.Datetime('Event Date', readonly=True)
    event_id = fields.Many2one('event.event', 'Event', required=True)
    draft_state = fields.Integer(' # No of Draft Registrations')
    cancel_state = fields.Integer(' # No of Cancelled Registrations')
    confirm_state = fields.Integer(' # No of Confirmed Registrations')
    seats_max = fields.Integer('Max Seats')
    nbevent = fields.Integer('Number of Events')
    nbregistration = fields.Integer('Number of Registrations')
    event_type_id = fields.Many2one('event.type', 'Event Type')
    registration_state = fields.Selection([('draft', 'Draft'), ('confirm', 'Confirmed'), ('done', 'Attended'), ('cancel', 'Cancelled')], 'Registration State', readonly=True, required=True)
    event_state = fields.Selection([('draft', 'Draft'), ('confirm', 'Confirmed'), ('done', 'Done'), ('cancel', 'Cancelled')], 'Event State', readonly=True, required=True)
    user_id = fields.Many2one('res.users', 'Event Responsible', readonly=True)
    name_registration = fields.Char('Participant / Contact Name', readonly=True)
    company_id = fields.Many2one('res.company', 'Company', readonly=True)

    def _select(self):
        if False:
            i = 10
            return i + 15
        return "\n            SELECT\n                e.id::varchar || '/' || coalesce(r.id::varchar,'') AS id,\n                e.id AS event_id,\n                e.user_id AS user_id,\n                r.name AS name_registration,\n                r.create_date AS create_date,\n                e.company_id AS company_id,\n                e.date_begin AS event_date,\n                count(r.id) AS nbevent,\n                count(r.event_id) AS nbregistration,\n                CASE WHEN r.state IN ('draft') THEN count(r.event_id) ELSE 0 END AS draft_state,\n                CASE WHEN r.state IN ('open','done') THEN count(r.event_id) ELSE 0 END AS confirm_state,\n                CASE WHEN r.state IN ('cancel') THEN count(r.event_id) ELSE 0 END AS cancel_state,\n                e.event_type_id AS event_type_id,\n                e.seats_max AS seats_max,\n                e.state AS event_state,\n                r.state AS registration_state\n            "

    def _from(self):
        if False:
            while True:
                i = 10
        return '\n            FROM\n                event_event e\n                LEFT JOIN event_registration r ON (e.id=r.event_id)\n            '

    def _group_by(self):
        if False:
            return 10
        return '\n            GROUP BY\n                event_id,\n                r.id,\n                registration_state,\n                event_type_id,\n                e.id,\n                e.date_begin,\n                e.user_id,\n                event_state,\n                e.company_id,\n                e.seats_max,\n                name_registration\n            '

    @api.model_cr
    def init(self):
        if False:
            for i in range(10):
                print('nop')
        tools.drop_view_if_exists(self.env.cr, self._table)
        self.env.cr.execute('CREATE or REPLACE VIEW %s as (%s %s %s)' % (self._table, self._select(), self._from(), self._group_by()))