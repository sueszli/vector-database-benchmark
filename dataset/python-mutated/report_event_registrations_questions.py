from odoo import api, fields, models, tools

class ReportEventRegistrationQuestions(models.Model):
    _name = 'event.question.report'
    _auto = False
    attendee_id = fields.Many2one(comodel_name='event.registration', string='Registration')
    question_id = fields.Many2one(comodel_name='event.question', string='Question')
    answer_id = fields.Many2one(comodel_name='event.answer', string='Answer')
    event_id = fields.Many2one(comodel_name='event.event', string='Event')

    @api.model_cr
    def init(self):
        if False:
            for i in range(10):
                print('nop')
        ' Event Question main report '
        tools.drop_view_if_exists(self._cr, 'event_question_report')
        self._cr.execute(' CREATE VIEW event_question_report AS (\n            SELECT\n                att_answer.id as id,\n                att_answer.event_registration_id as attendee_id,\n                answer.question_id as question_id,\n                answer.id as answer_id,\n                question.event_id as event_id\n            FROM\n                event_registration_answer as att_answer\n            LEFT JOIN\n                event_answer as answer ON answer.id = att_answer.event_answer_id\n            LEFT JOIN\n                event_question as question ON question.id = answer.question_id\n            GROUP BY\n                attendee_id,\n                event_id,\n                question_id,\n                answer_id,\n                att_answer.id\n        )')