from odoo import api, fields, models

class Tour(models.Model):
    _name = 'web_tour.tour'
    _description = 'Tours'
    _log_access = False
    name = fields.Char(string='Tour name', required=True)
    user_id = fields.Many2one('res.users', string='Consumed by')

    @api.model
    def consume(self, tour_names):
        if False:
            i = 10
            return i + 15
        " Sets given tours as consumed, meaning that\n            these tours won't be active anymore for that user "
        for name in tour_names:
            self.create({'name': name, 'user_id': self.env.uid})

    @api.model
    def get_consumed_tours(self):
        if False:
            i = 10
            return i + 15
        ' Returns the list of consumed tours for the current user '
        return [t.name for t in self.search([('user_id', '=', self.env.uid)])]