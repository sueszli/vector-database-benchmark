import datetime
import time
from odoo import api, fields, models
from odoo import tools
from odoo.addons.bus.models.bus import TIMEOUT
from odoo.tools.misc import DEFAULT_SERVER_DATETIME_FORMAT
DISCONNECTION_TIMER = TIMEOUT + 5
AWAY_TIMER = 1800

class BusPresence(models.Model):
    """ User Presence
        Its status is 'online', 'away' or 'offline'. This model should be a one2one, but is not
        attached to res_users to avoid database concurrence errors. Since the 'update' method is executed
        at each poll, if the user have multiple opened tabs, concurrence errors can happend, but are 'muted-logged'.
    """
    _name = 'bus.presence'
    _description = 'User Presence'
    _log_access = False
    _sql_constraints = [('bus_user_presence_unique', 'unique(user_id)', 'A user can only have one IM status.')]
    user_id = fields.Many2one('res.users', 'Users', required=True, index=True, ondelete='cascade')
    last_poll = fields.Datetime('Last Poll', default=lambda self: fields.Datetime.now())
    last_presence = fields.Datetime('Last Presence', default=lambda self: fields.Datetime.now())
    status = fields.Selection([('online', 'Online'), ('away', 'Away'), ('offline', 'Offline')], 'IM Status', default='offline')

    @api.model
    def update(self, inactivity_period):
        if False:
            print('Hello World!')
        ' Updates the last_poll and last_presence of the current user\n            :param inactivity_period: duration in milliseconds\n        '
        presence = self.search([('user_id', '=', self._uid)], limit=1)
        last_presence = datetime.datetime.now() - datetime.timedelta(milliseconds=inactivity_period)
        values = {'last_poll': time.strftime(DEFAULT_SERVER_DATETIME_FORMAT)}
        if not presence:
            values['user_id'] = self._uid
            values['last_presence'] = last_presence.strftime(DEFAULT_SERVER_DATETIME_FORMAT)
            self.create(values)
        else:
            if datetime.datetime.strptime(presence.last_presence, DEFAULT_SERVER_DATETIME_FORMAT) < last_presence:
                values['last_presence'] = last_presence.strftime(DEFAULT_SERVER_DATETIME_FORMAT)
            with tools.mute_logger('odoo.sql_db'):
                presence.write(values)
        self.env.cr.commit()