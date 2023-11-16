from odoo import models, api

class event_confirm(models.TransientModel):
    """Event Confirmation"""
    _name = 'event.confirm'

    @api.multi
    def confirm(self):
        if False:
            while True:
                i = 10
        events = self.env['event.event'].browse(self._context.get('event_ids', []))
        events.do_confirm()
        return {'type': 'ir.actions.act_window_close'}