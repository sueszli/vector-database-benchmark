import logging
from odoo import api, models
_logger = logging.getLogger(__name__)

class AutoVacuum(models.AbstractModel):
    """ Expose the vacuum method to the cron jobs mechanism. """
    _name = 'ir.autovacuum'

    @api.model
    def _gc_transient_models(self):
        if False:
            while True:
                i = 10
        for mname in self.env:
            model = self.env[mname]
            if model.is_transient():
                model._transient_vacuum(force=True)

    @api.model
    def _gc_user_logs(self):
        if False:
            for i in range(10):
                print('nop')
        self._cr.execute('\n            DELETE FROM res_users_log log1 WHERE EXISTS (\n                SELECT 1 FROM res_users_log log2\n                WHERE log1.create_uid = log2.create_uid\n                AND log1.create_date < log2.create_date\n            )\n        ')
        _logger.info("GC'd %d user log entries", self._cr.rowcount)

    @api.model
    def power_on(self):
        if False:
            while True:
                i = 10
        self.env['ir.attachment']._file_gc()
        self._gc_transient_models()
        self._gc_user_logs()
        return True