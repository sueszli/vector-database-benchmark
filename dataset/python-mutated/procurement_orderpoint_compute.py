from odoo import api, models, tools
import logging
import threading
_logger = logging.getLogger(__name__)

class ProcurementOrderpointConfirm(models.TransientModel):
    _name = 'procurement.orderpoint.compute'
    _description = 'Compute Minimum Stock Rules'

    def _procure_calculation_orderpoint(self):
        if False:
            i = 10
            return i + 15
        with api.Environment.manage():
            new_cr = self.pool.cursor()
            self = self.with_env(self.env(cr=new_cr))
            scheduler_cron = self.sudo().env.ref('procurement.ir_cron_scheduler_action')
            try:
                with tools.mute_logger('odoo.sql_db'):
                    self._cr.execute('SELECT id FROM ir_cron WHERE id = %s FOR UPDATE NOWAIT', (scheduler_cron.id,))
            except Exception:
                _logger.info('Attempt to run procurement scheduler aborted, as already running')
                self._cr.rollback()
                self._cr.close()
                return {}
            self.env['procurement.order']._procure_orderpoint_confirm(use_new_cursor=new_cr.dbname, company_id=self.env.user.company_id.id)
            new_cr.close()
            return {}

    @api.multi
    def procure_calculation(self):
        if False:
            print('Hello World!')
        threaded_calculation = threading.Thread(target=self._procure_calculation_orderpoint, args=())
        threaded_calculation.start()
        return {'type': 'ir.actions.act_window_close'}