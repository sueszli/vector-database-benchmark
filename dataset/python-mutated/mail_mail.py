from odoo import models

class mail_mail(models.Model):
    _inherit = 'mail.mail'

    def _postprocess_sent_message(self, cr, uid, mail, context=None, mail_sent=True):
        if False:
            return 10
        if mail_sent and mail.model == 'sale.order':
            so_obj = self.pool.get('sale.order')
            order = so_obj.browse(cr, uid, mail.res_id, context=context)
            partner = order.partner_id
            if partner not in order.message_partner_ids:
                so_obj.message_subscribe(cr, uid, [mail.res_id], [partner.id], context=context)
            for p in mail.partner_ids:
                if p not in order.message_partner_ids:
                    so_obj.message_subscribe(cr, uid, [mail.res_id], [p.id], context=context)
        return super(mail_mail, self)._postprocess_sent_message(cr, uid, mail=mail, context=context, mail_sent=mail_sent)