from odoo import fields, models

class MassMailingList(models.Model):
    _inherit = 'mail.mass_mailing.list'

    def _default_popup_content(self):
        if False:
            return 10
        return '<div class="modal-header text-center">\n    <h3 class="modal-title mt8">Odoo Presents</h3>\n</div>\n<div class="o_popup_message">\n    <font>7</font>\n    <strong>Business Hacks</strong>\n    <span> to<br/>boost your marketing</span>\n</div>\n<p class="o_message_paragraph">Join our Marketing newsletter and get <strong>this white paper instantly</strong></p>'
    popup_content = fields.Html(string='Website Popup Content', translate=True, sanitize_attributes=False, default=_default_popup_content)
    popup_redirect_url = fields.Char(string='Website Popup Redirect URL', default='/')