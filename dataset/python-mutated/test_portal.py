from odoo.addons.mail.tests.common import TestMail
from odoo.tools.misc import mute_logger

class TestPortal(TestMail):

    def test_mail_compose_access_rights(self):
        if False:
            for i in range(10):
                print('nop')
        self.group_pigs.write({'group_public_id': self.env.ref('base.group_portal').id})
        port_msg = self.group_pigs.message_post(body='Message')
        self.group_pigs.sudo(self.user_portal).message_post(body='I love Pigs', message_type='comment', subtype='mail.mt_comment')
        compose = self.env['mail.compose.message'].with_context({'default_composition_mode': 'comment', 'default_model': 'mail.channel', 'default_res_id': self.group_pigs.id}).sudo(self.user_portal).create({'subject': 'Subject', 'body': 'Body text', 'partner_ids': []})
        compose.send_mail()
        compose = self.env['mail.compose.message'].with_context({'default_composition_mode': 'comment', 'default_parent_id': port_msg.id}).sudo(self.user_portal).create({'subject': 'Subject', 'body': 'Body text'})
        compose.send_mail()

    @mute_logger('odoo.addons.mail.models.mail_mail')
    def test_invite_email_portal(self):
        if False:
            print('Hello World!')
        group_pigs = self.group_pigs
        base_url = self.env['ir.config_parameter'].get_param('web.base.url', default='')
        partner_carine = self.env['res.partner'].create({'name': 'Carine Poilvache', 'email': 'c@c'})
        self._init_mock_build_email()
        mail_invite = self.env['mail.wizard.invite'].with_context({'default_res_model': 'mail.channel', 'default_res_id': group_pigs.id}).create({'partner_ids': [(4, partner_carine.id)], 'send_mail': True})
        mail_invite.add_followers()
        self.assertEqual(group_pigs.message_partner_ids, partner_carine)
        self.assertEqual(len(self._mails), 1, 'sent email number incorrect, should be only for Bert')
        for sent_email in self._mails:
            self.assertEqual(sent_email.get('subject'), 'Invitation to follow Discussion channel: Pigs', 'invite: subject of invitation email is incorrect')
            self.assertIn('Administrator invited you to follow Discussion channel document: Pigs', sent_email.get('body'), 'invite: body of invitation email is incorrect')