from email.utils import formataddr
from .common import TestMail
from odoo import api
from odoo.exceptions import AccessError, except_orm
from odoo.tools import mute_logger

class TestMailGroup(TestMail):

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        super(TestMailGroup, cls).setUpClass()
        cls.registry('mail.channel')._revert_method('message_get_recipient_values')
        cls.group_private = cls.env['mail.channel'].with_context({'mail_create_nolog': True, 'mail_create_nosubscribe': True}).create({'name': 'Private', 'public': 'private'}).with_context({'mail_create_nosubscribe': False})

    @classmethod
    def tearDownClass(cls):
        if False:
            for i in range(10):
                print('nop')

        @api.multi
        def mail_group_message_get_recipient_values(self, notif_message=None, recipient_ids=None):
            if False:
                for i in range(10):
                    print('nop')
            return self.env['mail.thread'].message_get_recipient_values(notif_message=notif_message, recipient_ids=recipient_ids)
        cls.env['mail.channel']._patch_method('message_get_recipient_values', mail_group_message_get_recipient_values)
        super(TestMail, cls).tearDownClass()

    @mute_logger('odoo.addons.base.ir.ir_model', 'odoo.models')
    def test_access_rights_public(self):
        if False:
            print('Hello World!')
        self.group_public.sudo(self.user_public).read()
        with self.assertRaises(except_orm):
            self.group_pigs.sudo(self.user_public).read()
        self.group_private.write({'channel_partner_ids': [(4, self.user_public.partner_id.id)]})
        self.group_private.sudo(self.user_public).read()
        with self.assertRaises(AccessError):
            self.env['mail.channel'].sudo(self.user_public).create({'name': 'Test'})
        with self.assertRaises(AccessError):
            self.group_public.sudo(self.user_public).write({'name': 'Broutouschnouk'})
        with self.assertRaises(AccessError):
            self.group_public.sudo(self.user_public).unlink()

    @mute_logger('odoo.addons.base.ir.ir_model', 'odoo.models')
    def test_access_rights_groups(self):
        if False:
            return 10
        self.group_pigs.sudo(self.user_employee).read()
        self.env['mail.channel'].sudo(self.user_employee).create({'name': 'Test'})
        self.group_pigs.sudo(self.user_employee).write({'name': 'modified'})
        self.group_pigs.sudo(self.user_employee).unlink()
        with self.assertRaises(except_orm):
            self.group_private.sudo(self.user_employee).read()
        with self.assertRaises(AccessError):
            self.group_private.sudo(self.user_employee).write({'name': 're-modified'})

    def test_access_rights_followers_ko(self):
        if False:
            print('Hello World!')
        with self.assertRaises(AccessError):
            self.group_private.sudo(self.user_portal).name

    def test_access_rights_followers_portal(self):
        if False:
            for i in range(10):
                print('nop')
        self.group_private.write({'channel_partner_ids': [(4, self.user_portal.partner_id.id)]})
        chell_pigs = self.group_private.sudo(self.user_portal)
        trigger_read = chell_pigs.name
        for message in chell_pigs.message_ids:
            trigger_read = message.subject
        for partner in chell_pigs.message_partner_ids:
            if partner.id == self.user_portal.partner_id.id:
                continue
            with self.assertRaises(except_orm):
                trigger_read = partner.name

    def test_mail_group_notification_recipients_grouped(self):
        if False:
            for i in range(10):
                print('nop')
        self.env['ir.config_parameter'].set_param('mail.catchall.domain', 'schlouby.fr')
        self.group_private.write({'alias_name': 'Test'})
        self.group_private.message_subscribe_users([self.user_employee.id, self.user_portal.id])
        self.group_private.message_post(body='Test', message_type='comment', subtype='mt_comment')
        sent_emails = self._mails
        self.assertEqual(len(sent_emails), 1)
        for email in sent_emails:
            self.assertEqual(set(email['email_to']), set([formataddr((self.user_employee.name, self.user_employee.email)), formataddr((self.user_portal.name, self.user_portal.email))]))

    def test_mail_group_notification_recipients_separated(self):
        if False:
            print('Hello World!')
        self.group_private.write({'alias_name': False})
        self.group_private.message_subscribe_users([self.user_employee.id, self.user_portal.id])
        self.group_private.message_post(body='Test', message_type='comment', subtype='mt_comment')
        sent_emails = self._mails
        self.assertEqual(len(sent_emails), 2)
        for email in sent_emails:
            self.assertIn(email['email_to'][0], [formataddr((self.user_employee.name, self.user_employee.email)), formataddr((self.user_portal.name, self.user_portal.email))])