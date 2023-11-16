import itertools
from odoo.addons.mail.tests.common import TestMail
from odoo.exceptions import AccessError, except_orm
from odoo.tools import mute_logger

class TestMailMessage(TestMail):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(TestMailMessage, self).setUp()
        self.group_private = self.env['mail.channel'].with_context({'mail_create_nolog': True, 'mail_create_nosubscribe': True}).create({'name': 'Private', 'public': 'private'}).with_context({'mail_create_nosubscribe': False})
        self.message = self.env['mail.message'].create({'body': 'My Body', 'model': 'mail.channel', 'res_id': self.group_private.id})

    def test_mail_message_values_basic(self):
        if False:
            i = 10
            return i + 15
        self.env['ir.config_parameter'].search([('key', '=', 'mail.catchall.domain')]).unlink()
        msg = self.env['mail.message'].sudo(self.user_employee).create({'reply_to': 'test.reply@example.com', 'email_from': 'test.from@example.com'})
        self.assertIn('-private', msg.message_id, 'mail_message: message_id for a void message should be a "private" one')
        self.assertEqual(msg.reply_to, 'test.reply@example.com')
        self.assertEqual(msg.email_from, 'test.from@example.com')

    def test_mail_message_values_default(self):
        if False:
            return 10
        self.env['ir.config_parameter'].search([('key', '=', 'mail.catchall.domain')]).unlink()
        msg = self.env['mail.message'].sudo(self.user_employee).create({})
        self.assertIn('-private', msg.message_id, 'mail_message: message_id for a void message should be a "private" one')
        self.assertEqual(msg.reply_to, '%s <%s>' % (self.user_employee.name, self.user_employee.email))
        self.assertEqual(msg.email_from, '%s <%s>' % (self.user_employee.name, self.user_employee.email))

    def test_mail_message_values_alias(self):
        if False:
            i = 10
            return i + 15
        alias_domain = 'example.com'
        self.env['ir.config_parameter'].set_param('mail.catchall.domain', alias_domain)
        self.env['ir.config_parameter'].search([('key', '=', 'mail.catchall.alias')]).unlink()
        msg = self.env['mail.message'].sudo(self.user_employee).create({})
        self.assertIn('-private', msg.message_id, 'mail_message: message_id for a void message should be a "private" one')
        self.assertEqual(msg.reply_to, '%s <%s>' % (self.user_employee.name, self.user_employee.email))
        self.assertEqual(msg.email_from, '%s <%s>' % (self.user_employee.name, self.user_employee.email))

    def test_mail_message_values_alias_catchall(self):
        if False:
            return 10
        alias_domain = 'example.com'
        alias_catchall = 'pokemon'
        self.env['ir.config_parameter'].set_param('mail.catchall.domain', alias_domain)
        self.env['ir.config_parameter'].set_param('mail.catchall.alias', alias_catchall)
        msg = self.env['mail.message'].sudo(self.user_employee).create({})
        self.assertIn('-private', msg.message_id, 'mail_message: message_id for a void message should be a "private" one')
        self.assertEqual(msg.reply_to, '%s <%s@%s>' % (self.env.user.company_id.name, alias_catchall, alias_domain))
        self.assertEqual(msg.email_from, '%s <%s>' % (self.user_employee.name, self.user_employee.email))

    def test_mail_message_values_document_no_alias(self):
        if False:
            while True:
                i = 10
        self.env['ir.config_parameter'].search([('key', '=', 'mail.catchall.domain')]).unlink()
        msg = self.env['mail.message'].sudo(self.user_employee).create({'model': 'mail.channel', 'res_id': self.group_pigs.id})
        self.assertIn('-openerp-%d-mail.channel' % self.group_pigs.id, msg.message_id, 'mail_message: message_id for a void message should be a "private" one')
        self.assertEqual(msg.reply_to, '%s <%s>' % (self.user_employee.name, self.user_employee.email))
        self.assertEqual(msg.email_from, '%s <%s>' % (self.user_employee.name, self.user_employee.email))

    def test_mail_message_values_document_alias(self):
        if False:
            for i in range(10):
                print('nop')
        alias_domain = 'example.com'
        self.env['ir.config_parameter'].set_param('mail.catchall.domain', alias_domain)
        self.env['ir.config_parameter'].search([('key', '=', 'mail.catchall.alias')]).unlink()
        msg = self.env['mail.message'].sudo(self.user_employee).create({'model': 'mail.channel', 'res_id': self.group_pigs.id})
        self.assertIn('-openerp-%d-mail.channel' % self.group_pigs.id, msg.message_id, 'mail_message: message_id for a void message should be a "private" one')
        self.assertEqual(msg.reply_to, '%s %s <%s@%s>' % (self.env.user.company_id.name, self.group_pigs.name, self.group_pigs.alias_name, alias_domain))
        self.assertEqual(msg.email_from, '%s <%s>' % (self.user_employee.name, self.user_employee.email))

    def test_mail_message_values_document_alias_catchall(self):
        if False:
            while True:
                i = 10
        alias_domain = 'example.com'
        alias_catchall = 'pokemon'
        self.env['ir.config_parameter'].set_param('mail.catchall.domain', alias_domain)
        self.env['ir.config_parameter'].set_param('mail.catchall.alias', alias_catchall)
        msg = self.env['mail.message'].sudo(self.user_employee).create({'model': 'mail.channel', 'res_id': self.group_pigs.id})
        self.assertIn('-openerp-%d-mail.channel' % self.group_pigs.id, msg.message_id, 'mail_message: message_id for a void message should be a "private" one')
        self.assertEqual(msg.reply_to, '%s %s <%s@%s>' % (self.env.user.company_id.name, self.group_pigs.name, self.group_pigs.alias_name, alias_domain))
        self.assertEqual(msg.email_from, '%s <%s>' % (self.user_employee.name, self.user_employee.email))

    def test_mail_message_values_no_auto_thread(self):
        if False:
            i = 10
            return i + 15
        msg = self.env['mail.message'].sudo(self.user_employee).create({'model': 'mail.channel', 'res_id': self.group_pigs.id, 'no_auto_thread': True})
        self.assertIn('reply_to', msg.message_id)
        self.assertNotIn('mail.channel', msg.message_id)
        self.assertNotIn('-%d-' % self.group_pigs.id, msg.message_id)

    def test_mail_message_notify_from_mail_mail(self):
        if False:
            print('Hello World!')
        self.email_to_list = []
        mail = self.env['mail.mail'].create({'body_html': '<p>Test</p>', 'email_to': 'test@example.com', 'partner_ids': [(4, self.user_employee.partner_id.id)]})
        self.email_to_list.extend(itertools.chain.from_iterable((sent_email['email_to'] for sent_email in self._mails if sent_email.get('email_to'))))
        self.assertNotIn(u'Ernest Employee <e.e@example.com>', self.email_to_list)
        mail.send()
        self.email_to_list.extend(itertools.chain.from_iterable((sent_email['email_to'] for sent_email in self._mails if sent_email.get('email_to'))))
        self.assertNotIn(u'Ernest Employee <e.e@example.com>', self.email_to_list)
        self.assertIn(u'test@example.com', self.email_to_list)

    @mute_logger('odoo.addons.mail.models.mail_mail')
    def test_mail_message_access_search(self):
        if False:
            while True:
                i = 10
        msg1 = self.env['mail.message'].create({'subject': '_Test', 'body': 'A', 'subtype_id': self.ref('mail.mt_comment')})
        msg2 = self.env['mail.message'].create({'subject': '_Test', 'body': 'A+B', 'subtype_id': self.ref('mail.mt_comment'), 'partner_ids': [(6, 0, [self.user_public.partner_id.id])]})
        msg3 = self.env['mail.message'].create({'subject': '_Test', 'body': 'A Pigs', 'subtype_id': False, 'model': 'mail.channel', 'res_id': self.group_pigs.id})
        msg4 = self.env['mail.message'].create({'subject': '_Test', 'body': 'A+P Pigs', 'subtype_id': self.ref('mail.mt_comment'), 'model': 'mail.channel', 'res_id': self.group_pigs.id, 'partner_ids': [(6, 0, [self.user_public.partner_id.id])]})
        msg5 = self.env['mail.message'].create({'subject': '_Test', 'body': 'A+E Pigs', 'subtype_id': self.ref('mail.mt_comment'), 'model': 'mail.channel', 'res_id': self.group_pigs.id, 'partner_ids': [(6, 0, [self.user_employee.partner_id.id])]})
        msg6 = self.env['mail.message'].create({'subject': '_Test', 'body': 'A Birds', 'subtype_id': self.ref('mail.mt_comment'), 'model': 'mail.channel', 'res_id': self.group_private.id})
        msg7 = self.env['mail.message'].sudo(self.user_employee).create({'subject': '_Test', 'body': 'B', 'subtype_id': self.ref('mail.mt_comment')})
        msg8 = self.env['mail.message'].sudo(self.user_employee).create({'subject': '_Test', 'body': 'B+E', 'subtype_id': self.ref('mail.mt_comment'), 'partner_ids': [(6, 0, [self.user_employee.partner_id.id])]})
        messages = self.env['mail.message'].sudo(self.user_public).search([('subject', 'like', '_Test')])
        self.assertEqual(messages, msg2 | msg4)
        messages = self.env['mail.message'].sudo(self.user_employee).search([('subject', 'like', '_Test'), ('body', 'ilike', 'A')])
        self.assertEqual(messages, msg3 | msg4 | msg5)
        messages = self.env['mail.message'].sudo(self.user_employee).search([('subject', 'like', '_Test')])
        self.assertEqual(messages, msg3 | msg4 | msg5 | msg7 | msg8)
        messages = self.env['mail.message'].search([('subject', 'like', '_Test')])
        self.assertEqual(messages, msg1 | msg2 | msg3 | msg4 | msg5 | msg6 | msg7 | msg8)
        messages = self.env['mail.message'].sudo(self.user_portal).search([('subject', 'like', '_Test')])
        self.assertFalse(messages)
        self.group_pigs.write({'public': 'public'})
        messages = self.env['mail.message'].sudo(self.user_portal).search([('subject', 'like', '_Test')])
        self.assertEqual(messages, msg4 | msg5)

    @mute_logger('odoo.addons.base.ir.ir_model', 'odoo.models')
    def test_mail_message_access_read_crash(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(except_orm):
            self.message.sudo(self.user_employee).read()

    @mute_logger('odoo.models')
    def test_mail_message_access_read_crash_portal(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(except_orm):
            self.message.sudo(self.user_portal).read(['body', 'message_type', 'subtype_id'])

    def test_mail_message_access_read_ok_portal(self):
        if False:
            for i in range(10):
                print('nop')
        self.message.write({'subtype_id': self.ref('mail.mt_comment'), 'res_id': self.group_public.id})
        self.message.sudo(self.user_portal).read(['body', 'message_type', 'subtype_id'])

    def test_mail_message_access_read_notification(self):
        if False:
            i = 10
            return i + 15
        attachment = self.env['ir.attachment'].create({'datas': 'My attachment'.encode('base64'), 'name': 'doc.txt', 'datas_fname': 'doc.txt'})
        self.message.write({'attachment_ids': [(4, attachment.id)]})
        self.message.write({'partner_ids': [(4, self.user_employee.partner_id.id)]})
        self.message.sudo(self.user_employee).read()
        attachment.sudo(self.user_employee).read(['name', 'datas'])

    def test_mail_message_access_read_author(self):
        if False:
            i = 10
            return i + 15
        self.message.write({'author_id': self.user_employee.partner_id.id})
        self.message.sudo(self.user_employee).read()

    def test_mail_message_access_read_doc(self):
        if False:
            for i in range(10):
                print('nop')
        self.message.write({'model': 'mail.channel', 'res_id': self.group_public.id})
        self.message.sudo(self.user_employee).read()

    @mute_logger('odoo.addons.base.ir.ir_model')
    def test_mail_message_access_create_crash_public(self):
        if False:
            return 10
        with self.assertRaises(AccessError):
            self.env['mail.message'].sudo(self.user_public).create({'model': 'mail.channel', 'res_id': self.group_pigs.id, 'body': 'Test'})
        with self.assertRaises(AccessError):
            self.env['mail.message'].sudo(self.user_public).create({'model': 'mail.channel', 'res_id': self.group_public.id, 'body': 'Test'})

    @mute_logger('odoo.models')
    def test_mail_message_access_create_crash(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(except_orm):
            self.env['mail.message'].sudo(self.user_employee).create({'model': 'mail.channel', 'res_id': self.group_private.id, 'body': 'Test'})

    @mute_logger('odoo.models')
    def test_mail_message_access_create_doc(self):
        if False:
            while True:
                i = 10
        Message = self.env['mail.message'].sudo(self.user_employee)
        Message.create({'model': 'mail.channel', 'res_id': self.group_public.id, 'body': 'Test'})
        with self.assertRaises(except_orm):
            Message.create({'model': 'mail.channel', 'res_id': self.group_private.id, 'body': 'Test'})

    def test_mail_message_access_create_private(self):
        if False:
            while True:
                i = 10
        self.env['mail.message'].sudo(self.user_employee).create({'body': 'Test'})

    def test_mail_message_access_create_reply(self):
        if False:
            return 10
        self.message.write({'partner_ids': [(4, self.user_employee.partner_id.id)]})
        self.env['mail.message'].sudo(self.user_employee).create({'model': 'mail.channel', 'res_id': self.group_private.id, 'body': 'Test', 'parent_id': self.message.id})

    def test_message_set_star(self):
        if False:
            i = 10
            return i + 15
        msg = self.group_pigs.message_post(body='My Body', subject='1')
        msg_emp = self.env['mail.message'].sudo(self.user_employee).browse(msg.id)
        msg.toggle_message_starred()
        self.assertTrue(msg.starred)
        msg_emp.toggle_message_starred()
        self.assertTrue(msg_emp.starred)
        msg.toggle_message_starred()
        self.assertFalse(msg.starred)
        self.assertTrue(msg_emp.starred)

    def test_60_cache_invalidation(self):
        if False:
            while True:
                i = 10
        msg_cnt = len(self.group_pigs.message_ids)
        self.group_pigs.message_post(body='Hi!', subject='test')
        self.assertEqual(len(self.group_pigs.message_ids), msg_cnt + 1)