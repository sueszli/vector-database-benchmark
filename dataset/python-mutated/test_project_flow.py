from .test_project_base import TestProjectBase
from odoo.tools import mute_logger
EMAIL_TPL = 'Return-Path: <whatever-2a840@postmaster.twitter.com>\nX-Original-To: {to}\nDelivered-To: {to}\nTo: {to}\ncc: {cc}\nReceived: by mail1.odoo.com (Postfix, from userid 10002)\n    id 5DF9ABFB2A; Fri, 10 Aug 2012 16:16:39 +0200 (CEST)\nMessage-ID: {msg_id}\nDate: Tue, 29 Nov 2011 12:43:21 +0530\nFrom: {email_from}\nMIME-Version: 1.0\nSubject: {subject}\nContent-Type: text/plain; charset=ISO-8859-1; format=flowed\n\nHello,\n\nThis email should create a new entry in your module. Please check that it\neffectively works.\n\nThanks,\n\n--\nRaoul Boitempoils\nIntegrator at Agrolait'

class TestProjectFlow(TestProjectBase):

    def test_project_process_project_manager_duplicate(self):
        if False:
            i = 10
            return i + 15
        pigs = self.project_pigs.sudo(self.user_projectmanager)
        dogs = pigs.copy()
        self.assertEqual(len(dogs.tasks), 2, 'project: duplicating a project must duplicate its tasks')

    @mute_logger('odoo.addons.mail.mail_thread')
    def test_task_process_without_stage(self):
        if False:
            while True:
                i = 10
        task = self.format_and_process(EMAIL_TPL, to='project+pigs@mydomain.com, valid.lelitre@agrolait.com', cc='valid.other@gmail.com', email_from='%s' % self.user_projectuser.email, subject='Frogs', msg_id='<1198923581.41972151344608186760.JavaMail@agrolait.com>', target_model='project.task')
        self.assertEqual(len(task), 1, 'project: message_process: a new project.task should have been created')
        self.assertIn(self.partner_2, task.message_partner_ids, 'Partner in message cc is not added as a task followers.')
        self.assertEqual(len(task.message_ids), 2, 'project: message_process: newly created task should have 2 messages: creation and email')
        self.assertEqual(task.message_ids[1].subtype_id.name, 'Task Opened', 'project: message_process: first message of new task should have Task Created subtype')
        self.assertEqual(task.message_ids[0].author_id, self.user_projectuser.partner_id, 'project: message_process: second message should be the one from Agrolait (partner failed)')
        self.assertEqual(task.message_ids[0].subject, 'Frogs', 'project: message_process: second message should be the one from Agrolait (subject failed)')
        self.assertEqual(task.name, 'Frogs', 'project_task: name should be the email subject')
        self.assertEqual(task.project_id.id, self.project_pigs.id, 'project_task: incorrect project')
        self.assertEqual(task.stage_id.sequence, False, "project_task: shouldn't have a stage, i.e. sequence=False")

    @mute_logger('odoo.addons.mail.mail_thread')
    def test_task_process_with_stages(self):
        if False:
            i = 10
            return i + 15
        task = self.format_and_process(EMAIL_TPL, to='project+goats@mydomain.com, valid.lelitre@agrolait.com', cc='valid.other@gmail.com', email_from='%s' % self.user_projectuser.email, subject='Cats', msg_id='<1198923581.41972151344608186760.JavaMail@agrolait.com>', target_model='project.task')
        self.assertEqual(len(task), 1, 'project: message_process: a new project.task should have been created')
        self.assertIn(self.partner_2, task.message_partner_ids, 'Partner in message cc is not added as a task followers.')
        self.assertEqual(len(task.message_ids), 2, 'project: message_process: newly created task should have 2 messages: creation and email')
        self.assertEqual(task.message_ids[1].subtype_id.name, 'Task Opened', 'project: message_process: first message of new task should have Task Created subtype')
        self.assertEqual(task.message_ids[0].author_id, self.user_projectuser.partner_id, 'project: message_process: second message should be the one from Agrolait (partner failed)')
        self.assertEqual(task.message_ids[0].subject, 'Cats', 'project: message_process: second message should be the one from Agrolait (subject failed)')
        self.assertEqual(task.name, 'Cats', 'project_task: name should be the email subject')
        self.assertEqual(task.project_id.id, self.project_goats.id, 'project_task: incorrect project')
        self.assertEqual(task.stage_id.sequence, 1, 'project_task: should have a stage with sequence=1')