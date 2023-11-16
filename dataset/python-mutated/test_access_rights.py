from odoo.addons.project.tests.test_access_rights import TestPortalProjectBase
from odoo.exceptions import AccessError
from odoo.tools import mute_logger

class TestPortalProject(TestPortalProjectBase):

    @mute_logger('odoo.addons.base.ir.ir_model')
    def test_portal_project_access_rights(self):
        if False:
            for i in range(10):
                print('nop')
        pigs = self.project_pigs
        pigs.write({'privacy_visibility': 'portal'})
        pigs.sudo(self.user_projectuser).read(['user_id'])
        tasks = self.env['project.task'].sudo(self.user_projectuser).search([('project_id', '=', pigs.id)])
        self.assertEqual(tasks, self.task_1 | self.task_2 | self.task_3 | self.task_4 | self.task_5 | self.task_6, 'access rights: project user should see all tasks of a portal project')
        self.assertRaises(AccessError, pigs.sudo(self.user_noone).read, ['user_id'])
        self.assertRaises(AccessError, self.env['project.task'].sudo(self.user_noone).search, [('project_id', '=', pigs.id)])
        pigs.sudo(self.user_projectmanager).message_subscribe_users(user_ids=[self.user_portal.id])
        self.task_1.sudo(self.user_projectuser).message_subscribe_users(user_ids=[self.user_portal.id])
        self.task_3.sudo(self.user_projectuser).message_subscribe_users(user_ids=[self.user_portal.id])
        pigs.sudo(self.user_portal).read(['user_id'])
        self.assertRaises(AccessError, pigs.sudo(self.user_public).read, ['user_id'])
        self.assertRaises(AccessError, self.env['project.task'].sudo(self.user_public).search, [])
        self.task_1.sudo(self.user_projectuser).message_unsubscribe_users(user_ids=[self.user_portal.id])
        self.task_3.sudo(self.user_projectuser).message_unsubscribe_users(user_ids=[self.user_portal.id])