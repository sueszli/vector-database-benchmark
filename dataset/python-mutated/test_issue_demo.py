from odoo.addons.project_issue.tests.common import TestIssueUsers

class TestIssueDemo(TestIssueUsers):

    def test_issue_demo(self):
        if False:
            i = 10
            return i + 15
        self.ProjectIssue.sudo(self.project_manager.id).create({'name': 'Error in account module', 'task_id': self.ref('project.project_task_17')})
        self.ProjectIssue.sudo(self.project_manager.id).create({'name': 'Odoo Integration', 'project_id': self.ref('project.project_project_2')})