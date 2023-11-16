from awxkit.api.pages import UnifiedJob
from awxkit.api.resources import resources
from . import page
from awxkit import exceptions

class WorkflowApproval(UnifiedJob):

    def approve(self):
        if False:
            while True:
                i = 10
        try:
            self.related.approve.post()
        except exceptions.NoContent:
            pass

    def deny(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.related.deny.post()
        except exceptions.NoContent:
            pass
page.register_page(resources.workflow_approval, WorkflowApproval)

class WorkflowApprovals(page.PageList, WorkflowApproval):
    pass
page.register_page(resources.workflow_approvals, WorkflowApprovals)