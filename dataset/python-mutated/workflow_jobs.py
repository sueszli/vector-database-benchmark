from awxkit.api.pages import UnifiedJob
from awxkit.api.resources import resources
from . import page

class WorkflowJob(UnifiedJob):

    def __str__(self):
        if False:
            print('Hello World!')
        return super(UnifiedJob, self).__str__()

    def relaunch(self, payload={}):
        if False:
            i = 10
            return i + 15
        result = self.related.relaunch.post(payload)
        return self.walk(result.url)

    def failure_output_details(self):
        if False:
            while True:
                i = 10
        'Special implementation of this part of assert_status so that\n        workflow_job.assert_successful() will give a breakdown of failure\n        '
        node_list = self.related.workflow_nodes.get().results
        msg = '\nNode summary:'
        for node in node_list:
            msg += '\n{}: {}'.format(node.id, node.summary_fields.get('job'))
            for rel in ('failure_nodes', 'always_nodes', 'success_nodes'):
                val = getattr(node, rel, [])
                if val:
                    msg += ' {} {}'.format(rel, val)
        msg += '\n\nUnhandled individual job failures:\n'
        for node in node_list:
            if node.job and (not (node.failure_nodes or node.always_nodes)):
                job = node.related.job.get()
                try:
                    job.assert_successful()
                except Exception as e:
                    msg += str(e)
        return msg

    @property
    def result_stdout(self):
        if False:
            print('Hello World!')
        if 'result_stdout' not in self.json:
            return 'Unprovided AWX field.'
        else:
            return super(WorkflowJob, self).result_stdout
page.register_page(resources.workflow_job, WorkflowJob)

class WorkflowJobs(page.PageList, WorkflowJob):
    pass
page.register_page([resources.workflow_jobs, resources.workflow_job_template_jobs, resources.job_template_slice_workflow_jobs], WorkflowJobs)