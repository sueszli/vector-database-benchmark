from awx.main.models import Job, ProjectUpdate, InventoryUpdate, SystemJob, AdHocCommand, WorkflowJob
import logging
logger = logging.getLogger('awx.main.scheduler.dependency_graph')

class DependencyGraph(object):
    PROJECT_UPDATES = 'project_updates'
    INVENTORY_UPDATES = 'inventory_updates'
    JOB_TEMPLATE_JOBS = 'job_template_jobs'
    SYSTEM_JOB = 'system_job'
    INVENTORY_SOURCE_UPDATES = 'inventory_source_updates'
    WORKFLOW_JOB_TEMPLATES_JOBS = 'workflow_job_template_jobs'
    INVENTORY_SOURCES = 'inventory_source_ids'

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.data = {}
        self.data[self.PROJECT_UPDATES] = {}
        self.data[self.INVENTORY_UPDATES] = {}
        self.data[self.INVENTORY_SOURCE_UPDATES] = {}
        self.data[self.JOB_TEMPLATE_JOBS] = {}
        self.data[self.SYSTEM_JOB] = {}
        self.data[self.WORKFLOW_JOB_TEMPLATES_JOBS] = {}

    def mark_if_no_key(self, job_type, id, job):
        if False:
            return 10
        if id is None:
            logger.warning(f'Null dependency graph key from {job}, could be integrity error or bug, ignoring')
            return
        if id not in self.data[job_type]:
            self.data[job_type][id] = job

    def get_item(self, job_type, id):
        if False:
            i = 10
            return i + 15
        return self.data[job_type].get(id, None)

    def mark_system_job(self, job):
        if False:
            print('Hello World!')
        self.mark_if_no_key(self.SYSTEM_JOB, 'system_job', job)

    def mark_project_update(self, job):
        if False:
            return 10
        self.mark_if_no_key(self.PROJECT_UPDATES, job.project_id, job)

    def mark_inventory_update(self, job):
        if False:
            while True:
                i = 10
        if type(job) is AdHocCommand:
            self.mark_if_no_key(self.INVENTORY_UPDATES, job.inventory_id, job)
        else:
            self.mark_if_no_key(self.INVENTORY_UPDATES, job.inventory_source.inventory_id, job)

    def mark_inventory_source_update(self, job):
        if False:
            return 10
        self.mark_if_no_key(self.INVENTORY_SOURCE_UPDATES, job.inventory_source_id, job)

    def mark_job_template_job(self, job):
        if False:
            i = 10
            return i + 15
        self.mark_if_no_key(self.JOB_TEMPLATE_JOBS, job.job_template_id, job)

    def mark_workflow_job(self, job):
        if False:
            for i in range(10):
                print('nop')
        if job.workflow_job_template_id:
            self.mark_if_no_key(self.WORKFLOW_JOB_TEMPLATES_JOBS, job.workflow_job_template_id, job)
        elif job.unified_job_template_id:
            self.mark_if_no_key(self.WORKFLOW_JOB_TEMPLATES_JOBS, job.unified_job_template_id, job)

    def project_update_blocked_by(self, job):
        if False:
            return 10
        return self.get_item(self.PROJECT_UPDATES, job.project_id)

    def inventory_update_blocked_by(self, job):
        if False:
            i = 10
            return i + 15
        return self.get_item(self.INVENTORY_SOURCE_UPDATES, job.inventory_source_id)

    def job_blocked_by(self, job):
        if False:
            i = 10
            return i + 15
        project_block = self.get_item(self.PROJECT_UPDATES, job.project_id)
        inventory_block = self.get_item(self.INVENTORY_UPDATES, job.inventory_id)
        if job.allow_simultaneous is False:
            job_block = self.get_item(self.JOB_TEMPLATE_JOBS, job.job_template_id)
        else:
            job_block = None
        return project_block or inventory_block or job_block

    def workflow_job_blocked_by(self, job):
        if False:
            print('Hello World!')
        if job.allow_simultaneous is False:
            if job.workflow_job_template_id:
                return self.get_item(self.WORKFLOW_JOB_TEMPLATES_JOBS, job.workflow_job_template_id)
            elif job.unified_job_template_id:
                return self.get_item(self.WORKFLOW_JOB_TEMPLATES_JOBS, job.unified_job_template_id) or self.get_item(self.JOB_TEMPLATE_JOBS, job.unified_job_template_id)
        return None

    def system_job_blocked_by(self, job):
        if False:
            for i in range(10):
                print('nop')
        return self.get_item(self.SYSTEM_JOB, 'system_job')

    def ad_hoc_command_blocked_by(self, job):
        if False:
            i = 10
            return i + 15
        return self.get_item(self.INVENTORY_UPDATES, job.inventory_id)

    def task_blocked_by(self, job):
        if False:
            while True:
                i = 10
        if type(job) is ProjectUpdate:
            return self.project_update_blocked_by(job)
        elif type(job) is InventoryUpdate:
            return self.inventory_update_blocked_by(job)
        elif type(job) is Job:
            return self.job_blocked_by(job)
        elif type(job) is SystemJob:
            return self.system_job_blocked_by(job)
        elif type(job) is AdHocCommand:
            return self.ad_hoc_command_blocked_by(job)
        elif type(job) is WorkflowJob:
            return self.workflow_job_blocked_by(job)

    def add_job(self, job):
        if False:
            return 10
        if type(job) is ProjectUpdate:
            self.mark_project_update(job)
        elif type(job) is InventoryUpdate:
            self.mark_inventory_update(job)
            self.mark_inventory_source_update(job)
        elif type(job) is Job:
            self.mark_job_template_job(job)
        elif type(job) is WorkflowJob:
            self.mark_workflow_job(job)
        elif type(job) is SystemJob:
            self.mark_system_job(job)
        elif type(job) is AdHocCommand:
            self.mark_inventory_update(job)

    def add_jobs(self, jobs):
        if False:
            for i in range(10):
                print('nop')
        for j in jobs:
            self.add_job(j)