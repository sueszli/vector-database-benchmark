from contextlib import suppress
import awxkit.exceptions as exc
from awxkit.api.pages import base, WorkflowJobTemplate, UnifiedJobTemplate, JobTemplate
from awxkit.api.mixins import HasCreate, DSAdapter
from awxkit.api.resources import resources
from awxkit.utils import update_payload, PseudoNamespace, random_title
from . import page

class WorkflowJobTemplateNode(HasCreate, base.Base):
    dependencies = [WorkflowJobTemplate, UnifiedJobTemplate]
    NATURAL_KEY = ('workflow_job_template', 'identifier')

    def payload(self, workflow_job_template, unified_job_template, **kwargs):
        if False:
            return 10
        if not unified_job_template:
            payload = PseudoNamespace(workflow_job_template=workflow_job_template.id)
        else:
            payload = PseudoNamespace(workflow_job_template=workflow_job_template.id, unified_job_template=unified_job_template.id)
        optional_fields = ('diff_mode', 'extra_data', 'limit', 'scm_branch', 'job_tags', 'job_type', 'skip_tags', 'verbosity', 'extra_data', 'identifier', 'all_parents_must_converge', 'job_slice_count', 'forks', 'timeout', 'execution_environment')
        update_payload(payload, optional_fields, kwargs)
        if 'inventory' in kwargs:
            payload['inventory'] = kwargs['inventory'].id
        return payload

    def create_payload(self, workflow_job_template=WorkflowJobTemplate, unified_job_template=JobTemplate, **kwargs):
        if False:
            print('Hello World!')
        if not unified_job_template:
            self.create_and_update_dependencies(workflow_job_template)
            payload = self.payload(workflow_job_template=self.ds.workflow_job_template, unified_job_template=None, **kwargs)
        else:
            self.create_and_update_dependencies(workflow_job_template, unified_job_template)
            payload = self.payload(workflow_job_template=self.ds.workflow_job_template, unified_job_template=self.ds.unified_job_template, **kwargs)
        payload.ds = DSAdapter(self.__class__.__name__, self._dependency_store)
        return payload

    def create(self, workflow_job_template=WorkflowJobTemplate, unified_job_template=JobTemplate, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        payload = self.create_payload(workflow_job_template=workflow_job_template, unified_job_template=unified_job_template, **kwargs)
        return self.update_identity(WorkflowJobTemplateNodes(self.connection).post(payload))

    def _add_node(self, endpoint, unified_job_template, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        node = endpoint.post(dict(unified_job_template=unified_job_template.id, **kwargs))
        node.create_and_update_dependencies(self.ds.workflow_job_template, unified_job_template)
        return node

    def add_always_node(self, unified_job_template, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return self._add_node(self.related.always_nodes, unified_job_template, **kwargs)

    def add_failure_node(self, unified_job_template, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return self._add_node(self.related.failure_nodes, unified_job_template, **kwargs)

    def add_success_node(self, unified_job_template, **kwargs):
        if False:
            while True:
                i = 10
        return self._add_node(self.related.success_nodes, unified_job_template, **kwargs)

    def add_credential(self, credential):
        if False:
            print('Hello World!')
        with suppress(exc.NoContent):
            self.related.credentials.post(dict(id=credential.id, associate=True))

    def remove_credential(self, credential):
        if False:
            return 10
        with suppress(exc.NoContent):
            self.related.credentials.post(dict(id=credential.id, disassociate=True))

    def remove_all_credentials(self):
        if False:
            print('Hello World!')
        for cred in self.related.credentials.get().results:
            with suppress(exc.NoContent):
                self.related.credentials.post(dict(id=cred.id, disassociate=True))

    def make_approval_node(self, **kwargs):
        if False:
            i = 10
            return i + 15
        if 'name' not in kwargs:
            kwargs['name'] = 'approval node {}'.format(random_title())
        self.related.create_approval_template.post(kwargs)
        return self.get()

    def get_job_node(self, workflow_job):
        if False:
            i = 10
            return i + 15
        candidates = workflow_job.get_related('workflow_nodes', identifier=self.identifier)
        return candidates.results.pop()

    def add_label(self, label):
        if False:
            for i in range(10):
                print('nop')
        with suppress(exc.NoContent):
            self.related.labels.post(dict(id=label.id))

    def add_instance_group(self, instance_group):
        if False:
            return 10
        with suppress(exc.NoContent):
            self.related.instance_groups.post(dict(id=instance_group.id))
page.register_page([resources.workflow_job_template_node, (resources.workflow_job_template_nodes, 'post'), (resources.workflow_job_template_workflow_nodes, 'post')], WorkflowJobTemplateNode)

class WorkflowJobTemplateNodes(page.PageList, WorkflowJobTemplateNode):
    pass
page.register_page([resources.workflow_job_template_nodes, resources.workflow_job_template_workflow_nodes, resources.workflow_job_template_node_always_nodes, resources.workflow_job_template_node_failure_nodes, resources.workflow_job_template_node_success_nodes], WorkflowJobTemplateNodes)