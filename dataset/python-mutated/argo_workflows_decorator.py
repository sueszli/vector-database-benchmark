import json
import os
import time
from metaflow import current
from metaflow.decorators import StepDecorator
from metaflow.events import Trigger
from metaflow.metadata import MetaDatum
from metaflow.metaflow_config import ARGO_EVENTS_WEBHOOK_URL
from .argo_events import ArgoEvent

class ArgoWorkflowsInternalDecorator(StepDecorator):
    name = 'argo_workflows_internal'
    defaults = {'auto-emit-argo-events': True}

    def task_pre_step(self, step_name, task_datastore, metadata, run_id, task_id, flow, graph, retry_count, max_user_code_retries, ubf_context, inputs):
        if False:
            while True:
                i = 10
        self.task_id = task_id
        self.run_id = run_id
        triggers = []
        for (key, payload) in os.environ.items():
            if key.startswith('METAFLOW_ARGO_EVENT_PAYLOAD_'):
                if payload != 'null':
                    try:
                        payload = json.loads(payload)
                    except (TypeError, ValueError) as e:
                        payload = {}
                    triggers.append({'timestamp': payload.get('timestamp'), 'id': payload.get('id'), 'name': payload.get('name'), 'type': key[len('METAFLOW_ARGO_EVENT_PAYLOAD_'):].split('_', 1)[0]})
        meta = {}
        if triggers:
            current._update_env({'trigger': Trigger(triggers)})
            if step_name == 'start':
                meta['execution-triggers'] = json.dumps(triggers)
        meta['argo-workflow-template'] = os.environ['ARGO_WORKFLOW_TEMPLATE']
        meta['argo-workflow-name'] = os.environ['ARGO_WORKFLOW_NAME']
        meta['argo-workflow-namespace'] = os.environ['ARGO_WORKFLOW_NAMESPACE']
        meta['auto-emit-argo-events'] = self.attributes['auto-emit-argo-events']
        entries = [MetaDatum(field=k, value=v, type=k, tags=['attempt_id:{0}'.format(retry_count)]) for (k, v) in meta.items()]
        metadata.register_metadata(run_id, step_name, task_id, entries)

    def task_finished(self, step_name, flow, graph, is_task_ok, retry_count, max_user_code_retries):
        if False:
            print('Hello World!')
        if not is_task_ok:
            return
        if graph[step_name].type == 'foreach':
            with open('/mnt/out/splits', 'w') as file:
                json.dump(list(range(flow._foreach_num_splits)), file)
        with open('/mnt/out/task_id', 'w') as file:
            file.write(self.task_id)
        if self.attributes['auto-emit-argo-events']:
            event = ArgoEvent(name='metaflow.%s.%s' % (current.get('project_flow_name', flow.name), step_name))
            event.add_to_payload('id', current.pathspec)
            event.add_to_payload('pathspec', current.pathspec)
            event.add_to_payload('flow_name', flow.name)
            event.add_to_payload('run_id', self.run_id)
            event.add_to_payload('step_name', step_name)
            event.add_to_payload('task_id', self.task_id)
            for key in ('project_name', 'branch_name', 'is_user_branch', 'is_production', 'project_flow_name'):
                if current.get(key):
                    event.add_to_payload(key, current.get(key))
            event.add_to_payload('auto-generated-by-metaflow', True)
            event.safe_publish(ignore_errors=True)