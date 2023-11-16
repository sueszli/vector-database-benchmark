import base64
import json
import os
import re
import shlex
import sys
from collections import defaultdict
from hashlib import sha1
from metaflow import JSONType, current
from metaflow.decorators import flow_decorators
from metaflow.exception import MetaflowException
from metaflow.includefile import FilePathClass
from metaflow.metaflow_config import ARGO_EVENTS_EVENT, ARGO_EVENTS_EVENT_BUS, ARGO_EVENTS_EVENT_SOURCE, ARGO_EVENTS_INTERNAL_WEBHOOK_URL, ARGO_EVENTS_SERVICE_ACCOUNT, ARGO_EVENTS_WEBHOOK_AUTH, ARGO_WORKFLOWS_ENV_VARS_TO_SKIP, ARGO_WORKFLOWS_KUBERNETES_SECRETS, ARGO_WORKFLOWS_UI_URL, AWS_SECRETS_MANAGER_DEFAULT_REGION, AZURE_STORAGE_BLOB_SERVICE_ENDPOINT, CARD_AZUREROOT, CARD_GSROOT, CARD_S3ROOT, DATASTORE_SYSROOT_AZURE, DATASTORE_SYSROOT_GS, DATASTORE_SYSROOT_S3, DATATOOLS_S3ROOT, DEFAULT_METADATA, DEFAULT_SECRETS_BACKEND_TYPE, KUBERNETES_FETCH_EC2_METADATA, KUBERNETES_LABELS, KUBERNETES_NAMESPACE, KUBERNETES_NODE_SELECTOR, KUBERNETES_SANDBOX_INIT_SCRIPT, KUBERNETES_SECRETS, S3_ENDPOINT_URL, S3_SERVER_SIDE_ENCRYPTION, SERVICE_HEADERS, SERVICE_INTERNAL_URL, UI_URL
from metaflow.metaflow_config_funcs import config_values
from metaflow.mflog import BASH_SAVE_LOGS, bash_capture_logs, export_mflog_env_vars
from metaflow.parameters import deploy_time_eval
from metaflow.plugins.kubernetes.kubernetes import parse_kube_keyvalue_list, validate_kube_labels
from metaflow.util import compress_list, dict_to_cli_options, to_bytes, to_camelcase, to_unicode
from .argo_client import ArgoClient

class ArgoWorkflowsException(MetaflowException):
    headline = 'Argo Workflows error'

class ArgoWorkflowsSchedulingException(MetaflowException):
    headline = 'Argo Workflows scheduling error'

class ArgoWorkflows(object):

    def __init__(self, name, graph, flow, code_package_sha, code_package_url, production_token, metadata, flow_datastore, environment, event_logger, monitor, tags=None, namespace=None, username=None, max_workers=None, workflow_timeout=None, workflow_priority=None, auto_emit_argo_events=False, notify_on_error=False, notify_on_success=False, notify_slack_webhook_url=None, notify_pager_duty_integration_key=None):
        if False:
            return 10
        self.name = name
        self.graph = graph
        self.flow = flow
        self.code_package_sha = code_package_sha
        self.code_package_url = code_package_url
        self.production_token = production_token
        self.metadata = metadata
        self.flow_datastore = flow_datastore
        self.environment = environment
        self.event_logger = event_logger
        self.monitor = monitor
        self.tags = tags
        self.namespace = namespace
        self.username = username
        self.max_workers = max_workers
        self.workflow_timeout = workflow_timeout
        self.workflow_priority = workflow_priority
        self.auto_emit_argo_events = auto_emit_argo_events
        self.notify_on_error = notify_on_error
        self.notify_on_success = notify_on_success
        self.notify_slack_webhook_url = notify_slack_webhook_url
        self.notify_pager_duty_integration_key = notify_pager_duty_integration_key
        self.parameters = self._process_parameters()
        (self.triggers, self.trigger_options) = self._process_triggers()
        (self._schedule, self._timezone) = self._get_schedule()
        self.kubernetes_labels = self._get_kubernetes_labels()
        self._workflow_template = self._compile_workflow_template()
        self._sensor = self._compile_sensor()

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return str(self._workflow_template)

    def deploy(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            ArgoClient(namespace=KUBERNETES_NAMESPACE).register_workflow_template(self.name, self._workflow_template.to_json())
        except Exception as e:
            raise ArgoWorkflowsException(str(e))

    @staticmethod
    def _sanitize(name):
        if False:
            i = 10
            return i + 15
        return name.replace('_', '-')

    @staticmethod
    def list_templates(flow_name, all=False):
        if False:
            while True:
                i = 10
        client = ArgoClient(namespace=KUBERNETES_NAMESPACE)
        templates = client.get_workflow_templates()
        if templates is None:
            return []
        template_names = [template['metadata']['name'] for template in templates if all or flow_name == template['metadata'].get('annotations', {}).get('metaflow/flow_name', None)]
        return template_names

    @staticmethod
    def delete(name):
        if False:
            for i in range(10):
                print('nop')
        client = ArgoClient(namespace=KUBERNETES_NAMESPACE)
        schedule_deleted = client.delete_cronworkflow(name)
        sensor_deleted = client.delete_sensor(name)
        workflow_deleted = client.delete_workflow_template(name)
        if workflow_deleted is None:
            raise ArgoWorkflowsException("The workflow *%s* doesn't exist on Argo Workflows." % name)
        return (schedule_deleted, sensor_deleted, workflow_deleted)

    @staticmethod
    def terminate(flow_name, name):
        if False:
            print('Hello World!')
        client = ArgoClient(namespace=KUBERNETES_NAMESPACE)
        response = client.terminate_workflow(name)
        if response is None:
            raise ArgoWorkflowsException('No execution found for {flow_name}/{run_id} in Argo Workflows.'.format(flow_name=flow_name, run_id=name))

    @staticmethod
    def get_workflow_status(flow_name, name):
        if False:
            while True:
                i = 10
        client = ArgoClient(namespace=KUBERNETES_NAMESPACE)
        workflow = client.get_workflow(name)
        if workflow:
            status = workflow.get('status', {}).get('phase')
            return status
        else:
            raise ArgoWorkflowsException('No execution found for {flow_name}/{run_id} in Argo Workflows.'.format(flow_name=flow_name, run_id=name))

    @staticmethod
    def suspend(name):
        if False:
            while True:
                i = 10
        client = ArgoClient(namespace=KUBERNETES_NAMESPACE)
        client.suspend_workflow(name)
        return True

    @staticmethod
    def unsuspend(name):
        if False:
            print('Hello World!')
        client = ArgoClient(namespace=KUBERNETES_NAMESPACE)
        client.unsuspend_workflow(name)
        return True

    @classmethod
    def trigger(cls, name, parameters=None):
        if False:
            while True:
                i = 10
        if parameters is None:
            parameters = {}
        try:
            workflow_template = ArgoClient(namespace=KUBERNETES_NAMESPACE).get_workflow_template(name)
        except Exception as e:
            raise ArgoWorkflowsException(str(e))
        if workflow_template is None:
            raise ArgoWorkflowsException("The workflow *%s* doesn't exist on Argo Workflows in namespace *%s*. Please deploy your flow first." % (name, KUBERNETES_NAMESPACE))
        else:
            try:
                workflow_template['metadata']['annotations']['metaflow/owner']
            except KeyError as e:
                raise ArgoWorkflowsException('An existing non-metaflow workflow with the same name as *%s* already exists in Argo Workflows. \nPlease modify the name of this flow or delete your existing workflow on Argo Workflows before proceeding.' % name)
        try:
            return ArgoClient(namespace=KUBERNETES_NAMESPACE).trigger_workflow_template(name, parameters)
        except Exception as e:
            raise ArgoWorkflowsException(str(e))

    @staticmethod
    def _get_kubernetes_labels():
        if False:
            i = 10
            return i + 15
        '\n        Get Kubernetes labels from environment variable.\n        Parses the string into a dict and validates that values adhere to Kubernetes restrictions.\n        '
        if not KUBERNETES_LABELS:
            return {}
        env_labels = KUBERNETES_LABELS.split(',')
        env_labels = parse_kube_keyvalue_list(env_labels, False)
        validate_kube_labels(env_labels)
        return env_labels

    def _get_schedule(self):
        if False:
            print('Hello World!')
        schedule = self.flow._flow_decorators.get('schedule')
        if schedule:
            schedule = schedule[0]
            return (' '.join(schedule.schedule.split()[:5]), schedule.timezone)
        return (None, None)

    def schedule(self):
        if False:
            i = 10
            return i + 15
        try:
            argo_client = ArgoClient(namespace=KUBERNETES_NAMESPACE)
            argo_client.schedule_workflow_template(self.name, self._schedule, self._timezone)
            sensor_name = self.name.replace('.', '-')
            if self._sensor:
                argo_client.register_sensor(sensor_name, self._sensor.to_json())
            else:
                argo_client.delete_sensor(sensor_name)
        except Exception as e:
            raise ArgoWorkflowsSchedulingException(str(e))

    def trigger_explanation(self):
        if False:
            print('Hello World!')
        if self.flow._flow_decorators.get('schedule'):
            return 'This workflow triggers automatically via the CronWorkflow *%s*.' % self.name
        elif self.flow._flow_decorators.get('trigger'):
            return 'This workflow triggers automatically when the upstream %s is/are published.' % self.list_to_prose([event['name'] for event in self.triggers], 'event')
        elif self.flow._flow_decorators.get('trigger_on_finish'):
            return 'This workflow triggers automatically when the upstream %s succeed(s)' % self.list_to_prose([event['name'][len('metaflow.'):-len('.end')] for event in self.triggers], 'flow')
        else:
            return 'No triggers defined. You need to launch this workflow manually.'

    @classmethod
    def get_existing_deployment(cls, name):
        if False:
            i = 10
            return i + 15
        workflow_template = ArgoClient(namespace=KUBERNETES_NAMESPACE).get_workflow_template(name)
        if workflow_template is not None:
            try:
                return (workflow_template['metadata']['annotations']['metaflow/owner'], workflow_template['metadata']['annotations']['metaflow/production_token'])
            except KeyError as e:
                raise ArgoWorkflowsException('An existing non-metaflow workflow with the same name as *%s* already exists in Argo Workflows. \nPlease modify the name of this flow or delete your existing workflow on Argo Workflows before proceeding.' % name)
        return None

    @classmethod
    def get_execution(cls, name):
        if False:
            return 10
        workflow = ArgoClient(namespace=KUBERNETES_NAMESPACE).get_workflow(name)
        if workflow is not None:
            try:
                return (workflow['metadata']['annotations']['metaflow/owner'], workflow['metadata']['annotations']['metaflow/production_token'], workflow['metadata']['annotations']['metaflow/flow_name'], workflow['metadata']['annotations'].get('metaflow/branch_name', None), workflow['metadata']['annotations'].get('metaflow/project_name', None))
            except KeyError:
                raise ArgoWorkflowsException('A non-metaflow workflow *%s* already exists in Argo Workflows.' % name)
        return None

    def _process_parameters(self):
        if False:
            while True:
                i = 10
        parameters = {}
        has_schedule = self.flow._flow_decorators.get('schedule') is not None
        seen = set()
        for (var, param) in self.flow._get_parameters():
            norm = param.name.lower()
            if norm in seen:
                raise MetaflowException('Parameter *%s* is specified twice. Note that parameter names are case-insensitive.' % param.name)
            seen.add(norm)
            if param.kwargs.get('type') == JSONType or isinstance(param.kwargs.get('type'), FilePathClass):
                param_type = str(param.kwargs.get('type').name)
            else:
                param_type = str(param.kwargs.get('type').__name__)
            is_required = param.kwargs.get('required', False)
            if 'default' not in param.kwargs and is_required and has_schedule:
                raise MetaflowException('The parameter *%s* does not have a default and is required. Scheduling such parameters via Argo CronWorkflows is not currently supported.' % param.name)
            default_value = deploy_time_eval(param.kwargs.get('default'))
            if not is_required or default_value is not None:
                default_value = json.dumps(default_value)
            parameters[param.name] = dict(name=param.name, value=default_value, type=param_type, description=param.kwargs.get('help'), is_required=is_required)
        return parameters

    def _process_triggers(self):
        if False:
            i = 10
            return i + 15
        if self.flow._flow_decorators.get('trigger') and self.flow._flow_decorators.get('trigger_on_finish'):
            raise ArgoWorkflowsException("Argo Workflows doesn't support both *@trigger* and *@trigger_on_finish* decorators concurrently yet. Use one or the other for now.")
        triggers = []
        options = None
        if self.flow._flow_decorators.get('trigger'):
            seen = set()
            params = set([param.name.lower() for (var, param) in self.flow._get_parameters()])
            for event in self.flow._flow_decorators.get('trigger')[0].triggers:
                parameters = {}
                if not re.match('^[A-Za-z0-9_.-]+$', event['name']):
                    raise ArgoWorkflowsException('Invalid event name *%s* in *@trigger* decorator. Only alphanumeric characters, underscores(_), dashes(-) and dots(.) are allowed.' % event['name'])
                for (key, value) in event.get('parameters', {}).items():
                    if not re.match('^[A-Za-z0-9_]+$', value):
                        raise ArgoWorkflowsException('Invalid event payload key *%s* for event *%s* in *@trigger* decorator. Only alphanumeric characters and underscores(_) are allowed.' % (value, event['name']))
                    if key.lower() not in params:
                        raise ArgoWorkflowsException('Parameter *%s* defined in the event mappings for *@trigger* decorator not found in the flow.' % key)
                    if key.lower() in seen:
                        raise ArgoWorkflowsException('Duplicate entries for parameter *%s* defined in the event mappings for *@trigger* decorator.' % key.lower())
                    seen.add(key.lower())
                    parameters[key.lower()] = value
                event['parameters'] = parameters
                event['type'] = 'event'
            triggers.extend(self.flow._flow_decorators.get('trigger')[0].triggers)
            if len(triggers) == 1 and (not triggers[0].get('parameters')):
                triggers[0]['parameters'] = dict(zip(params, params))
            options = self.flow._flow_decorators.get('trigger')[0].options
        if self.flow._flow_decorators.get('trigger_on_finish'):
            for event in self.flow._flow_decorators.get('trigger_on_finish')[0].triggers:
                triggers.append({'name': 'metaflow.%s.end' % '.'.join((v for v in [event.get('project') or current.get('project_name'), event.get('branch') or current.get('branch_name'), event['flow']] if v)), 'filters': {'auto-generated-by-metaflow': True, 'project_name': event.get('project') or current.get('project_name'), 'branch_name': event.get('branch') or current.get('branch_name')}, 'type': 'run', 'flow': event['flow']})
            options = self.flow._flow_decorators.get('trigger_on_finish')[0].options
        for event in triggers:
            event['sanitized_name'] = '%s_%s' % (event['name'].replace('.', '').replace('-', '').replace('@', '').replace('+', ''), to_unicode(base64.b32encode(sha1(to_bytes(event['name'])).digest()))[:4].lower())
        return (triggers, options)

    def _compile_workflow_template(self):
        if False:
            i = 10
            return i + 15
        annotations = {'metaflow/production_token': self.production_token, 'metaflow/owner': self.username, 'metaflow/user': 'argo-workflows', 'metaflow/flow_name': self.flow.name}
        if self.parameters:
            annotations.update({'metaflow/parameters': json.dumps(self.parameters)})
        if current.get('project_name'):
            annotations.update({'metaflow/project_name': current.project_name, 'metaflow/branch_name': current.branch_name, 'metaflow/project_flow_name': current.project_flow_name})
        if self.tags:
            annotations.update({'metaflow/tags': json.dumps(self.tags)})
        if self.triggers:
            annotations.update({'metaflow/triggers': json.dumps([{key: trigger.get(key) for key in ['name', 'type']} for trigger in self.triggers])})
        if self.notify_on_error:
            annotations.update({'metaflow/notify_on_error': json.dumps({'slack': bool(self.notify_slack_webhook_url), 'pager_duty': bool(self.notify_pager_duty_integration_key)})})
        if self.notify_on_success:
            annotations.update({'metaflow/notify_on_success': json.dumps({'slack': bool(self.notify_slack_webhook_url), 'pager_duty': bool(self.notify_pager_duty_integration_key)})})
        return WorkflowTemplate().metadata(ObjectMeta().name(self.name).namespace(KUBERNETES_NAMESPACE).label('app.kubernetes.io/name', 'metaflow-flow').label('app.kubernetes.io/part-of', 'metaflow').annotations(annotations)).spec(WorkflowSpec().active_deadline_seconds(self.workflow_timeout).parallelism(self.max_workers).priority(self.workflow_priority).workflow_metadata(Metadata().label('app.kubernetes.io/name', 'metaflow-run').label('app.kubernetes.io/part-of', 'metaflow').annotations({**annotations, **{'metaflow/run_id': 'argo-{{workflow.name}}'}})).arguments(Arguments().parameters([Parameter(parameter['name']).value(parameter['value']).description(parameter.get('description')) for parameter in self.parameters.values()] + [Parameter(event['sanitized_name']).value(json.dumps(None)).description('auto-set by metaflow. safe to ignore.') for event in self.triggers])).pod_metadata(Metadata().label('app.kubernetes.io/name', 'metaflow-task').label('app.kubernetes.io/part-of', 'metaflow').annotations(annotations).labels(self.kubernetes_labels)).entrypoint(self.flow.name).hooks({**({'notify-slack-on-success': LifecycleHook().expression("workflow.status == 'Succeeded'").template('notify-slack-on-success')} if self.notify_on_success and self.notify_slack_webhook_url else {}), **({'notify-pager-duty-on-success': LifecycleHook().expression("workflow.status == 'Succeeded'").template('notify-pager-duty-on-success')} if self.notify_on_success and self.notify_pager_duty_integration_key else {}), **({'notify-slack-on-failure': LifecycleHook().expression("workflow.status == 'Failed'").template('notify-slack-on-error'), 'notify-slack-on-error': LifecycleHook().expression("workflow.status == 'Error'").template('notify-slack-on-error')} if self.notify_on_error and self.notify_slack_webhook_url else {}), **({'notify-pager-duty-on-failure': LifecycleHook().expression("workflow.status == 'Failed'").template('notify-pager-duty-on-error'), 'notify-pager-duty-on-error': LifecycleHook().expression("workflow.status == 'Error'").template('notify-pager-duty-on-error')} if self.notify_on_error and self.notify_pager_duty_integration_key else {}), **({'exit': LifecycleHook().template('exit-hook-hack')} if self.notify_on_error or self.notify_on_success else {})}).templates(self._dag_templates()).templates(self._container_templates()).templates(self._exit_hook_templates()))

    def _dag_templates(self):
        if False:
            return 10

        def _visit(node, exit_node=None, templates=None, dag_tasks=None):
            if False:
                return 10
            if dag_tasks is None:
                dag_tasks = []
            if templates is None:
                templates = []
            if exit_node is not None and exit_node is node.name:
                return (templates, dag_tasks)
            if node.name == 'start':
                dag_task = DAGTask(self._sanitize(node.name)).template(self._sanitize(node.name))
            elif node.is_inside_foreach and self.graph[node.in_funcs[0]].type == 'foreach':
                parameters = [Parameter('input-paths').value('{{inputs.parameters.input-paths}}'), Parameter('split-index').value('{{inputs.parameters.split-index}}')]
                dag_task = DAGTask(self._sanitize(node.name)).template(self._sanitize(node.name)).arguments(Arguments().parameters(parameters))
            else:
                parameters = [Parameter('input-paths').value(compress_list(['argo-{{workflow.name}}/%s/{{tasks.%s.outputs.parameters.task-id}}' % (n, self._sanitize(n)) for n in node.in_funcs]))]
                dag_task = DAGTask(self._sanitize(node.name)).dependencies([self._sanitize(in_func) for in_func in node.in_funcs]).template(self._sanitize(node.name)).arguments(Arguments().parameters(parameters))
            dag_tasks.append(dag_task)
            if node.type == 'end':
                return ([Template(self.flow.name).dag(DAGTemplate().fail_fast().tasks(dag_tasks))] + templates, dag_tasks)
            if node.type == 'split':
                for n in node.out_funcs:
                    _visit(self.graph[n], node.matching_join, templates, dag_tasks)
                return _visit(self.graph[node.matching_join], exit_node, templates, dag_tasks)
            elif node.type == 'foreach':
                foreach_template_name = self._sanitize('%s-foreach-%s' % (node.name, node.foreach_param))
                foreach_task = DAGTask(foreach_template_name).dependencies([self._sanitize(node.name)]).template(foreach_template_name).arguments(Arguments().parameters([Parameter('input-paths').value('argo-{{workflow.name}}/%s/{{tasks.%s.outputs.parameters.task-id}}' % (node.name, self._sanitize(node.name))), Parameter('split-index').value('{{item}}')])).with_param('{{tasks.%s.outputs.parameters.num-splits}}' % self._sanitize(node.name))
                dag_tasks.append(foreach_task)
                (templates, dag_tasks_1) = _visit(self.graph[node.out_funcs[0]], node.matching_join, templates, [])
                templates.append(Template(foreach_template_name).inputs(Inputs().parameters([Parameter('input-paths'), Parameter('split-index')])).outputs(Outputs().parameters([Parameter('task-id').valueFrom({'parameter': '{{tasks.%s.outputs.parameters.task-id}}' % self._sanitize(self.graph[node.matching_join].in_funcs[0])})])).dag(DAGTemplate().fail_fast().tasks(dag_tasks_1)))
                join_foreach_task = DAGTask(self._sanitize(self.graph[node.matching_join].name)).template(self._sanitize(self.graph[node.matching_join].name)).dependencies([foreach_template_name]).arguments(Arguments().parameters([Parameter('input-paths').value('argo-{{workflow.name}}/%s/{{tasks.%s.outputs.parameters}}' % (self.graph[node.matching_join].in_funcs[-1], foreach_template_name))]))
                dag_tasks.append(join_foreach_task)
                return _visit(self.graph[self.graph[node.matching_join].out_funcs[0]], exit_node, templates, dag_tasks)
            if node.type in ('linear', 'join', 'start'):
                return _visit(self.graph[node.out_funcs[0]], exit_node, templates, dag_tasks)
            else:
                raise ArgoWorkflowsException('Node type *%s* for step *%s* is not currently supported by Argo Workflows.' % (node.type, node.name))
        (templates, _) = _visit(node=self.graph['start'])
        return templates

    def _container_templates(self):
        if False:
            return 10
        try:
            from kubernetes import client as kubernetes_sdk
        except (NameError, ImportError):
            raise MetaflowException("Could not import Python package 'kubernetes'. Install kubernetes sdk (https://pypi.org/project/kubernetes/) first.")
        for node in self.graph:
            script_name = os.path.basename(sys.argv[0])
            executable = self.environment.executable(node.name)
            entrypoint = [executable, script_name]
            run_id = 'argo-{{workflow.name}}'
            task_str = node.name + '-{{workflow.creationTimestamp}}'
            if node.name != 'start':
                task_str += '-{{inputs.parameters.input-paths}}'
            if any((self.graph[n].type == 'foreach' for n in node.in_funcs)):
                task_str += '-{{inputs.parameters.split-index}}'
            task_id_expr = "export METAFLOW_TASK_ID=(t-$(echo %s | md5sum | cut -d ' ' -f 1 | tail -c 9))" % task_str
            task_id = '$METAFLOW_TASK_ID'
            max_user_code_retries = 0
            max_error_retries = 0
            minutes_between_retries = '2'
            for decorator in node.decorators:
                if decorator.name == 'retry':
                    minutes_between_retries = decorator.attributes.get('minutes_between_retries', minutes_between_retries)
                (user_code_retries, error_retries) = decorator.step_task_retry_count()
                max_user_code_retries = max(max_user_code_retries, user_code_retries)
                max_error_retries = max(max_error_retries, error_retries)
            user_code_retries = max_user_code_retries
            total_retries = max_user_code_retries + max_error_retries
            retry_count = '{{retries}}' if max_user_code_retries + max_error_retries else 0
            minutes_between_retries = int(minutes_between_retries)
            mflog_expr = export_mflog_env_vars(datastore_type=self.flow_datastore.TYPE, stdout_path='$PWD/.logs/mflog_stdout', stderr_path='$PWD/.logs/mflog_stderr', flow_name=self.flow.name, run_id=run_id, step_name=node.name, task_id=task_id, retry_count=retry_count)
            init_cmds = ' && '.join(['${METAFLOW_INIT_SCRIPT:+eval \\"${METAFLOW_INIT_SCRIPT}\\"}', 'mkdir -p $PWD/.logs', task_id_expr, mflog_expr] + self.environment.get_package_commands(self.code_package_url, self.flow_datastore.TYPE))
            step_cmds = self.environment.bootstrap_commands(node.name, self.flow_datastore.TYPE)
            input_paths = '{{inputs.parameters.input-paths}}'
            top_opts_dict = {'with': [decorator.make_decorator_spec() for decorator in node.decorators if not decorator.statically_defined]}
            for deco in flow_decorators():
                top_opts_dict.update(deco.get_top_level_options())
            top_level = list(dict_to_cli_options(top_opts_dict)) + ['--quiet', '--metadata=%s' % self.metadata.TYPE, '--environment=%s' % self.environment.TYPE, '--datastore=%s' % self.flow_datastore.TYPE, '--datastore-root=%s' % self.flow_datastore.datastore_root, '--event-logger=%s' % self.event_logger.TYPE, '--monitor=%s' % self.monitor.TYPE, '--no-pylint', '--with=argo_workflows_internal:auto-emit-argo-events=%i' % self.auto_emit_argo_events]
            if node.name == 'start':
                task_id_params = '%s-params' % task_id
                init = entrypoint + top_level + ['init', '--run-id %s' % run_id, '--task-id %s' % task_id_params] + ['--%s={{workflow.parameters.%s}}' % (parameter['name'], parameter['name']) for parameter in self.parameters.values()]
                if self.tags:
                    init.extend(('--tag %s' % tag for tag in self.tags))
                exists = entrypoint + ['dump', '--max-value-size=0', '%s/_parameters/%s' % (run_id, task_id_params)]
                step_cmds.extend(['if ! %s >/dev/null 2>/dev/null; then %s; fi' % (' '.join(exists), ' '.join(init))])
                input_paths = '%s/_parameters/%s' % (run_id, task_id_params)
            elif node.type == 'join' and self.graph[node.split_parents[-1]].type == 'foreach':
                input_paths = '$(python -m metaflow.plugins.argo.process_input_paths %s)' % input_paths
            step = ['step', node.name, '--run-id %s' % run_id, '--task-id %s' % task_id, '--retry-count %s' % retry_count, '--max-user-code-retries %d' % user_code_retries, '--input-paths %s' % input_paths]
            if any((self.graph[n].type == 'foreach' for n in node.in_funcs)):
                step.append('--split-index {{inputs.parameters.split-index}}')
            if self.tags:
                step.extend(('--tag %s' % tag for tag in self.tags))
            if self.namespace is not None:
                step.append('--namespace=%s' % self.namespace)
            step_cmds.extend([' '.join(entrypoint + top_level + step)])
            cmd_str = '%s; c=$?; %s; exit $c' % (' && '.join([init_cmds, bash_capture_logs(' && '.join(step_cmds))]), BASH_SAVE_LOGS)
            cmds = shlex.split('bash -c "%s"' % cmd_str)
            resources = dict([deco for deco in node.decorators if deco.name == 'kubernetes'][0].attributes)
            if resources['namespace'] and resources['namespace'] != KUBERNETES_NAMESPACE:
                raise ArgoWorkflowsException('Multi-namespace Kubernetes execution of flows in Argo Workflows is not currently supported. \nStep *%s* is trying to override the default Kubernetes namespace *%s*.' % (node.name, KUBERNETES_NAMESPACE))
            run_time_limit = [deco for deco in node.decorators if deco.name == 'kubernetes'][0].run_time_limit
            env = dict([deco for deco in node.decorators if deco.name == 'environment'][0].attributes['vars'])
            env.update({k: v for (k, v) in config_values() if k.startswith('METAFLOW_CONDA_') or k.startswith('METAFLOW_DEBUG_')})
            env.update({**{'METAFLOW_CODE_URL': self.code_package_url, 'METAFLOW_CODE_SHA': self.code_package_sha, 'METAFLOW_CODE_DS': self.flow_datastore.TYPE, 'METAFLOW_SERVICE_URL': SERVICE_INTERNAL_URL, 'METAFLOW_SERVICE_HEADERS': json.dumps(SERVICE_HEADERS), 'METAFLOW_USER': 'argo-workflows', 'METAFLOW_DATASTORE_SYSROOT_S3': DATASTORE_SYSROOT_S3, 'METAFLOW_DATATOOLS_S3ROOT': DATATOOLS_S3ROOT, 'METAFLOW_DEFAULT_DATASTORE': self.flow_datastore.TYPE, 'METAFLOW_DEFAULT_METADATA': DEFAULT_METADATA, 'METAFLOW_CARD_S3ROOT': CARD_S3ROOT, 'METAFLOW_KUBERNETES_WORKLOAD': 1, 'METAFLOW_KUBERNETES_FETCH_EC2_METADATA': KUBERNETES_FETCH_EC2_METADATA, 'METAFLOW_RUNTIME_ENVIRONMENT': 'kubernetes', 'METAFLOW_OWNER': self.username}, **{'METAFLOW_ARGO_EVENTS_EVENT': ARGO_EVENTS_EVENT, 'METAFLOW_ARGO_EVENTS_EVENT_BUS': ARGO_EVENTS_EVENT_BUS, 'METAFLOW_ARGO_EVENTS_EVENT_SOURCE': ARGO_EVENTS_EVENT_SOURCE, 'METAFLOW_ARGO_EVENTS_SERVICE_ACCOUNT': ARGO_EVENTS_SERVICE_ACCOUNT, 'METAFLOW_ARGO_EVENTS_WEBHOOK_URL': ARGO_EVENTS_INTERNAL_WEBHOOK_URL, 'METAFLOW_ARGO_EVENTS_WEBHOOK_AUTH': ARGO_EVENTS_WEBHOOK_AUTH}, **{'METAFLOW_FLOW_NAME': self.flow.name, 'METAFLOW_STEP_NAME': node.name, 'METAFLOW_RUN_ID': run_id, 'METAFLOW_RETRY_COUNT': retry_count, 'METAFLOW_PRODUCTION_TOKEN': self.production_token, 'ARGO_WORKFLOW_TEMPLATE': self.name, 'ARGO_WORKFLOW_NAME': '{{workflow.name}}', 'ARGO_WORKFLOW_NAMESPACE': KUBERNETES_NAMESPACE}, **self.metadata.get_runtime_environment('argo-workflows')})
            env['METAFLOW_S3_ENDPOINT_URL'] = S3_ENDPOINT_URL
            env['METAFLOW_INIT_SCRIPT'] = KUBERNETES_SANDBOX_INIT_SCRIPT
            env['METAFLOW_DEFAULT_SECRETS_BACKEND_TYPE'] = DEFAULT_SECRETS_BACKEND_TYPE
            env['METAFLOW_AWS_SECRETS_MANAGER_DEFAULT_REGION'] = AWS_SECRETS_MANAGER_DEFAULT_REGION
            env['METAFLOW_AZURE_STORAGE_BLOB_SERVICE_ENDPOINT'] = AZURE_STORAGE_BLOB_SERVICE_ENDPOINT
            env['METAFLOW_DATASTORE_SYSROOT_AZURE'] = DATASTORE_SYSROOT_AZURE
            env['METAFLOW_CARD_AZUREROOT'] = CARD_AZUREROOT
            env['METAFLOW_DATASTORE_SYSROOT_GS'] = DATASTORE_SYSROOT_GS
            env['METAFLOW_CARD_GSROOT'] = CARD_GSROOT
            if self.triggers:
                for event in self.triggers:
                    env['METAFLOW_ARGO_EVENT_PAYLOAD_%s_%s' % (event['type'], event['sanitized_name'])] = '{{workflow.parameters.%s}}' % event['sanitized_name']
            if S3_SERVER_SIDE_ENCRYPTION is not None:
                env['METAFLOW_S3_SERVER_SIDE_ENCRYPTION'] = S3_SERVER_SIDE_ENCRYPTION
            metaflow_version = self.environment.get_environment_info()
            metaflow_version['flow_name'] = self.graph.name
            metaflow_version['production_token'] = self.production_token
            env['METAFLOW_VERSION'] = json.dumps(metaflow_version)
            inputs = []
            if node.name != 'start':
                inputs.append(Parameter('input-paths'))
            if any((self.graph[n].type == 'foreach' for n in node.in_funcs)):
                inputs.append(Parameter('split-index'))
            outputs = []
            if node.name != 'end':
                outputs = [Parameter('task-id').valueFrom({'path': '/mnt/out/task_id'})]
            if node.type == 'foreach':
                outputs.append(Parameter('num-splits').valueFrom({'path': '/mnt/out/splits'}))
            env = {k: v for (k, v) in env.items() if v is not None and k not in set(ARGO_WORKFLOWS_ENV_VARS_TO_SKIP.split(','))}
            use_tmpfs = resources['use_tmpfs']
            tmpfs_size = resources['tmpfs_size']
            tmpfs_path = resources['tmpfs_path']
            tmpfs_tempdir = resources['tmpfs_tempdir']
            tmpfs_enabled = use_tmpfs or (tmpfs_size and (not use_tmpfs))
            if tmpfs_enabled and tmpfs_tempdir:
                env['METAFLOW_TEMPDIR'] = tmpfs_path
            yield Template(self._sanitize(node.name)).active_deadline_seconds(run_time_limit).service_account_name(resources['service_account']).inputs(Inputs().parameters(inputs)).outputs(Outputs().parameters(outputs)).fail_fast().retry_strategy(times=total_retries, minutes_between_retries=minutes_between_retries).metadata(ObjectMeta().annotation('metaflow/step_name', node.name).annotation('metaflow/attempt', retry_count)).empty_dir_volume('out').empty_dir_volume('tmpfs-ephemeral-volume', medium='Memory', size_limit=tmpfs_size if tmpfs_enabled else 0).pvc_volumes(resources.get('persistent_volume_claims')).node_selectors(resources.get('node_selector')).tolerations(resources.get('tolerations')).container(to_camelcase(kubernetes_sdk.V1Container(name=self._sanitize(node.name), command=cmds, env=[kubernetes_sdk.V1EnvVar(name=k, value=str(v)) for (k, v) in env.items()] + [kubernetes_sdk.V1EnvVar(name=k, value_from=kubernetes_sdk.V1EnvVarSource(field_ref=kubernetes_sdk.V1ObjectFieldSelector(field_path=str(v)))) for (k, v) in {'METAFLOW_KUBERNETES_POD_NAMESPACE': 'metadata.namespace', 'METAFLOW_KUBERNETES_POD_NAME': 'metadata.name', 'METAFLOW_KUBERNETES_POD_ID': 'metadata.uid', 'METAFLOW_KUBERNETES_SERVICE_ACCOUNT_NAME': 'spec.serviceAccountName', 'METAFLOW_KUBERNETES_NODE_IP': 'status.hostIP'}.items()], image=resources['image'], image_pull_policy=resources['image_pull_policy'], resources=kubernetes_sdk.V1ResourceRequirements(requests={'cpu': str(resources['cpu']), 'memory': '%sM' % str(resources['memory']), 'ephemeral-storage': '%sM' % str(resources['disk'])}, limits={'%s.com/gpu'.lower() % resources['gpu_vendor']: str(resources['gpu']) for k in [0] if resources['gpu'] is not None}), env_from=[kubernetes_sdk.V1EnvFromSource(secret_ref=kubernetes_sdk.V1SecretEnvSource(name=str(k))) for k in list([] if not resources.get('secrets') else [resources.get('secrets')] if isinstance(resources.get('secrets'), str) else resources.get('secrets')) + KUBERNETES_SECRETS.split(',') + ARGO_WORKFLOWS_KUBERNETES_SECRETS.split(',') if k], volume_mounts=[kubernetes_sdk.V1VolumeMount(name='out', mount_path='/mnt/out')] + ([kubernetes_sdk.V1VolumeMount(name='tmpfs-ephemeral-volume', mount_path=tmpfs_path)] if tmpfs_enabled else []) + ([kubernetes_sdk.V1VolumeMount(name=claim, mount_path=path) for (claim, path) in resources.get('persistent_volume_claims').items()] if resources.get('persistent_volume_claims') is not None else [])).to_dict()))

    def _exit_hook_templates(self):
        if False:
            while True:
                i = 10
        templates = []
        if self.notify_on_error:
            templates.append(self._slack_error_template())
            templates.append(self._pager_duty_alert_template())
        if self.notify_on_success:
            templates.append(self._slack_success_template())
            templates.append(self._pager_duty_change_template())
        if self.notify_on_error or self.notify_on_success:
            templates.append(Template('exit-hook-hack').http(Http('GET').url(self.notify_slack_webhook_url or 'https://events.pagerduty.com/v2/enqueue').success_condition('true == true')))
        return templates

    def _pager_duty_alert_template(self):
        if False:
            while True:
                i = 10
        if self.notify_pager_duty_integration_key is None:
            return None
        return Template('notify-pager-duty-on-error').http(Http('POST').url('https://events.pagerduty.com/v2/enqueue').header('Content-Type', 'application/json').body(json.dumps({'event_action': 'trigger', 'routing_key': self.notify_pager_duty_integration_key, 'payload': {'source': '{{workflow.name}}', 'severity': 'info', 'summary': 'Metaflow run %s/argo-{{workflow.name}} failed!' % self.flow.name, 'custom_details': {'Flow': self.flow.name, 'Run ID': 'argo-{{workflow.name}}'}}, 'links': self._pager_duty_notification_links()})))

    def _pager_duty_change_template(self):
        if False:
            while True:
                i = 10
        if self.notify_pager_duty_integration_key is None:
            return None
        return Template('notify-pager-duty-on-success').http(Http('POST').url('https://events.pagerduty.com/v2/change/enqueue').header('Content-Type', 'application/json').body(json.dumps({'routing_key': self.notify_pager_duty_integration_key, 'payload': {'summary': 'Metaflow run %s/argo-{{workflow.name}} Succeeded' % self.flow.name, 'source': '{{workflow.name}}', 'custom_details': {'Flow': self.flow.name, 'Run ID': 'argo-{{workflow.name}}'}}, 'links': self._pager_duty_notification_links()})))

    def _pager_duty_notification_links(self):
        if False:
            while True:
                i = 10
        links = []
        if UI_URL:
            links.append({'href': '%s/%s/%s' % (UI_URL.rstrip('/'), self.flow.name, 'argo-{{workflow.name}}'), 'text': 'Metaflow UI'})
        if ARGO_WORKFLOWS_UI_URL:
            links.append({'href': '%s/workflows/%s/%s' % (ARGO_WORKFLOWS_UI_URL.rstrip('/'), '{{workflow.namespace}}', '{{workflow.name}}'), 'text': 'Argo UI'})
        return links

    def _slack_error_template(self):
        if False:
            i = 10
            return i + 15
        if self.notify_slack_webhook_url is None:
            return None
        return Template('notify-slack-on-error').http(Http('POST').url(self.notify_slack_webhook_url).body(json.dumps({'text': ':rotating_light: _%s/argo-{{workflow.name}}_ failed!' % self.flow.name})))

    def _slack_success_template(self):
        if False:
            while True:
                i = 10
        if self.notify_slack_webhook_url is None:
            return None
        return Template('notify-slack-on-success').http(Http('POST').url(self.notify_slack_webhook_url).body(json.dumps({'text': ':white_check_mark: _%s/argo-{{workflow.name}}_ succeeded!' % self.flow.name})))

    def _compile_sensor(self):
        if False:
            print('Hello World!')
        if not self.triggers:
            return {}
        if ARGO_EVENTS_EVENT is None:
            raise ArgoWorkflowsException("An Argo Event name hasn't been configured for your deployment yet. Please see this article for more details on event names - https://argoproj.github.io/argo-events/eventsources/naming/. It is very likely that all events for your deployment share the same name. You can configure it by executing `metaflow configure kubernetes` or setting METAFLOW_ARGO_EVENTS_EVENT in your configuration. If in doubt, reach out for support at http://chat.metaflow.org")
        if ARGO_EVENTS_EVENT_SOURCE is None:
            raise ArgoWorkflowsException("An Argo Event Source name hasn't been configured for your deployment yet. Please see this article for more details on event names - https://argoproj.github.io/argo-events/eventsources/naming/. You can configure it by executing `metaflow configure kubernetes` or setting METAFLOW_ARGO_EVENTS_EVENT_SOURCE in your configuration. If in doubt, reach out for support at http://chat.metaflow.org")
        if ARGO_EVENTS_SERVICE_ACCOUNT is None:
            raise ArgoWorkflowsException("An Argo Event service account hasn't been configured for your deployment yet. Please see this article for more details on event names - https://argoproj.github.io/argo-events/service-accounts/. You can configure it by executing `metaflow configure kubernetes` or setting METAFLOW_ARGO_EVENTS_SERVICE_ACCOUNT in your configuration. If in doubt, reach out for support at http://chat.metaflow.org")
        try:
            from kubernetes import client as kubernetes_sdk
        except (NameError, ImportError):
            raise MetaflowException("Could not import Python package 'kubernetes'. Install kubernetes sdk (https://pypi.org/project/kubernetes/) first.")
        labels = {'app.kubernetes.io/part-of': 'metaflow'}
        annotations = {'metaflow/production_token': self.production_token, 'metaflow/owner': self.username, 'metaflow/user': 'argo-workflows', 'metaflow/flow_name': self.flow.name}
        if current.get('project_name'):
            annotations.update({'metaflow/project_name': current.project_name, 'metaflow/branch_name': current.branch_name, 'metaflow/project_flow_name': current.project_flow_name})
        trigger_annotations = {'metaflow/triggered_by': json.dumps([{key: trigger.get(key) for key in ['name', 'type']} for trigger in self.triggers])}
        return Sensor().metadata(ObjectMeta().name(self.name.replace('.', '-')).namespace(KUBERNETES_NAMESPACE).label('app.kubernetes.io/name', 'metaflow-sensor').label('app.kubernetes.io/part-of', 'metaflow').labels(self.kubernetes_labels).annotations(annotations)).spec(SensorSpec().template(SensorTemplate().metadata(ObjectMeta().label('app.kubernetes.io/name', 'metaflow-sensor').label('app.kubernetes.io/part-of', 'metaflow').annotations(annotations)).container(to_camelcase(kubernetes_sdk.V1Container(name='main', resources=kubernetes_sdk.V1ResourceRequirements(requests={'cpu': '100m', 'memory': '250Mi'}, limits={'cpu': '100m', 'memory': '250Mi'})))).service_account_name(ARGO_EVENTS_SERVICE_ACCOUNT)).replicas(1).event_bus_name(ARGO_EVENTS_EVENT_BUS).trigger(Trigger().template(TriggerTemplate(self.name).argo_workflow_trigger(ArgoWorkflowTrigger().source({'resource': {'apiVersion': 'argoproj.io/v1alpha1', 'kind': 'Workflow', 'metadata': {'generateName': '%s-' % self.name, 'namespace': KUBERNETES_NAMESPACE, 'annotations': {'metaflow/triggered_by': json.dumps([{key: trigger.get(key) for key in ['name', 'type']} for trigger in self.triggers])}}, 'spec': {'arguments': {'parameters': [Parameter(parameter['name']).value(parameter['value']).to_json() for parameter in self.parameters.values()] + [Parameter(event['sanitized_name']).value(json.dumps(None)).to_json() for event in self.triggers]}, 'workflowTemplateRef': {'name': self.name}}}}).parameters([y for x in list((list((TriggerParameter().src(dependency_name=event['sanitized_name'], data_template='{{ .Input.body.payload.%s | toJson }}' % v, value=self.parameters[parameter_name]['value']).dest('spec.arguments.parameters.#(name=%s).value' % parameter_name) for (parameter_name, v) in event.get('parameters', {}).items())) for event in self.triggers)) for y in x] + [TriggerParameter().src(dependency_name=event['sanitized_name'], data_key='body.payload', value=json.dumps(None)).dest('spec.arguments.parameters.#(name=%s).value' % event['sanitized_name']) for event in self.triggers])).conditions_reset(cron=self.trigger_options.get('reset_at', {}).get('cron'), timezone=self.trigger_options.get('reset_at', {}).get('timezone')))).dependencies((EventDependency(event['sanitized_name']).event_name(ARGO_EVENTS_EVENT).event_source_name(ARGO_EVENTS_EVENT_SOURCE).filters(EventDependencyFilter().exprs([{'expr': "name == '%s'" % event['name'], 'fields': [{'name': 'name', 'path': 'body.payload.name'}]}] + [{'expr': 'true == true', 'fields': [{'name': 'field', 'path': 'body.payload.%s' % v}]} for (parameter_name, v) in event.get('parameters', {}).items() if self.parameters[parameter_name]['is_required']] + [{'expr': "field == '%s'" % v, 'fields': [{'name': 'field', 'path': 'body.payload.%s' % filter_key}]} for (filter_key, v) in event.get('filters', {}).items() if v])) for event in self.triggers)))

    def list_to_prose(self, items, singular):
        if False:
            return 10
        items = ['*%s*' % item for item in items]
        item_count = len(items)
        plural = singular + 's'
        item_type = singular
        if item_count == 1:
            result = items[0]
        elif item_count == 2:
            result = '%s and %s' % (items[0], items[1])
            item_type = plural
        elif item_count > 2:
            result = '%s and %s' % (', '.join(items[0:item_count - 1]), items[item_count - 1])
            item_type = plural
        else:
            result = ''
        if result:
            result = '%s %s' % (result, item_type)
        return result

class WorkflowTemplate(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        tree = lambda : defaultdict(tree)
        self.payload = tree()
        self.payload['apiVersion'] = 'argoproj.io/v1alpha1'
        self.payload['kind'] = 'WorkflowTemplate'

    def metadata(self, object_meta):
        if False:
            print('Hello World!')
        self.payload['metadata'] = object_meta.to_json()
        return self

    def spec(self, workflow_spec):
        if False:
            while True:
                i = 10
        self.payload['spec'] = workflow_spec.to_json()
        return self

    def to_json(self):
        if False:
            for i in range(10):
                print('nop')
        return self.payload

    def __str__(self):
        if False:
            while True:
                i = 10
        return json.dumps(self.payload, indent=4)

class ObjectMeta(object):

    def __init__(self):
        if False:
            print('Hello World!')
        tree = lambda : defaultdict(tree)
        self.payload = tree()

    def annotation(self, key, value):
        if False:
            i = 10
            return i + 15
        self.payload['annotations'][key] = str(value)
        return self

    def annotations(self, annotations):
        if False:
            i = 10
            return i + 15
        if 'annotations' not in self.payload:
            self.payload['annotations'] = {}
        self.payload['annotations'].update(annotations)
        return self

    def generate_name(self, generate_name):
        if False:
            i = 10
            return i + 15
        self.payload['generateName'] = generate_name
        return self

    def label(self, key, value):
        if False:
            while True:
                i = 10
        self.payload['labels'][key] = str(value)
        return self

    def labels(self, labels):
        if False:
            return 10
        if 'labels' not in self.payload:
            self.payload['labels'] = {}
        self.payload['labels'].update(labels or {})
        return self

    def name(self, name):
        if False:
            i = 10
            return i + 15
        self.payload['name'] = name
        return self

    def namespace(self, namespace):
        if False:
            i = 10
            return i + 15
        self.payload['namespace'] = namespace
        return self

    def to_json(self):
        if False:
            return 10
        return self.payload

    def __str__(self):
        if False:
            print('Hello World!')
        return json.dumps(self.to_json(), indent=4)

class WorkflowSpec(object):

    def __init__(self):
        if False:
            print('Hello World!')
        tree = lambda : defaultdict(tree)
        self.payload = tree()

    def active_deadline_seconds(self, active_deadline_seconds):
        if False:
            i = 10
            return i + 15
        if active_deadline_seconds is not None:
            self.payload['activeDeadlineSeconds'] = int(active_deadline_seconds)
        return self

    def automount_service_account_token(self, mount=True):
        if False:
            for i in range(10):
                print('nop')
        self.payload['automountServiceAccountToken'] = mount
        return self

    def arguments(self, arguments):
        if False:
            while True:
                i = 10
        self.payload['arguments'] = arguments.to_json()
        return self

    def archive_logs(self, archive_logs=True):
        if False:
            while True:
                i = 10
        self.payload['archiveLogs'] = archive_logs
        return self

    def entrypoint(self, entrypoint):
        if False:
            return 10
        self.payload['entrypoint'] = entrypoint
        return self

    def parallelism(self, parallelism):
        if False:
            print('Hello World!')
        self.payload['parallelism'] = int(parallelism)
        return self

    def pod_metadata(self, metadata):
        if False:
            while True:
                i = 10
        self.payload['podMetadata'] = metadata.to_json()
        return self

    def priority(self, priority):
        if False:
            while True:
                i = 10
        if priority is not None:
            self.payload['priority'] = int(priority)
        return self

    def workflow_metadata(self, workflow_metadata):
        if False:
            return 10
        self.payload['workflowMetadata'] = workflow_metadata.to_json()
        return self

    def service_account_name(self, service_account_name):
        if False:
            while True:
                i = 10
        self.payload['serviceAccountName'] = service_account_name
        return self

    def templates(self, templates):
        if False:
            for i in range(10):
                print('nop')
        if 'templates' not in self.payload:
            self.payload['templates'] = []
        for template in templates:
            self.payload['templates'].append(template.to_json())
        return self

    def hooks(self, hooks):
        if False:
            for i in range(10):
                print('nop')
        if 'hooks' not in self.payload:
            self.payload['hooks'] = {}
        for (k, v) in hooks.items():
            self.payload['hooks'].update({k: v.to_json()})
        return self

    def to_json(self):
        if False:
            while True:
                i = 10
        return self.payload

    def __str__(self):
        if False:
            return 10
        return json.dumps(self.to_json(), indent=4)

class Metadata(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        tree = lambda : defaultdict(tree)
        self.payload = tree()

    def annotation(self, key, value):
        if False:
            for i in range(10):
                print('nop')
        self.payload['annotations'][key] = str(value)
        return self

    def annotations(self, annotations):
        if False:
            for i in range(10):
                print('nop')
        if 'annotations' not in self.payload:
            self.payload['annotations'] = {}
        self.payload['annotations'].update(annotations)
        return self

    def label(self, key, value):
        if False:
            i = 10
            return i + 15
        self.payload['labels'][key] = str(value)
        return self

    def labels(self, labels):
        if False:
            i = 10
            return i + 15
        if 'labels' not in self.payload:
            self.payload['labels'] = {}
        self.payload['labels'].update(labels or {})
        return self

    def labels_from(self, labels_from):
        if False:
            for i in range(10):
                print('nop')
        if 'labelsFrom' not in self.payload:
            self.payload['labelsFrom'] = {}
        for (k, v) in labels_from.items():
            self.payload['labelsFrom'].update({k: {'expression': v}})
        return self

    def to_json(self):
        if False:
            return 10
        return self.payload

    def __str__(self):
        if False:
            return 10
        return json.dumps(self.to_json(), indent=4)

class Template(object):

    def __init__(self, name):
        if False:
            i = 10
            return i + 15
        tree = lambda : defaultdict(tree)
        self.payload = tree()
        self.payload['name'] = name

    def active_deadline_seconds(self, active_deadline_seconds):
        if False:
            while True:
                i = 10
        self.payload['activeDeadlineSeconds'] = int(active_deadline_seconds)
        return self

    def dag(self, dag_template):
        if False:
            i = 10
            return i + 15
        self.payload['dag'] = dag_template.to_json()
        return self

    def container(self, container):
        if False:
            print('Hello World!')
        self.payload['container'] = container
        return self

    def http(self, http):
        if False:
            i = 10
            return i + 15
        self.payload['http'] = http.to_json()
        return self

    def inputs(self, inputs):
        if False:
            i = 10
            return i + 15
        self.payload['inputs'] = inputs.to_json()
        return self

    def outputs(self, outputs):
        if False:
            i = 10
            return i + 15
        self.payload['outputs'] = outputs.to_json()
        return self

    def fail_fast(self, fail_fast=True):
        if False:
            return 10
        self.payload['failFast'] = fail_fast
        return self

    def metadata(self, metadata):
        if False:
            print('Hello World!')
        self.payload['metadata'] = metadata.to_json()
        return self

    def service_account_name(self, service_account_name):
        if False:
            i = 10
            return i + 15
        self.payload['serviceAccountName'] = service_account_name
        return self

    def retry_strategy(self, times, minutes_between_retries):
        if False:
            i = 10
            return i + 15
        if times > 0:
            self.payload['retryStrategy'] = {'retryPolicy': 'Always', 'limit': times, 'backoff': {'duration': '%sm' % minutes_between_retries}}
        return self

    def empty_dir_volume(self, name, medium=None, size_limit=None):
        if False:
            return 10
        '\n        Create and attach an emptyDir volume for Kubernetes.\n\n        Parameters:\n        -----------\n        name: str\n            name for the volume\n        size_limit: int (optional)\n            sizeLimit (in MiB) for the volume\n        medium: str (optional)\n            storage medium of the emptyDir\n        '
        if size_limit == 0:
            return self
        if 'volumes' not in self.payload:
            self.payload['volumes'] = []
        self.payload['volumes'].append({'name': name, 'emptyDir': {**({'sizeLimit': '{}Mi'.format(size_limit)} if size_limit else {}), **({'medium': medium} if medium else {})}})
        return self

    def pvc_volumes(self, pvcs=None):
        if False:
            while True:
                i = 10
        '\n        Create and attach Persistent Volume Claims as volumes.\n\n        Parameters:\n        -----------\n        pvcs: Optional[Dict]\n            a dictionary of pvc\'s and the paths they should be mounted to. e.g.\n            {"pv-claim-1": "/mnt/path1", "pv-claim-2": "/mnt/path2"}\n        '
        if pvcs is None:
            return self
        if 'volumes' not in self.payload:
            self.payload['volumes'] = []
        for claim in pvcs.keys():
            self.payload['volumes'].append({'name': claim, 'persistentVolumeClaim': {'claimName': claim}})
        return self

    def node_selectors(self, node_selectors):
        if False:
            while True:
                i = 10
        if 'nodeSelector' not in self.payload:
            self.payload['nodeSelector'] = {}
        if node_selectors:
            self.payload['nodeSelector'].update(node_selectors)
        return self

    def tolerations(self, tolerations):
        if False:
            for i in range(10):
                print('nop')
        self.payload['tolerations'] = tolerations
        return self

    def to_json(self):
        if False:
            for i in range(10):
                print('nop')
        return self.payload

    def __str__(self):
        if False:
            while True:
                i = 10
        return json.dumps(self.payload, indent=4)

class Inputs(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        tree = lambda : defaultdict(tree)
        self.payload = tree()

    def parameters(self, parameters):
        if False:
            for i in range(10):
                print('nop')
        if 'parameters' not in self.payload:
            self.payload['parameters'] = []
        for parameter in parameters:
            self.payload['parameters'].append(parameter.to_json())
        return self

    def to_json(self):
        if False:
            print('Hello World!')
        return self.payload

    def __str__(self):
        if False:
            while True:
                i = 10
        return json.dumps(self.payload, indent=4)

class Outputs(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        tree = lambda : defaultdict(tree)
        self.payload = tree()

    def parameters(self, parameters):
        if False:
            while True:
                i = 10
        if 'parameters' not in self.payload:
            self.payload['parameters'] = []
        for parameter in parameters:
            self.payload['parameters'].append(parameter.to_json())
        return self

    def to_json(self):
        if False:
            while True:
                i = 10
        return self.payload

    def __str__(self):
        if False:
            while True:
                i = 10
        return json.dumps(self.payload, indent=4)

class Parameter(object):

    def __init__(self, name):
        if False:
            print('Hello World!')
        tree = lambda : defaultdict(tree)
        self.payload = tree()
        self.payload['name'] = name

    def value(self, value):
        if False:
            i = 10
            return i + 15
        self.payload['value'] = value
        return self

    def default(self, value):
        if False:
            print('Hello World!')
        self.payload['default'] = value
        return self

    def valueFrom(self, value_from):
        if False:
            for i in range(10):
                print('nop')
        self.payload['valueFrom'] = value_from
        return self

    def description(self, description):
        if False:
            while True:
                i = 10
        self.payload['description'] = description
        return self

    def to_json(self):
        if False:
            print('Hello World!')
        return self.payload

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return json.dumps(self.payload, indent=4)

class DAGTemplate(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        tree = lambda : defaultdict(tree)
        self.payload = tree()

    def fail_fast(self, fail_fast=True):
        if False:
            print('Hello World!')
        self.payload['failFast'] = fail_fast
        return self

    def tasks(self, tasks):
        if False:
            print('Hello World!')
        if 'tasks' not in self.payload:
            self.payload['tasks'] = []
        for task in tasks:
            self.payload['tasks'].append(task.to_json())
        return self

    def to_json(self):
        if False:
            print('Hello World!')
        return self.payload

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return json.dumps(self.payload, indent=4)

class DAGTask(object):

    def __init__(self, name):
        if False:
            for i in range(10):
                print('nop')
        tree = lambda : defaultdict(tree)
        self.payload = tree()
        self.payload['name'] = name

    def arguments(self, arguments):
        if False:
            for i in range(10):
                print('nop')
        self.payload['arguments'] = arguments.to_json()
        return self

    def dependencies(self, dependencies):
        if False:
            while True:
                i = 10
        self.payload['dependencies'] = dependencies
        return self

    def template(self, template):
        if False:
            print('Hello World!')
        self.payload['template'] = template
        return self

    def inline(self, template):
        if False:
            print('Hello World!')
        self.payload['inline'] = template.to_json()
        return self

    def with_param(self, with_param):
        if False:
            return 10
        self.payload['withParam'] = with_param
        return self

    def to_json(self):
        if False:
            i = 10
            return i + 15
        return self.payload

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return json.dumps(self.payload, indent=4)

class Arguments(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        tree = lambda : defaultdict(tree)
        self.payload = tree()

    def parameters(self, parameters):
        if False:
            return 10
        if 'parameters' not in self.payload:
            self.payload['parameters'] = []
        for parameter in parameters:
            self.payload['parameters'].append(parameter.to_json())
        return self

    def to_json(self):
        if False:
            while True:
                i = 10
        return self.payload

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return json.dumps(self.payload, indent=4)

class Sensor(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        tree = lambda : defaultdict(tree)
        self.payload = tree()
        self.payload['apiVersion'] = 'argoproj.io/v1alpha1'
        self.payload['kind'] = 'Sensor'

    def metadata(self, object_meta):
        if False:
            for i in range(10):
                print('nop')
        self.payload['metadata'] = object_meta.to_json()
        return self

    def spec(self, sensor_spec):
        if False:
            i = 10
            return i + 15
        self.payload['spec'] = sensor_spec.to_json()
        return self

    def to_json(self):
        if False:
            return 10
        return self.payload

    def __str__(self):
        if False:
            while True:
                i = 10
        return json.dumps(self.payload, indent=4)

class SensorSpec(object):

    def __init__(self):
        if False:
            print('Hello World!')
        tree = lambda : defaultdict(tree)
        self.payload = tree()

    def replicas(self, replicas=1):
        if False:
            return 10
        self.payload['replicas'] = int(replicas)
        return self

    def template(self, sensor_template):
        if False:
            print('Hello World!')
        self.payload['template'] = sensor_template.to_json()
        return self

    def trigger(self, trigger):
        if False:
            print('Hello World!')
        if 'triggers' not in self.payload:
            self.payload['triggers'] = []
        self.payload['triggers'].append(trigger.to_json())
        return self

    def dependencies(self, dependencies):
        if False:
            i = 10
            return i + 15
        if 'dependencies' not in self.payload:
            self.payload['dependencies'] = []
        for dependency in dependencies:
            self.payload['dependencies'].append(dependency.to_json())
        return self

    def event_bus_name(self, event_bus_name):
        if False:
            print('Hello World!')
        self.payload['eventBusName'] = event_bus_name
        return self

    def to_json(self):
        if False:
            i = 10
            return i + 15
        return self.payload

    def __str__(self):
        if False:
            print('Hello World!')
        return json.dumps(self.to_json(), indent=4)

class SensorTemplate(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        tree = lambda : defaultdict(tree)
        self.payload = tree()

    def service_account_name(self, service_account_name):
        if False:
            for i in range(10):
                print('nop')
        self.payload['serviceAccountName'] = service_account_name
        return self

    def metadata(self, object_meta):
        if False:
            for i in range(10):
                print('nop')
        self.payload['metadata'] = object_meta.to_json()
        return self

    def container(self, container):
        if False:
            i = 10
            return i + 15
        self.payload['container'] = container
        return self

    def to_json(self):
        if False:
            print('Hello World!')
        return self.payload

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return json.dumps(self.to_json(), indent=4)

class EventDependency(object):

    def __init__(self, name):
        if False:
            while True:
                i = 10
        tree = lambda : defaultdict(tree)
        self.payload = tree()
        self.payload['name'] = name

    def event_source_name(self, event_source_name):
        if False:
            return 10
        self.payload['eventSourceName'] = event_source_name
        return self

    def event_name(self, event_name):
        if False:
            return 10
        self.payload['eventName'] = event_name
        return self

    def filters(self, event_dependency_filter):
        if False:
            return 10
        self.payload['filters'] = event_dependency_filter.to_json()
        return self

    def transform(self, event_dependency_transformer=None):
        if False:
            while True:
                i = 10
        if event_dependency_transformer:
            self.payload['transform'] = event_dependency_transformer
        return self

    def filters_logical_operator(self, logical_operator):
        if False:
            while True:
                i = 10
        self.payload['filtersLogicalOperator'] = logical_operator.to_json()
        return self

    def to_json(self):
        if False:
            return 10
        return self.payload

    def __str__(self):
        if False:
            while True:
                i = 10
        return json.dumps(self.to_json(), indent=4)

class EventDependencyFilter(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        tree = lambda : defaultdict(tree)
        self.payload = tree()

    def exprs(self, exprs):
        if False:
            for i in range(10):
                print('nop')
        self.payload['exprs'] = exprs
        return self

    def context(self, event_context):
        if False:
            for i in range(10):
                print('nop')
        self.payload['context'] = event_context
        return self

    def to_json(self):
        if False:
            return 10
        return self.payload

    def __str__(self):
        if False:
            return 10
        return json.dumps(self.to_json(), indent=4)

class Trigger(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        tree = lambda : defaultdict(tree)
        self.payload = tree()

    def template(self, trigger_template):
        if False:
            print('Hello World!')
        self.payload['template'] = trigger_template.to_json()
        return self

    def parameters(self, trigger_parameters):
        if False:
            i = 10
            return i + 15
        if 'parameters' not in self.payload:
            self.payload['parameters'] = []
        for trigger_parameter in trigger_parameters:
            self.payload['parameters'].append(trigger_parameter.to_json())
        return self

    def policy(self, trigger_policy):
        if False:
            print('Hello World!')
        self.payload['policy'] = trigger_policy.to_json()
        return self

    def to_json(self):
        if False:
            while True:
                i = 10
        return self.payload

    def __str__(self):
        if False:
            while True:
                i = 10
        return json.dumps(self.to_json(), indent=4)

class TriggerTemplate(object):

    def __init__(self, name):
        if False:
            while True:
                i = 10
        tree = lambda : defaultdict(tree)
        self.payload = tree()
        self.payload['name'] = name

    def argo_workflow_trigger(self, argo_workflow_trigger):
        if False:
            i = 10
            return i + 15
        self.payload['argoWorkflow'] = argo_workflow_trigger.to_json()
        return self

    def conditions_reset(self, cron, timezone):
        if False:
            print('Hello World!')
        if cron:
            self.payload['conditionsReset'] = [{'byTime': {'cron': cron, 'timezone': timezone}}]
        return self

    def to_json(self):
        if False:
            i = 10
            return i + 15
        return self.payload

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return json.dumps(self.payload, indent=4)

class ArgoWorkflowTrigger(object):

    def __init__(self):
        if False:
            print('Hello World!')
        tree = lambda : defaultdict(tree)
        self.payload = tree()
        self.payload['operation'] = 'submit'
        self.payload['group'] = 'argoproj.io'
        self.payload['version'] = 'v1alpha1'
        self.payload['resource'] = 'workflows'

    def source(self, source):
        if False:
            print('Hello World!')
        self.payload['source'] = source
        return self

    def parameters(self, trigger_parameters):
        if False:
            return 10
        if 'parameters' not in self.payload:
            self.payload['parameters'] = []
        for trigger_parameter in trigger_parameters:
            self.payload['parameters'].append(trigger_parameter.to_json())
        return self

    def to_json(self):
        if False:
            return 10
        return self.payload

    def __str__(self):
        if False:
            print('Hello World!')
        return json.dumps(self.payload, indent=4)

class TriggerParameter(object):

    def __init__(self):
        if False:
            print('Hello World!')
        tree = lambda : defaultdict(tree)
        self.payload = tree()

    def src(self, dependency_name, value, data_key=None, data_template=None):
        if False:
            return 10
        self.payload['src'] = {'dependencyName': dependency_name, 'dataKey': data_key, 'dataTemplate': data_template, 'value': value, 'useRawData': False}
        return self

    def dest(self, dest):
        if False:
            while True:
                i = 10
        self.payload['dest'] = dest
        return self

    def to_json(self):
        if False:
            return 10
        return self.payload

    def __str__(self):
        if False:
            return 10
        return json.dumps(self.payload, indent=4)

class Http(object):

    def __init__(self, method):
        if False:
            for i in range(10):
                print('nop')
        tree = lambda : defaultdict(tree)
        self.payload = tree()
        self.payload['method'] = method
        self.payload['headers'] = []

    def header(self, header, value):
        if False:
            return 10
        self.payload['headers'].append({'name': header, 'value': value})
        return self

    def body(self, body):
        if False:
            return 10
        self.payload['body'] = str(body)
        return self

    def url(self, url):
        if False:
            print('Hello World!')
        self.payload['url'] = url
        return self

    def success_condition(self, success_condition):
        if False:
            print('Hello World!')
        self.payload['successCondition'] = success_condition
        return self

    def to_json(self):
        if False:
            i = 10
            return i + 15
        return self.payload

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return json.dumps(self.payload, indent=4)

class LifecycleHook(object):

    def __init__(self):
        if False:
            print('Hello World!')
        tree = lambda : defaultdict(tree)
        self.payload = tree()

    def expression(self, expression):
        if False:
            i = 10
            return i + 15
        self.payload['expression'] = str(expression)
        return self

    def template(self, template):
        if False:
            while True:
                i = 10
        self.payload['template'] = template
        return self

    def to_json(self):
        if False:
            print('Hello World!')
        return self.payload

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return json.dumps(self.payload, indent=4)