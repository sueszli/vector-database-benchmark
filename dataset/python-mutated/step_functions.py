import hashlib
import json
import os
import random
import string
import sys
from collections import defaultdict
from metaflow import R
from metaflow.decorators import flow_decorators
from metaflow.exception import MetaflowException
from metaflow.metaflow_config import EVENTS_SFN_ACCESS_IAM_ROLE, S3_ENDPOINT_URL, SFN_DYNAMO_DB_TABLE, SFN_EXECUTION_LOG_GROUP_ARN, SFN_IAM_ROLE
from metaflow.parameters import deploy_time_eval
from metaflow.util import dict_to_cli_options, to_pascalcase
from ..batch.batch import Batch
from .event_bridge_client import EventBridgeClient
from .step_functions_client import StepFunctionsClient

class StepFunctionsException(MetaflowException):
    headline = 'AWS Step Functions error'

class StepFunctionsSchedulingException(MetaflowException):
    headline = 'AWS Step Functions scheduling error'

class StepFunctions(object):

    def __init__(self, name, graph, flow, code_package_sha, code_package_url, production_token, metadata, flow_datastore, environment, event_logger, monitor, tags=None, namespace=None, username=None, max_workers=None, workflow_timeout=None, is_project=False):
        if False:
            i = 10
            return i + 15
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
        self._client = StepFunctionsClient()
        self._workflow = self._compile()
        self._cron = self._cron()
        self._state_machine_arn = None

    def to_json(self):
        if False:
            return 10
        return self._workflow.to_json(pretty=True)

    def trigger_explanation(self):
        if False:
            while True:
                i = 10
        if self._cron:
            return 'This workflow triggers automatically via a cron schedule *%s* defined in AWS EventBridge.' % self.event_bridge_rule
        else:
            return 'No triggers defined. You need to launch this workflow manually.'

    def deploy(self, log_execution_history):
        if False:
            i = 10
            return i + 15
        if SFN_IAM_ROLE is None:
            raise StepFunctionsException('No IAM role found for AWS Step Functions. You can create one following the instructions listed at *https://admin-docs.metaflow.org/metaflow-on-aws/deployment-guide/manual-deployment#scheduling* and re-configure Metaflow using *metaflow configure aws* on your terminal.')
        if log_execution_history:
            if SFN_EXECUTION_LOG_GROUP_ARN is None:
                raise StepFunctionsException('No AWS CloudWatch Logs log group ARN found for emitting state machine execution logs for your workflow. You can set it in your environment by using the METAFLOW_SFN_EXECUTION_LOG_GROUP_ARN environment variable.')
        try:
            self._state_machine_arn = self._client.push(name=self.name, definition=self.to_json(), role_arn=SFN_IAM_ROLE, log_execution_history=log_execution_history)
        except Exception as e:
            raise StepFunctionsException(repr(e))

    def schedule(self):
        if False:
            i = 10
            return i + 15
        if EVENTS_SFN_ACCESS_IAM_ROLE is None:
            raise StepFunctionsSchedulingException('No IAM role found for AWS Events Bridge. You can create one following the instructions listed at *https://admin-docs.metaflow.org/metaflow-on-aws/deployment-guide/manual-deployment#scheduling* and re-configure Metaflow using *metaflow configure aws* on your terminal.')
        try:
            self.event_bridge_rule = EventBridgeClient(self.name).cron(self._cron).role_arn(EVENTS_SFN_ACCESS_IAM_ROLE).state_machine_arn(self._state_machine_arn).schedule()
        except Exception as e:
            raise StepFunctionsSchedulingException(repr(e))

    @classmethod
    def delete(cls, name):
        if False:
            for i in range(10):
                print('nop')
        schedule_deleted = EventBridgeClient(name).delete()
        sfn_deleted = StepFunctionsClient().delete(name)
        if sfn_deleted is None:
            raise StepFunctionsException("The workflow *%s* doesn't exist on AWS Step Functions." % name)
        return (schedule_deleted, sfn_deleted)

    @classmethod
    def trigger(cls, name, parameters):
        if False:
            for i in range(10):
                print('nop')
        try:
            state_machine = StepFunctionsClient().get(name)
        except Exception as e:
            raise StepFunctionsException(repr(e))
        if state_machine is None:
            raise StepFunctionsException("The workflow *%s* doesn't exist on AWS Step Functions. Please deploy your flow first." % name)
        input = json.dumps({'Parameters': json.dumps(parameters)})
        if len(input) > 20480:
            raise StepFunctionsException("Length of parameter names and values shouldn't exceed 20480 as imposed by AWS Step Functions.")
        try:
            state_machine_arn = state_machine.get('stateMachineArn')
            return StepFunctionsClient().trigger(state_machine_arn, input)
        except Exception as e:
            raise StepFunctionsException(repr(e))

    @classmethod
    def list(cls, name, states):
        if False:
            for i in range(10):
                print('nop')
        try:
            state_machine = StepFunctionsClient().get(name)
        except Exception as e:
            raise StepFunctionsException(repr(e))
        if state_machine is None:
            raise StepFunctionsException("The workflow *%s* doesn't exist on AWS Step Functions." % name)
        try:
            state_machine_arn = state_machine.get('stateMachineArn')
            return StepFunctionsClient().list_executions(state_machine_arn, states)
        except Exception as e:
            raise StepFunctionsException(repr(e))

    @classmethod
    def get_existing_deployment(cls, name):
        if False:
            while True:
                i = 10
        workflow = StepFunctionsClient().get(name)
        if workflow is not None:
            try:
                start = json.loads(workflow['definition'])['States']['start']
                parameters = start['Parameters']['Parameters']
                return (parameters.get('metaflow.owner'), parameters.get('metaflow.production_token'))
            except KeyError as e:
                raise StepFunctionsException('An existing non-metaflow workflow with the same name as *%s* already exists in AWS Step Functions. Please modify the name of this flow or delete your existing workflow on AWS Step Functions.' % name)
        return None

    def _compile(self):
        if False:
            i = 10
            return i + 15
        if self.flow._flow_decorators.get('trigger') or self.flow._flow_decorators.get('trigger_on_finish'):
            raise StepFunctionsException('Deploying flows with @trigger or @trigger_on_finish decorator(s) to AWS Step Functions is not supported currently.')

        def _visit(node, workflow, exit_node=None):
            if False:
                i = 10
                return i + 15
            if node.parallel_foreach:
                raise StepFunctionsException('Deploying flows with @parallel decorator(s) to AWS Step Functions is not supported currently.')
            state = State(node.name).batch(self._batch(node)).output_path("$.['JobId', 'Parameters', 'Index', 'SplitParentTaskId']")
            if node.type == 'end' or exit_node in node.out_funcs:
                workflow.add_state(state.end())
            elif node.type in ('start', 'linear', 'join'):
                workflow.add_state(state.next(node.out_funcs[0]))
                _visit(self.graph[node.out_funcs[0]], workflow, exit_node)
            elif node.type == 'split':
                branch_name = hashlib.sha224('&'.join(node.out_funcs).encode('utf-8')).hexdigest()
                workflow.add_state(state.next(branch_name))
                branch = Parallel(branch_name).next(node.matching_join)
                for n in node.out_funcs:
                    branch.branch(_visit(self.graph[n], Workflow(n).start_at(n), node.matching_join))
                workflow.add_state(branch)
                _visit(self.graph[node.matching_join], workflow, exit_node)
            elif node.type == 'foreach':
                cardinality_state_name = '#%s' % node.out_funcs[0]
                workflow.add_state(state.next(cardinality_state_name))
                cardinality_state = State(cardinality_state_name).dynamo_db(SFN_DYNAMO_DB_TABLE, '$.JobId', 'for_each_cardinality').result_path('$.Result')
                iterator_name = '*%s' % node.out_funcs[0]
                workflow.add_state(cardinality_state.next(iterator_name))
                workflow.add_state(Map(iterator_name).items_path('$.Result.Item.for_each_cardinality.NS').parameter('JobId.$', '$.JobId').parameter('SplitParentTaskId.$', '$.JobId').parameter('Parameters.$', '$.Parameters').parameter('Index.$', '$$.Map.Item.Value').next(node.matching_join).iterator(_visit(self.graph[node.out_funcs[0]], Workflow(node.out_funcs[0]).start_at(node.out_funcs[0]), node.matching_join)).max_concurrency(self.max_workers).output_path('$.[0]'))
                _visit(self.graph[node.matching_join], workflow, exit_node)
            else:
                raise StepFunctionsException('Node type *%s* for  step *%s* is not currently supported by AWS Step Functions.' % (node.type, node.name))
            return workflow
        workflow = Workflow(self.name).start_at('start')
        if self.workflow_timeout:
            workflow.timeout_seconds(self.workflow_timeout)
        return _visit(self.graph['start'], workflow)

    def _cron(self):
        if False:
            print('Hello World!')
        schedule = self.flow._flow_decorators.get('schedule')
        if schedule:
            schedule = schedule[0]
            if schedule.timezone is not None:
                raise StepFunctionsException('Step Functions does not support scheduling with a timezone.')
            return schedule.schedule
        return None

    def _process_parameters(self):
        if False:
            for i in range(10):
                print('nop')
        parameters = []
        has_schedule = self._cron() is not None
        seen = set()
        for (var, param) in self.flow._get_parameters():
            norm = param.name.lower()
            if norm in seen:
                raise MetaflowException('Parameter *%s* is specified twice. Note that parameter names are case-insensitive.' % param.name)
            seen.add(norm)
            is_required = param.kwargs.get('required', False)
            if 'default' not in param.kwargs and is_required and has_schedule:
                raise MetaflowException('The parameter *%s* does not have a default and is required. Scheduling such parameters via AWS Event Bridge is not currently supported.' % param.name)
            value = deploy_time_eval(param.kwargs.get('default'))
            parameters.append(dict(name=param.name, value=value))
        return parameters

    def _batch(self, node):
        if False:
            for i in range(10):
                print('nop')
        attrs = {'metaflow.user': 'SFN', 'metaflow.owner': self.username, 'metaflow.flow_name': self.flow.name, 'metaflow.step_name': node.name, 'metaflow.run_id.$': '$$.Execution.Name', 'metaflow.version': self.environment.get_environment_info()['metaflow_version'], 'step_name': node.name}
        if node.name == 'start':
            attrs['metaflow.production_token'] = self.production_token
        env_deco = [deco for deco in node.decorators if deco.name == 'environment']
        env = {}
        if env_deco:
            env = env_deco[0].attributes['vars'].copy()
        if S3_ENDPOINT_URL is not None:
            env['METAFLOW_S3_ENDPOINT_URL'] = S3_ENDPOINT_URL
        if node.name == 'start':
            parameters = self._process_parameters()
            if parameters:
                env['METAFLOW_PARAMETERS'] = '$.Parameters'
                default_parameters = {}
                for parameter in parameters:
                    if parameter['value'] is not None:
                        default_parameters[parameter['name']] = parameter['value']
                env['METAFLOW_DEFAULT_PARAMETERS'] = json.dumps(default_parameters)
            input_paths = None
        else:
            if node.parallel_foreach:
                raise StepFunctionsException('Parallel steps are not supported yet with AWS step functions.')
            if node.type == 'join' and self.graph[node.split_parents[-1]].type == 'foreach':
                input_paths = 'sfn-${METAFLOW_RUN_ID}/%s/:${METAFLOW_PARENT_TASK_IDS}' % node.in_funcs[0]
                env['METAFLOW_SPLIT_PARENT_TASK_ID'] = '$.Parameters.split_parent_task_id_%s' % node.split_parents[-1]
            elif len(node.in_funcs) == 1:
                input_paths = 'sfn-${METAFLOW_RUN_ID}/%s/${METAFLOW_PARENT_TASK_ID}' % node.in_funcs[0]
                env['METAFLOW_PARENT_TASK_ID'] = '$.JobId'
            else:
                input_paths = 'sfn-${METAFLOW_RUN_ID}:' + ','.join(('/${METAFLOW_PARENT_%s_STEP}/${METAFLOW_PARENT_%s_TASK_ID}' % (idx, idx) for (idx, _) in enumerate(node.in_funcs)))
                for (idx, _) in enumerate(node.in_funcs):
                    env['METAFLOW_PARENT_%s_TASK_ID' % idx] = '$.[%s].JobId' % idx
                    env['METAFLOW_PARENT_%s_STEP' % idx] = '$.[%s].Parameters.step_name' % idx
            env['METAFLOW_INPUT_PATHS'] = input_paths
            if node.is_inside_foreach:
                if any((self.graph[n].type == 'foreach' for n in node.in_funcs)):
                    attrs['split_parent_task_id_%s.$' % node.split_parents[-1]] = '$.SplitParentTaskId'
                    for parent in node.split_parents[:-1]:
                        if self.graph[parent].type == 'foreach':
                            attrs['split_parent_task_id_%s.$' % parent] = '$.Parameters.split_parent_task_id_%s' % parent
                elif node.type == 'join':
                    if self.graph[node.split_parents[-1]].type == 'foreach':
                        attrs['split_parent_task_id_%s.$' % node.split_parents[-1]] = '$.Parameters.split_parent_task_id_%s' % node.split_parents[-1]
                        for parent in node.split_parents[:-1]:
                            if self.graph[parent].type == 'foreach':
                                attrs['split_parent_task_id_%s.$' % parent] = '$.Parameters.split_parent_task_id_%s' % parent
                    else:
                        for parent in node.split_parents:
                            if self.graph[parent].type == 'foreach':
                                attrs['split_parent_task_id_%s.$' % parent] = '$.[0].Parameters.split_parent_task_id_%s' % parent
                else:
                    for parent in node.split_parents:
                        if self.graph[parent].type == 'foreach':
                            attrs['split_parent_task_id_%s.$' % parent] = '$.Parameters.split_parent_task_id_%s' % parent
                if any((self.graph[n].type == 'join' and self.graph[self.graph[n].split_parents[-1]].type == 'foreach' for n in node.out_funcs)):
                    env['METAFLOW_SPLIT_PARENT_TASK_ID_FOR_FOREACH_JOIN'] = attrs['split_parent_task_id_%s.$' % self.graph[node.out_funcs[0]].split_parents[-1]]
                if node.type == 'foreach':
                    if self.workflow_timeout:
                        env['METAFLOW_SFN_WORKFLOW_TIMEOUT'] = self.workflow_timeout
            if any((self.graph[n].type == 'foreach' for n in node.in_funcs)):
                env['METAFLOW_SPLIT_INDEX'] = '$.Index'
        env['METAFLOW_CODE_URL'] = self.code_package_url
        env['METAFLOW_FLOW_NAME'] = attrs['metaflow.flow_name']
        env['METAFLOW_STEP_NAME'] = attrs['metaflow.step_name']
        env['METAFLOW_RUN_ID'] = attrs['metaflow.run_id.$']
        env['METAFLOW_PRODUCTION_TOKEN'] = self.production_token
        env['SFN_STATE_MACHINE'] = self.name
        env['METAFLOW_OWNER'] = attrs['metaflow.owner']
        metadata_env = self.metadata.get_runtime_environment('step-functions')
        env.update(metadata_env)
        metaflow_version = self.environment.get_environment_info()
        metaflow_version['flow_name'] = self.graph.name
        metaflow_version['production_token'] = self.production_token
        env['METAFLOW_VERSION'] = json.dumps(metaflow_version)
        if node.type == 'foreach' or (node.is_inside_foreach and any((self.graph[n].type == 'join' and self.graph[self.graph[n].split_parents[-1]].type == 'foreach' for n in node.out_funcs))) or (node.type == 'join' and self.graph[node.split_parents[-1]].type == 'foreach'):
            if SFN_DYNAMO_DB_TABLE is None:
                raise StepFunctionsException('An AWS DynamoDB table is needed to support foreach in your flow. You can create one following the instructions listed at *https://admin-docs.metaflow.org/metaflow-on-aws/deployment-guide/manual-deployment#scheduling* and re-configure Metaflow using *metaflow configure aws* on your terminal.')
            env['METAFLOW_SFN_DYNAMO_DB_TABLE'] = SFN_DYNAMO_DB_TABLE
        env = {k: v for (k, v) in env.items() if v is not None}
        batch_deco = [deco for deco in node.decorators if deco.name == 'batch'][0]
        resources = {}
        resources.update(batch_deco.attributes)
        (user_code_retries, total_retries) = self._get_retries(node)
        task_spec = {'flow_name': attrs['metaflow.flow_name'], 'step_name': attrs['metaflow.step_name'], 'run_id': 'sfn-$METAFLOW_RUN_ID', 'task_id': '$AWS_BATCH_JOB_ID', 'retry_count': '$((AWS_BATCH_JOB_ATTEMPT-1))'}
        return Batch(self.metadata, self.environment).create_job(step_name=node.name, step_cli=self._step_cli(node, input_paths, self.code_package_url, user_code_retries), task_spec=task_spec, code_package_sha=self.code_package_sha, code_package_url=self.code_package_url, code_package_ds=self.flow_datastore.TYPE, image=resources['image'], queue=resources['queue'], iam_role=resources['iam_role'], execution_role=resources['execution_role'], cpu=resources['cpu'], gpu=resources['gpu'], memory=resources['memory'], run_time_limit=batch_deco.run_time_limit, shared_memory=resources['shared_memory'], max_swap=resources['max_swap'], swappiness=resources['swappiness'], efa=resources['efa'], use_tmpfs=resources['use_tmpfs'], tmpfs_tempdir=resources['tmpfs_tempdir'], tmpfs_size=resources['tmpfs_size'], tmpfs_path=resources['tmpfs_path'], inferentia=resources['inferentia'], env=env, attrs=attrs, host_volumes=resources['host_volumes']).attempts(total_retries + 1)

    def _get_retries(self, node):
        if False:
            while True:
                i = 10
        max_user_code_retries = 0
        max_error_retries = 0
        for deco in node.decorators:
            (user_code_retries, error_retries) = deco.step_task_retry_count()
            max_user_code_retries = max(max_user_code_retries, user_code_retries)
            max_error_retries = max(max_error_retries, error_retries)
        return (max_user_code_retries, max_user_code_retries + max_error_retries)

    def _step_cli(self, node, paths, code_package_url, user_code_retries):
        if False:
            i = 10
            return i + 15
        cmds = []
        script_name = os.path.basename(sys.argv[0])
        executable = self.environment.executable(node.name)
        if R.use_r():
            entrypoint = [R.entrypoint()]
        else:
            entrypoint = [executable, script_name]
        task_id = '${AWS_BATCH_JOB_ID}'
        top_opts_dict = {'with': [decorator.make_decorator_spec() for decorator in node.decorators if not decorator.statically_defined]}
        for deco in flow_decorators():
            top_opts_dict.update(deco.get_top_level_options())
        top_opts = list(dict_to_cli_options(top_opts_dict))
        top_level = top_opts + ['--quiet', '--metadata=%s' % self.metadata.TYPE, '--environment=%s' % self.environment.TYPE, '--datastore=%s' % self.flow_datastore.TYPE, '--datastore-root=%s' % self.flow_datastore.datastore_root, '--event-logger=%s' % self.event_logger.TYPE, '--monitor=%s' % self.monitor.TYPE, '--no-pylint', '--with=step_functions_internal']
        if node.name == 'start':
            task_id_params = '%s-params' % task_id
            param_file = ''.join((random.choice(string.ascii_lowercase) for _ in range(10)))
            export_params = 'python -m metaflow.plugins.aws.step_functions.set_batch_environment parameters %s && . `pwd`/%s' % (param_file, param_file)
            params = entrypoint + top_level + ['init', '--run-id sfn-$METAFLOW_RUN_ID', '--task-id %s' % task_id_params]
            if self.tags:
                params.extend(('--tag %s' % tag for tag in self.tags))
            exists = entrypoint + ['dump', '--max-value-size=0', 'sfn-${METAFLOW_RUN_ID}/_parameters/%s' % task_id_params]
            cmd = 'if ! %s >/dev/null 2>/dev/null; then %s && %s; fi' % (' '.join(exists), export_params, ' '.join(params))
            cmds.append(cmd)
            paths = 'sfn-${METAFLOW_RUN_ID}/_parameters/%s' % task_id_params
        if node.type == 'join' and self.graph[node.split_parents[-1]].type == 'foreach':
            parent_tasks_file = ''.join((random.choice(string.ascii_lowercase) for _ in range(10)))
            export_parent_tasks = 'python -m metaflow.plugins.aws.step_functions.set_batch_environment parent_tasks %s && . `pwd`/%s' % (parent_tasks_file, parent_tasks_file)
            cmds.append(export_parent_tasks)
        step = ['step', node.name, '--run-id sfn-$METAFLOW_RUN_ID', '--task-id %s' % task_id, '--retry-count $((AWS_BATCH_JOB_ATTEMPT-1))', '--max-user-code-retries %d' % user_code_retries, '--input-paths %s' % paths]
        if any((self.graph[n].type == 'foreach' for n in node.in_funcs)):
            step.append('--split-index $METAFLOW_SPLIT_INDEX')
        if self.tags:
            step.extend(('--tag %s' % tag for tag in self.tags))
        if self.namespace is not None:
            step.append('--namespace=%s' % self.namespace)
        cmds.append(' '.join(entrypoint + top_level + step))
        return ' && '.join(cmds)

class Workflow(object):

    def __init__(self, name):
        if False:
            while True:
                i = 10
        self.name = name
        tree = lambda : defaultdict(tree)
        self.payload = tree()

    def start_at(self, start_at):
        if False:
            print('Hello World!')
        self.payload['StartAt'] = start_at
        return self

    def add_state(self, state):
        if False:
            for i in range(10):
                print('nop')
        self.payload['States'][state.name] = state.payload
        return self

    def timeout_seconds(self, timeout_seconds):
        if False:
            while True:
                i = 10
        self.payload['TimeoutSeconds'] = timeout_seconds
        return self

    def to_json(self, pretty=False):
        if False:
            return 10
        return json.dumps(self.payload, indent=4 if pretty else None)

class State(object):

    def __init__(self, name):
        if False:
            while True:
                i = 10
        self.name = name
        tree = lambda : defaultdict(tree)
        self.payload = tree()
        self.payload['Type'] = 'Task'

    def resource(self, resource):
        if False:
            return 10
        self.payload['Resource'] = resource
        return self

    def next(self, state):
        if False:
            while True:
                i = 10
        self.payload['Next'] = state
        return self

    def end(self):
        if False:
            i = 10
            return i + 15
        self.payload['End'] = True
        return self

    def parameter(self, name, value):
        if False:
            while True:
                i = 10
        self.payload['Parameters'][name] = value
        return self

    def output_path(self, output_path):
        if False:
            print('Hello World!')
        self.payload['OutputPath'] = output_path
        return self

    def result_path(self, result_path):
        if False:
            for i in range(10):
                print('nop')
        self.payload['ResultPath'] = result_path
        return self

    def _partition(self):
        if False:
            print('Hello World!')
        return SFN_IAM_ROLE.split(':')[1]

    def batch(self, job):
        if False:
            for i in range(10):
                print('nop')
        self.resource('arn:%s:states:::batch:submitJob.sync' % self._partition()).parameter('JobDefinition', job.payload['jobDefinition']).parameter('JobName', job.payload['jobName']).parameter('JobQueue', job.payload['jobQueue']).parameter('Parameters', job.payload['parameters']).parameter('ContainerOverrides', to_pascalcase(job.payload['containerOverrides'])).parameter('RetryStrategy', to_pascalcase(job.payload['retryStrategy'])).parameter('Timeout', to_pascalcase(job.payload['timeout']))
        if 'tags' in job.payload:
            self.parameter('Tags', job.payload['tags'])
        return self

    def dynamo_db(self, table_name, primary_key, values):
        if False:
            for i in range(10):
                print('nop')
        self.resource('arn:%s:states:::dynamodb:getItem' % self._partition()).parameter('TableName', table_name).parameter('Key', {'pathspec': {'S.$': primary_key}}).parameter('ConsistentRead', True).parameter('ProjectionExpression', values)
        return self

class Parallel(object):

    def __init__(self, name):
        if False:
            return 10
        self.name = name
        tree = lambda : defaultdict(tree)
        self.payload = tree()
        self.payload['Type'] = 'Parallel'

    def branch(self, workflow):
        if False:
            while True:
                i = 10
        if 'Branches' not in self.payload:
            self.payload['Branches'] = []
        self.payload['Branches'].append(workflow.payload)
        return self

    def next(self, state):
        if False:
            while True:
                i = 10
        self.payload['Next'] = state
        return self

    def output_path(self, output_path):
        if False:
            i = 10
            return i + 15
        self.payload['OutputPath'] = output_path
        return self

    def result_path(self, result_path):
        if False:
            print('Hello World!')
        self.payload['ResultPath'] = result_path
        return self

class Map(object):

    def __init__(self, name):
        if False:
            print('Hello World!')
        self.name = name
        tree = lambda : defaultdict(tree)
        self.payload = tree()
        self.payload['Type'] = 'Map'
        self.payload['MaxConcurrency'] = 0

    def iterator(self, workflow):
        if False:
            i = 10
            return i + 15
        self.payload['Iterator'] = workflow.payload
        return self

    def next(self, state):
        if False:
            return 10
        self.payload['Next'] = state
        return self

    def items_path(self, items_path):
        if False:
            print('Hello World!')
        self.payload['ItemsPath'] = items_path
        return self

    def parameter(self, name, value):
        if False:
            i = 10
            return i + 15
        self.payload['Parameters'][name] = value
        return self

    def max_concurrency(self, max_concurrency):
        if False:
            for i in range(10):
                print('nop')
        self.payload['MaxConcurrency'] = max_concurrency
        return self

    def output_path(self, output_path):
        if False:
            return 10
        self.payload['OutputPath'] = output_path
        return self

    def result_path(self, result_path):
        if False:
            print('Hello World!')
        self.payload['ResultPath'] = result_path
        return self