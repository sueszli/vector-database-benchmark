from metaflow.metaflow_config import AWS_SANDBOX_ENABLED, AWS_SANDBOX_REGION, SFN_EXECUTION_LOG_GROUP_ARN

class StepFunctionsClient(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        from ..aws_client import get_aws_client
        self._client = get_aws_client('stepfunctions')

    def search(self, name):
        if False:
            return 10
        paginator = self._client.get_paginator('list_state_machines')
        return next((state_machine for page in paginator.paginate() for state_machine in page['stateMachines'] if state_machine['name'] == name), None)

    def push(self, name, definition, role_arn, log_execution_history):
        if False:
            while True:
                i = 10
        try:
            response = self._client.create_state_machine(name=name, definition=definition, roleArn=role_arn, loggingConfiguration=self._default_logging_configuration(log_execution_history))
            state_machine_arn = response['stateMachineArn']
        except self._client.exceptions.StateMachineAlreadyExists as e:
            state_machine_arn = e.response['Error']['Message'].split("'")[1]
            self._client.update_state_machine(stateMachineArn=state_machine_arn, definition=definition, roleArn=role_arn, loggingConfiguration=self._default_logging_configuration(log_execution_history))
        return state_machine_arn

    def get(self, name):
        if False:
            for i in range(10):
                print('nop')
        state_machine_arn = self.get_state_machine_arn(name)
        if state_machine_arn is None:
            return None
        try:
            return self._client.describe_state_machine(stateMachineArn=state_machine_arn)
        except self._client.exceptions.StateMachineDoesNotExist:
            return None

    def trigger(self, state_machine_arn, input):
        if False:
            while True:
                i = 10
        return self._client.start_execution(stateMachineArn=state_machine_arn, input=input)

    def list_executions(self, state_machine_arn, states):
        if False:
            return 10
        if len(states) > 0:
            return (execution for state in states for page in self._client.get_paginator('list_executions').paginate(stateMachineArn=state_machine_arn, statusFilter=state) for execution in page['executions'])
        return (execution for page in self._client.get_paginator('list_executions').paginate(stateMachineArn=state_machine_arn) for execution in page['executions'])

    def terminate_execution(self, state_machine_arn, execution_arn):
        if False:
            return 10
        pass

    def _default_logging_configuration(self, log_execution_history):
        if False:
            i = 10
            return i + 15
        if log_execution_history:
            return {'level': 'ALL', 'includeExecutionData': True, 'destinations': [{'cloudWatchLogsLogGroup': {'logGroupArn': SFN_EXECUTION_LOG_GROUP_ARN}}]}
        else:
            return {'level': 'OFF'}

    def get_state_machine_arn(self, name):
        if False:
            return 10
        if AWS_SANDBOX_ENABLED:
            from ..aws_client import get_aws_client
            account_id = get_aws_client('sts').get_caller_identity().get('Account')
            region = AWS_SANDBOX_REGION
            return 'arn:aws:states:%s:%s:stateMachine:%s' % (region, account_id, name)
        else:
            state_machine = self.search(name)
            if state_machine:
                return state_machine['stateMachineArn']
            return None

    def delete(self, name):
        if False:
            i = 10
            return i + 15
        state_machine_arn = self.get_state_machine_arn(name)
        if state_machine_arn is None:
            return None
        return self._client.delete_state_machine(stateMachineArn=state_machine_arn)