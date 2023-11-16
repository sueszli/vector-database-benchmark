"""
Purpose

Shows how to use the AWS SDK for Python (Boto3) with AWS Step Functions to
create and run state machines.
"""
import json
import logging
from botocore.exceptions import ClientError
logger = logging.getLogger(__name__)

class StepFunctionsStateMachine:
    """Encapsulates Step Functions state machine functions."""

    def __init__(self, stepfunctions_client):
        if False:
            print('Hello World!')
        '\n        :param stepfunctions_client: A Boto3 Step Functions client.\n        '
        self.stepfunctions_client = stepfunctions_client
        self.state_machine_name = None
        self.state_machine_arn = None

    def _clear(self):
        if False:
            while True:
                i = 10
        '\n        Clears the object of its instance data.\n        '
        self.state_machine_name = None
        self.state_machine_arn = None

    def create(self, name, definition, role_arn):
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates a new state machine.\n\n        :param name: The name of the new state machine.\n        :param definition: A dict that contains all of the state and flow control\n                           information. The dict is translated to JSON before it is\n                           uploaded.\n        :param role_arn: A role that grants Step Functions permission to access any\n                         AWS services that are specified in the definition.\n        :return: The Amazon Resource Name (ARN) of the new state machine.\n        '
        try:
            response = self.stepfunctions_client.create_state_machine(name=name, definition=json.dumps(definition), roleArn=role_arn)
            self.state_machine_name = name
            self.state_machine_arn = response['stateMachineArn']
            logger.info('Created state machine %s. ARN is %s.', name, self.state_machine_arn)
        except ClientError:
            logger.exception("Couldn't create state machine %s.", name)
            raise
        else:
            return self.state_machine_arn

    def update(self, definition, role_arn=None):
        if False:
            i = 10
            return i + 15
        '\n        Updates an existing state machine. Any runs currently operating do not update\n        until they are stopped.\n\n        :param definition: A dict that contains all of the state and flow control\n                           information for the state machine. This completely replaces\n                           the existing definition.\n        :param role_arn: A role that grants Step Functions permission to access any\n                         AWS services that are specified in the definition.\n        '
        if self.state_machine_arn is None:
            raise ValueError
        try:
            kwargs = {'stateMachineArn': self.state_machine_arn, 'definition': json.dumps(definition)}
            if role_arn is not None:
                kwargs['roleArn'] = role_arn
            self.stepfunctions_client.update_state_machine(**kwargs)
            logger.info('Updated state machine %s.', self.state_machine_name)
        except ClientError:
            logger.exception("Couldn't update state machine %s.", self.state_machine_name)
            raise

    def delete(self):
        if False:
            while True:
                i = 10
        '\n        Deletes a state machine and all associated run information.\n        '
        if self.state_machine_arn is None:
            raise ValueError
        try:
            self.stepfunctions_client.delete_state_machine(stateMachineArn=self.state_machine_arn)
            logger.info('Deleted state machine %s.', self.state_machine_name)
            self._clear()
        except ClientError:
            logger.exception("Couldn't delete state machine %s.", self.state_machine_name)
            raise

    def find(self, state_machine_name):
        if False:
            print('Hello World!')
        '\n        Finds a state machine by name. This function iterates the state machines for\n        the current account until it finds a match and returns the first matching\n        state machine.\n\n        :param state_machine_name: The name of the state machine to find.\n        :return: The ARN of the named state machine when found; otherwise, None.\n        '
        self._clear()
        try:
            paginator = self.stepfunctions_client.get_paginator('list_state_machines')
            for page in paginator.paginate():
                for machine in page['stateMachines']:
                    if machine['name'] == state_machine_name:
                        self.state_machine_name = state_machine_name
                        self.state_machine_arn = machine['stateMachineArn']
                        break
                if self.state_machine_arn is not None:
                    break
            if self.state_machine_arn is not None:
                logger.info('Found state machine %s with ARN %s.', self.state_machine_name, self.state_machine_arn)
            else:
                logger.info("Couldn't find state machine %s.", state_machine_name)
        except ClientError:
            logger.exception("Couldn't find state machine %s.", state_machine_name)
            raise
        else:
            return self.state_machine_arn

    def describe(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Gets metadata about a state machine.\n\n        :return: The metadata about the state machine.\n        '
        if self.state_machine_arn is None:
            raise ValueError
        try:
            response = self.stepfunctions_client.describe_state_machine(stateMachineArn=self.state_machine_arn)
            logger.info('Got metadata for state machine %s.', self.state_machine_name)
        except ClientError:
            logger.exception("Couldn't get metadata for state machine %s.", self.state_machine_name)
            raise
        else:
            return response

    def start_run(self, run_name, run_input=None):
        if False:
            print('Hello World!')
        '\n        Starts a run with the current state definition.\n\n        :param run_name: The name of the run. This name must be unique for all runs\n                         for the state machine.\n        :param run_input: Data that is passed as input to the run.\n        :return: The ARN of the run.\n        '
        if self.state_machine_arn is None:
            raise ValueError
        try:
            kwargs = {'stateMachineArn': self.state_machine_arn, 'name': run_name}
            if run_input is not None:
                kwargs['input'] = json.dumps(run_input)
            response = self.stepfunctions_client.start_execution(**kwargs)
            run_arn = response['executionArn']
            logger.info('Started run %s. ARN is %s.', run_name, run_arn)
        except ClientError:
            logger.exception("Couldn't start run %s.", run_name)
            raise
        else:
            return run_arn

    def list_runs(self, run_status=None):
        if False:
            print('Hello World!')
        '\n        Lists the runs for the state machine.\n\n        :param run_status: When specified, only lists runs that have the specified\n                           status. Otherwise, all runs are listed.\n        :return: The list of runs.\n        '
        if self.state_machine_arn is None:
            raise ValueError
        try:
            kwargs = {'stateMachineArn': self.state_machine_arn}
            if run_status is not None:
                kwargs['statusFilter'] = run_status
            response = self.stepfunctions_client.list_executions(**kwargs)
            runs = response['executions']
            logger.info('Got %s runs for state machine %s.', len(runs), self.state_machine_name)
        except ClientError:
            logger.exception("Couldn't get runs for state machine %s.", self.state_machine_name)
            raise
        else:
            return runs

    def stop_run(self, run_arn, cause):
        if False:
            print('Hello World!')
        '\n        Stops a run.\n\n        :param run_arn: The run to stop.\n        :param cause: A description of why the run was stopped.\n        '
        try:
            self.stepfunctions_client.stop_execution(executionArn=run_arn, cause=cause)
            logger.info('Stopping run %s.', run_arn)
        except ClientError:
            logger.exception("Couldn't stop run %s.", run_arn)
            raise