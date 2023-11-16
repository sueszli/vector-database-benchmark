"""
Purpose

Shows how to use the AWS SDK for Python (Boto3) with AWS Step Functions to create and
manage state machines.
"""
import logging
from botocore.exceptions import ClientError
logger = logging.getLogger(__name__)

class StateMachine:
    """Encapsulates Step Functions state machine actions."""

    def __init__(self, stepfunctions_client):
        if False:
            print('Hello World!')
        '\n        :param stepfunctions_client: A Boto3 Step Functions client.\n        '
        self.stepfunctions_client = stepfunctions_client

    def create(self, name, definition, role_arn):
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates a state machine with the specific definition. The state machine assumes\n        the provided role before it starts a run.\n\n        :param name: The name to give the state machine.\n        :param definition: The Amazon States Language definition of the steps in the\n                           the state machine.\n        :param role_arn: The Amazon Resource Name (ARN) of the role that is assumed by\n                         Step Functions when the state machine is run.\n        :return: The ARN of the newly created state machine.\n        '
        try:
            response = self.stepfunctions_client.create_state_machine(name=name, definition=definition, roleArn=role_arn)
        except ClientError as err:
            logger.error("Couldn't create state machine %s. Here's why: %s: %s", name, err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return response['stateMachineArn']

    def find(self, name):
        if False:
            for i in range(10):
                print('nop')
        '\n        Find a state machine by name. This requires listing the state machines until\n        one is found with a matching name.\n\n        :param name: The name of the state machine to search for.\n        :return: The ARN of the state machine if found; otherwise, None.\n        '
        try:
            paginator = self.stepfunctions_client.get_paginator('list_state_machines')
            for page in paginator.paginate():
                for state_machine in page.get('stateMachines', []):
                    if state_machine['name'] == name:
                        return state_machine['stateMachineArn']
        except ClientError as err:
            logger.error("Couldn't list state machines. Here's why: %s: %s", err.response['Error']['Code'], err.response['Error']['Message'])
            raise

    def describe(self, state_machine_arn):
        if False:
            print('Hello World!')
        '\n        Get data about a state machine.\n\n        :param state_machine_arn: The ARN of the state machine to look up.\n        :return: The retrieved state machine data.\n        '
        try:
            response = self.stepfunctions_client.describe_state_machine(stateMachineArn=state_machine_arn)
        except ClientError as err:
            logger.error("Couldn't describe state machine %s. Here's why: %s: %s", state_machine_arn, err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return response

    def start(self, state_machine_arn, run_input):
        if False:
            while True:
                i = 10
        '\n        Start a run of a state machine with a specified input. A run is also known\n        as an "execution" in Step Functions.\n\n        :param state_machine_arn: The ARN of the state machine to run.\n        :param run_input: The input to the state machine, in JSON format.\n        :return: The ARN of the run. This can be used to get information about the run,\n                 including its current status and final output.\n        '
        try:
            response = self.stepfunctions_client.start_execution(stateMachineArn=state_machine_arn, input=run_input)
        except ClientError as err:
            logger.error("Couldn't start state machine %s. Here's why: %s: %s", state_machine_arn, err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return response['executionArn']

    def describe_run(self, run_arn):
        if False:
            print('Hello World!')
        '\n        Get data about a state machine run, such as its current status or final output.\n\n        :param run_arn: The ARN of the run to look up.\n        :return: The retrieved run data.\n        '
        try:
            response = self.stepfunctions_client.describe_execution(executionArn=run_arn)
        except ClientError as err:
            logger.error("Couldn't describe run %s. Here's why: %s: %s", run_arn, err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return response

    def delete(self, state_machine_arn):
        if False:
            i = 10
            return i + 15
        '\n        Delete a state machine and all of its run data.\n\n        :param state_machine_arn: The ARN of the state machine to delete.\n        '
        try:
            response = self.stepfunctions_client.delete_state_machine(stateMachineArn=state_machine_arn)
        except ClientError as err:
            logger.error("Couldn't delete state machine %s. Here's why: %s: %s", state_machine_arn, err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return response