"""
Purpose

Shows how to use the AWS SDK for Python (Boto3) with AWS Step Functions to
do the following:

1. Create an activity.
2. Create a state machine from an Amazon States Language definition that contains the
previously created activity as a step.
3. Run the state machine and respond to the activity with user input.
4. Get the final status and output after the run completes.
5. Delete resources created by the example.
"""
import json
import logging
import sys
import boto3
from botocore.exceptions import ClientError
from activities import Activity
from state_machines import StateMachine
sys.path.append('../..')
import demo_tools.question as q
from demo_tools.retries import wait
logger = logging.getLogger(__name__)

class StateMachineScenario:
    """Runs an interactive scenario that shows how to get started using Step Functions."""

    def __init__(self, activity, state_machine, iam_client):
        if False:
            return 10
        '\n        :param activity: An object that wraps activity actions.\n        :param state_machine: An object that wraps state machine actions.\n        :param iam_client: A Boto3 AWS Identity and Access Management (IAM) client.\n        '
        self.activity = activity
        self.state_machine = state_machine
        self.iam_client = iam_client
        self.state_machine_role = None

    def prerequisites(self, state_machine_role_name):
        if False:
            while True:
                i = 10
        '\n        Finds or creates an IAM role that can be assumed by Step Functions.\n        A role of this kind is required to create a state machine.\n        The state machine used in this example does not call any additional services,\n        so it needs no additional permissions.\n\n        :param state_machine_role_name: The name of the role.\n        :return: Data about the role.\n        '
        trust_policy = {'Version': '2012-10-17', 'Statement': [{'Sid': '', 'Effect': 'Allow', 'Principal': {'Service': 'states.amazonaws.com'}, 'Action': 'sts:AssumeRole'}]}
        try:
            role = self.iam_client.get_role(RoleName=state_machine_role_name)
            print(f'Prerequisite IAM role {state_machine_role_name} already exists.')
        except ClientError as err:
            if err.response['Error']['Code'] == 'NoSuchEntity':
                role = None
            else:
                logger.error("Couldn't get prerequisite IAM role %s. Here's why: %s: %s", state_machine_role_name, err.response['Error']['Code'], err.response['Error']['Message'])
                raise
        if role is None:
            try:
                role = self.iam_client.create_role(RoleName=state_machine_role_name, AssumeRolePolicyDocument=json.dumps(trust_policy))
            except ClientError as err:
                logger.error("Couldn't create prerequisite IAM role %s. Here's why: %s: %s", state_machine_role_name, err.response['Error']['Code'], err.response['Error']['Message'])
                raise
        self.state_machine_role = role['Role']

    def find_or_create_activity(self, activity_name):
        if False:
            return 10
        '\n        Finds or creates a Step Functions activity.\n\n        :param activity_name: The name of the activity.\n        :return: The Amazon Resource Name (ARN) of the activity.\n        '
        print("First, let's set up an activity and state machine.")
        activity_arn = self.activity.find(activity_name)
        if activity_arn is None:
            activity_arn = self.activity.create(activity_name)
            print(f'Activity {activity_name} created. Its Amazon Resource Name (ARN) is {activity_arn}.')
        else:
            print(f'Activity {activity_name} already exists.')
        return activity_arn

    def find_or_create_state_machine(self, state_machine_name, activity_arn, state_machine_file):
        if False:
            print('Hello World!')
        "\n        Finds or creates a Step Functions state machine.\n\n        :param state_machine_name: The name of the state machine.\n        :param activity_arn: The ARN of an activity that is used as a step in the state\n                             machine. This ARN is injected into the state machine\n                             definition that's used to create the state machine.\n        :param state_machine_file: The path to a file containing the state machine\n                                   definition.\n        :return: The ARN of the state machine.\n        "
        state_machine_arn = self.state_machine.find(state_machine_name)
        if state_machine_arn is None:
            with open(state_machine_file) as state_machine_file:
                state_machine_def = state_machine_file.read().replace('{{DOC_EXAMPLE_ACTIVITY_ARN}}', activity_arn)
                state_machine_arn = self.state_machine.create(state_machine_name, state_machine_def, self.state_machine_role['Arn'])
            print(f'State machine {state_machine_name} created.')
        else:
            print(f'State machine {state_machine_name} already exists.')
        print('-' * 88)
        print(f"Here's some information about state machine {state_machine_name}:")
        state_machine_info = self.state_machine.describe(state_machine_arn)
        for field in ['name', 'status', 'stateMachineArn', 'roleArn']:
            print(f'\t{field}: {state_machine_info[field]}')
        return state_machine_arn

    def run_state_machine(self, state_machine_arn, activity_arn):
        if False:
            i = 10
            return i + 15
        '\n        Run the state machine. The state machine used in this example is a simple\n        chat simulation. It contains an activity step in a loop that is used for user\n        interaction. When the state machine gets to the activity step, it waits for\n        an external application to get task data and submit a response. This function\n        acts as the activity application by getting task input and responding with\n        user input.\n\n        :param state_machine_arn: The ARN of the state machine.\n        :param activity_arn: The ARN of the activity used as a step in the state machine.\n        :return: The ARN of the run.\n        '
        print(f"Let's run the state machine. It's a simplistic, non-AI chat simulator we'll call ChatSFN.")
        user_name = q.ask('What should ChatSFN call you? ', q.non_empty)
        run_input = {'name': user_name}
        print('Starting state machine...')
        run_arn = self.state_machine.start(state_machine_arn, json.dumps(run_input))
        action = None
        while action != 'done':
            activity_task = self.activity.get_task(activity_arn)
            task_input = json.loads(activity_task['input'])
            print(f"ChatSFN: {task_input['message']}")
            action = task_input['actions'][q.choose('What now? ', task_input['actions'])]
            task_response = {'action': action}
            self.activity.send_task_success(activity_task['taskToken'], json.dumps(task_response))
        return run_arn

    def finish_state_machine_run(self, run_arn):
        if False:
            for i in range(10):
                print('nop')
        '\n        Wait for the state machine run to finish, then print final status and output.\n\n        :param run_arn: The ARN of the run to retrieve.\n        '
        print(f"Let's get the final output from the state machine:")
        status = 'RUNNING'
        while status == 'RUNNING':
            run_output = self.state_machine.describe_run(run_arn)
            status = run_output['status']
            if status == 'RUNNING':
                print("The state machine is still running, let's wait for it to finish.")
                wait(1)
            elif status == 'SUCCEEDED':
                print(f"ChatSFN: {json.loads(run_output['output'])['message']}")
            else:
                print(f'Run status: {status}.')

    def cleanup(self, state_machine_name, state_machine_arn, activity_name, activity_arn, state_machine_role_name):
        if False:
            while True:
                i = 10
        '\n        Clean up resources created by this example.\n\n        :param state_machine_name: The name of the state machine.\n        :param state_machine_arn: The ARN of the state machine.\n        :param activity_name: The name of the activity.\n        :param activity_arn: The ARN of the activity.\n        :param state_machine_role_name: The name of the role used by the state machine.\n        '
        if q.ask('Do you want to delete the state machine, activity, and role created for this example? (y/n) ', q.is_yesno):
            self.state_machine.delete(state_machine_arn)
            print(f'Deleted state machine {state_machine_name}.')
            self.activity.delete(activity_arn)
            print(f'Deleted activity {activity_name}.')
            self.iam_client.delete_role(RoleName=state_machine_role_name)
            print(f'Deleted role {state_machine_role_name}.')

    def run_scenario(self, activity_name, state_machine_name):
        if False:
            i = 10
            return i + 15
        print('-' * 88)
        print('Welcome to the AWS Step Functions state machines demo.')
        print('-' * 88)
        activity_arn = self.find_or_create_activity(activity_name)
        state_machine_arn = self.find_or_create_state_machine(state_machine_name, activity_arn, '../../../resources/sample_files/chat_sfn_state_machine.json')
        print('-' * 88)
        run_arn = self.run_state_machine(state_machine_arn, activity_arn)
        print('-' * 88)
        self.finish_state_machine_run(run_arn)
        print('-' * 88)
        self.cleanup(state_machine_name, state_machine_arn, activity_name, activity_arn, self.state_machine_role['RoleName'])
        print('-' * 88)
        print('\nThanks for watching!')
        print('-' * 88)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    try:
        stepfunctions_client = boto3.client('stepfunctions')
        iam_client = boto3.client('iam')
        scenario = StateMachineScenario(Activity(stepfunctions_client), StateMachine(stepfunctions_client), iam_client)
        scenario.prerequisites('doc-example-state-machine-chat')
        scenario.run_scenario('doc-example-activity', 'doc-example-state-machine')
    except Exception:
        logging.exception('Something went wrong with the demo.')