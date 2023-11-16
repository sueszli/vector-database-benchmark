"""
Purpose

Shows how to use the AWS SDK for Python (Boto3) with AWS Step Functions to
create and manage activities. An activity is used by a state machine to pause its
execution and let external code get current state data and send a response before the
state machine is resumed.
"""
import logging
from botocore.exceptions import ClientError
logger = logging.getLogger(__name__)

class Activity:
    """Encapsulates Step Function activity actions."""

    def __init__(self, stepfunctions_client):
        if False:
            print('Hello World!')
        '\n        :param stepfunctions_client: A Boto3 Step Functions client.\n        '
        self.stepfunctions_client = stepfunctions_client

    def create(self, name):
        if False:
            print('Hello World!')
        '\n        Create an activity.\n\n        :param name: The name of the activity to create.\n        :return: The Amazon Resource Name (ARN) of the newly created activity.\n        '
        try:
            response = self.stepfunctions_client.create_activity(name=name)
        except ClientError as err:
            logger.error("Couldn't create activity %s. Here's why: %s: %s", name, err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return response['activityArn']

    def find(self, name):
        if False:
            print('Hello World!')
        '\n        Find an activity by name. This requires listing activities until one is found\n        with a matching name.\n\n        :param name: The name of the activity to search for.\n        :return: If found, the ARN of the activity; otherwise, None.\n        '
        try:
            paginator = self.stepfunctions_client.get_paginator('list_activities')
            for page in paginator.paginate():
                for activity in page.get('activities', []):
                    if activity['name'] == name:
                        return activity['activityArn']
        except ClientError as err:
            logger.error("Couldn't list activities. Here's why: %s: %s", err.response['Error']['Code'], err.response['Error']['Message'])
            raise

    def get_task(self, activity_arn):
        if False:
            return 10
        '\n        Gets task data for an activity. When a state machine is waiting for the\n        specified activity, a response is returned with data from the state machine.\n        When a state machine is not waiting, this call blocks for 60 seconds.\n\n        :param activity_arn: The ARN of the activity to get task data for.\n        :return: The task data for the activity.\n        '
        try:
            response = self.stepfunctions_client.get_activity_task(activityArn=activity_arn)
        except ClientError as err:
            logger.error("Couldn't get a task for activity %s. Here's why: %s: %s", activity_arn, err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return response

    def send_task_success(self, task_token, task_response):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sends a success response to a waiting activity step. A state machine with an\n        activity step waits for the activity to get task data and then respond with\n        either success or failure before it resumes processing.\n\n        :param task_token: The token associated with the task. This is included in the\n                           response to the get_activity_task action and must be sent\n                           without modification.\n        :param task_response: The response data from the activity. This data is\n                              received and processed by the state machine.\n        '
        try:
            self.stepfunctions_client.send_task_success(taskToken=task_token, output=task_response)
        except ClientError as err:
            logger.error("Couldn't send task success. Here's why: %s: %s", err.response['Error']['Code'], err.response['Error']['Message'])
            raise

    def delete(self, activity_arn):
        if False:
            i = 10
            return i + 15
        '\n        Delete an activity.\n\n        :param activity_arn: The ARN of the activity to delete.\n        '
        try:
            response = self.stepfunctions_client.delete_activity(activityArn=activity_arn)
        except ClientError as err:
            logger.error("Couldn't delete activity %s. Here's why: %s: %s", activity_arn, err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return response