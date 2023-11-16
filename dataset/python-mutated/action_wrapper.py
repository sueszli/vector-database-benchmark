"""
Purpose

Shows how to use the AWS SDK for Python (Boto3) with Amazon EC2 Auto Scaling to
manage groups and instances.
"""
import logging
from botocore.exceptions import ClientError
logger = logging.getLogger(__name__)

class AutoScalingWrapper:
    """Encapsulates Amazon EC2 Auto Scaling actions."""

    def __init__(self, autoscaling_client):
        if False:
            return 10
        '\n        :param autoscaling_client: A Boto3 Amazon EC2 Auto Scaling client.\n        '
        self.autoscaling_client = autoscaling_client

    def create_group(self, group_name, group_zones, launch_template_name, min_size, max_size):
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates an Auto Scaling group.\n\n        :param group_name: The name to give to the group.\n        :param group_zones: The Availability Zones in which instances can be created.\n        :param launch_template_name: The name of an existing Amazon EC2 launch template.\n                                     The launch template specifies the configuration of\n                                     instances that are created by auto scaling activities.\n        :param min_size: The minimum number of active instances in the group.\n        :param max_size: The maximum number of active instances in the group.\n        '
        try:
            self.autoscaling_client.create_auto_scaling_group(AutoScalingGroupName=group_name, AvailabilityZones=group_zones, LaunchTemplate={'LaunchTemplateName': launch_template_name, 'Version': '$Default'}, MinSize=min_size, MaxSize=max_size)
        except ClientError as err:
            logger.error("Couldn't create group %s. Here's why: %s: %s", group_name, err.response['Error']['Code'], err.response['Error']['Message'])
            raise

    def update_group(self, group_name, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Updates an Auto Scaling group.\n\n        :param group_name: The name of the group to update.\n        :param kwargs: Keyword arguments to pass through to the service.\n        '
        try:
            self.autoscaling_client.update_auto_scaling_group(AutoScalingGroupName=group_name, **kwargs)
        except ClientError as err:
            logger.error("Couldn't update group %s. Here's why: %s: %s", group_name, err.response['Error']['Code'], err.response['Error']['Message'])
            raise

    def delete_group(self, group_name):
        if False:
            while True:
                i = 10
        '\n        Deletes an Auto Scaling group. All instances must be stopped before the\n        group can be deleted.\n\n        :param group_name: The name of the group to delete.\n        '
        try:
            self.autoscaling_client.delete_auto_scaling_group(AutoScalingGroupName=group_name)
        except ClientError as err:
            logger.error("Couldn't delete group %s. Here's why: %s: %s", group_name, err.response['Error']['Code'], err.response['Error']['Message'])
            raise

    def describe_group(self, group_name):
        if False:
            for i in range(10):
                print('nop')
        '\n        Gets information about an Auto Scaling group.\n\n        :param group_name: The name of the group to look up.\n        :return: Information about the group, if found.\n        '
        try:
            response = self.autoscaling_client.describe_auto_scaling_groups(AutoScalingGroupNames=[group_name])
        except ClientError as err:
            logger.error("Couldn't describe group %s. Here's why: %s: %s", group_name, err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            groups = response.get('AutoScalingGroups', [])
            return groups[0] if len(groups) > 0 else None

    def terminate_instance(self, instance_id, decrease_capacity):
        if False:
            i = 10
            return i + 15
        '\n        Stops an instance.\n\n        :param instance_id: The ID of the instance to stop.\n        :param decrease_capacity: Specifies whether to decrease the desired capacity\n                                  of the group. When passing True for this parameter,\n                                  you can stop an instance without having a replacement\n                                  instance start when the desired capacity threshold is\n                                  crossed.\n        :return: The scaling activity that occurs in response to this action.\n        '
        try:
            response = self.autoscaling_client.terminate_instance_in_auto_scaling_group(InstanceId=instance_id, ShouldDecrementDesiredCapacity=decrease_capacity)
        except ClientError as err:
            logger.error("Couldn't terminate instance %s. Here's why: %s: %s", instance_id, err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return response['Activity']

    def set_desired_capacity(self, group_name, capacity):
        if False:
            i = 10
            return i + 15
        '\n        Sets the desired capacity of the group. Amazon EC2 Auto Scaling tries to keep the\n        number of running instances equal to the desired capacity.\n\n        :param group_name: The name of the group to update.\n        :param capacity: The desired number of running instances.\n        '
        try:
            self.autoscaling_client.set_desired_capacity(AutoScalingGroupName=group_name, DesiredCapacity=capacity, HonorCooldown=False)
        except ClientError as err:
            logger.error("Couldn't set desired capacity %s. Here's why: %s: %s", group_name, err.response['Error']['Code'], err.response['Error']['Message'])
            raise

    def describe_instances(self, instance_ids):
        if False:
            while True:
                i = 10
        '\n        Gets information about instances.\n\n        :param instance_ids: A list of instance IDs to look up.\n        :return: Information about instances, or an empty list if none are found.\n        '
        try:
            response = self.autoscaling_client.describe_auto_scaling_instances(InstanceIds=instance_ids)
        except ClientError as err:
            logger.error("Couldn't describe instances %s. Here's why: %s: %s", instance_ids, err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return response['AutoScalingInstances']

    def describe_scaling_activities(self, group_name):
        if False:
            print('Hello World!')
        '\n        Gets information about scaling activities for the group. Scaling activities\n        are things like instances stopping or starting in response to user requests\n        or capacity changes.\n\n        :param group_name: The name of the group to look up.\n        :return: The list of scaling activities for the group, ordered with the most\n                 recent activity first.\n        '
        try:
            response = self.autoscaling_client.describe_scaling_activities(AutoScalingGroupName=group_name)
        except ClientError as err:
            logger.error("Couldn't describe scaling activities %s. Here's why: %s: %s", group_name, err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return response['Activities']

    def enable_metrics(self, group_name, metrics):
        if False:
            return 10
        '\n        Enables CloudWatch metric collection for Amazon EC2 Auto Scaling activities.\n\n        :param group_name: The name of the group to enable.\n        :param metrics: A list of metrics to collect.\n        '
        try:
            self.autoscaling_client.enable_metrics_collection(AutoScalingGroupName=group_name, Metrics=metrics, Granularity='1Minute')
        except ClientError as err:
            logger.error("Couldn't enable metrics on %s. Here's why: %s: %s", group_name, err.response['Error']['Code'], err.response['Error']['Message'])
            raise

    def disable_metrics(self, group_name):
        if False:
            for i in range(10):
                print('nop')
        '\n        Stops CloudWatch metric collection for the Auto Scaling group.\n\n        :param group_name: The name of the group.\n        '
        try:
            self.autoscaling_client.disable_metrics_collection(AutoScalingGroupName=group_name)
        except ClientError as err:
            logger.error("Couldn't disable metrics %s. Here's why: %s: %s", group_name, err.response['Error']['Code'], err.response['Error']['Message'])
            raise