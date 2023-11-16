import base64
import json
import logging
import time
from os import remove, chmod
import boto3
from botocore.exceptions import ClientError
log = logging.getLogger(__name__)

class AutoScalerError(Exception):
    pass

class AutoScaler:
    """
    Encapsulates Amazon EC2 Auto Scaling and EC2 management actions.
    """

    def __init__(self, resource_prefix, inst_type, ami_param, autoscaling_client, ec2_client, ssm_client, iam_client):
        if False:
            print('Hello World!')
        '\n        :param resource_prefix: The prefix for naming AWS resources that are created by this class.\n        :param inst_type: The type of EC2 instance to create, such as t3.micro.\n        :param ami_param: The Systems Manager parameter used to look up the AMI that is\n                          created.\n        :param autoscaling_client: A Boto3 EC2 Auto Scaling client.\n        :param ec2_client: A Boto3 EC2 client.\n        :param ssm_client: A Boto3 Systems Manager client.\n        :param iam_client: A Boto3 IAM client.\n        '
        self.inst_type = inst_type
        self.ami_param = ami_param
        self.autoscaling_client = autoscaling_client
        self.ec2_client = ec2_client
        self.ssm_client = ssm_client
        self.iam_client = iam_client
        self.launch_template_name = f'{resource_prefix}-template'
        self.group_name = f'{resource_prefix}-group'
        self.instance_policy_name = f'{resource_prefix}-pol'
        self.instance_role_name = f'{resource_prefix}-role'
        self.instance_profile_name = f'{resource_prefix}-prof'
        self.bad_creds_policy_name = f'{resource_prefix}-bc-pol'
        self.bad_creds_role_name = f'{resource_prefix}-bc-role'
        self.bad_creds_profile_name = f'{resource_prefix}-bc-prof'
        self.key_pair_name = f'{resource_prefix}-key-pair'

    @classmethod
    def from_client(cls, resource_prefix):
        if False:
            while True:
                i = 10
        '\n        Creates this class from Boto3 clients.\n\n        :param resource_prefix: The prefix for naming AWS resources that are created by this class.\n        '
        as_client = boto3.client('autoscaling')
        ec2_client = boto3.client('ec2')
        ssm_client = boto3.client('ssm')
        iam_client = boto3.client('iam')
        return cls(resource_prefix, 't3.micro', '/aws/service/ami-amazon-linux-latest/amzn2-ami-hvm-x86_64-gp2', as_client, ec2_client, ssm_client, iam_client)

    def create_instance_profile(self, policy_file, policy_name, role_name, profile_name, aws_managed_policies=()):
        if False:
            print('Hello World!')
        "\n        Creates a policy, role, and profile that is associated with instances created by\n        this class. An instance's associated profile defines a role that is assumed by the\n        instance. The role has attached policies that specify the AWS permissions granted to\n        clients that run on the instance.\n\n        :param policy_file: The name of a JSON file that contains the policy definition to\n                            create and attach to the role.\n        :param policy_name: The name to give the created policy.\n        :param role_name: The name to give the created role.\n        :param profile_name: The name to the created profile.\n        :param aws_managed_policies: Additional AWS-managed policies that are attached to\n                                     the role, such as AmazonSSMManagedInstanceCore to grant\n                                     use of Systems Manager to send commands to the instance.\n        :return: The ARN of the profile that is created.\n        "
        assume_role_doc = {'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow', 'Principal': {'Service': 'ec2.amazonaws.com'}, 'Action': 'sts:AssumeRole'}]}
        with open(policy_file) as file:
            instance_policy_doc = file.read()
        policy_arn = None
        try:
            pol_response = self.iam_client.create_policy(PolicyName=policy_name, PolicyDocument=instance_policy_doc)
            policy_arn = pol_response['Policy']['Arn']
            log.info('Created policy with ARN %s.', policy_arn)
        except ClientError as err:
            if err.response['Error']['Code'] == 'EntityAlreadyExists':
                log.info('Policy %s already exists, nothing to do.', policy_name)
                list_pol_response = self.iam_client.list_policies(Scope='Local')
                for pol in list_pol_response['Policies']:
                    if pol['PolicyName'] == policy_name:
                        policy_arn = pol['Arn']
                        break
            if policy_arn is None:
                raise AutoScalerError(f"Couldn't create policy {policy_name}: {err}")
        try:
            self.iam_client.create_role(RoleName=role_name, AssumeRolePolicyDocument=json.dumps(assume_role_doc))
            self.iam_client.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
            for aws_policy in aws_managed_policies:
                self.iam_client.attach_role_policy(RoleName=role_name, PolicyArn=f'arn:aws:iam::aws:policy/{aws_policy}')
            log.info('Created role %s and attached policy %s.', role_name, policy_arn)
        except ClientError as err:
            if err.response['Error']['Code'] == 'EntityAlreadyExists':
                log.info('Role %s already exists, nothing to do.', role_name)
            else:
                raise AutoScalerError(f"Couldn't create role {role_name}: {err}")
        try:
            profile_response = self.iam_client.create_instance_profile(InstanceProfileName=profile_name)
            waiter = self.iam_client.get_waiter('instance_profile_exists')
            waiter.wait(InstanceProfileName=profile_name)
            time.sleep(10)
            profile_arn = profile_response['InstanceProfile']['Arn']
            self.iam_client.add_role_to_instance_profile(InstanceProfileName=profile_name, RoleName=role_name)
            log.info('Created profile %s and added role %s.', profile_name, role_name)
        except ClientError as err:
            if err.response['Error']['Code'] == 'EntityAlreadyExists':
                prof_response = self.iam_client.get_instance_profile(InstanceProfileName=profile_name)
                profile_arn = prof_response['InstanceProfile']['Arn']
                log.info('Instance profile %s already exists, nothing to do.', profile_name)
            else:
                raise AutoScalerError(f"Couldn't create profile {profile_name} and attach it to role\n{role_name}: {err}")
        return profile_arn

    def get_instance_profile(self, instance_id):
        if False:
            i = 10
            return i + 15
        '\n        Gets data about the profile associated with an instance.\n\n        :param instance_id: The ID of the instance to look up.\n        :return: The profile data.\n        '
        try:
            response = self.ec2_client.describe_iam_instance_profile_associations(Filters=[{'Name': 'instance-id', 'Values': [instance_id]}])
        except ClientError as err:
            raise AutoScalerError(f"Couldn't get instance profile association for instance {instance_id}: {err}")
        else:
            return response['IamInstanceProfileAssociations'][0]

    def replace_instance_profile(self, instance_id, new_instance_profile_name, profile_association_id):
        if False:
            return 10
        '\n        Replaces the profile associated with a running instance. After the profile is\n        replaced, the instance is rebooted to ensure that it uses the new profile. When\n        the instance is ready, Systems Manager is used to restart the Python web server.\n\n        :param instance_id: The ID of the instance to update.\n        :param new_instance_profile_name: The name of the new profile to associate with\n                                          the specified instance.\n        :param profile_association_id: The ID of the existing profile association for the\n                                       instance.\n        '
        try:
            self.ec2_client.replace_iam_instance_profile_association(IamInstanceProfile={'Name': new_instance_profile_name}, AssociationId=profile_association_id)
            log.info('Replaced instance profile for association %s with profile %s.', profile_association_id, new_instance_profile_name)
            time.sleep(5)
            inst_ready = False
            tries = 0
            while not inst_ready:
                if tries % 6 == 0:
                    self.ec2_client.reboot_instances(InstanceIds=[instance_id])
                    log.info('Rebooting instance %s and waiting for it to to be ready.', instance_id)
                tries += 1
                time.sleep(10)
                response = self.ssm_client.describe_instance_information()
                for info in response['InstanceInformationList']:
                    if info['InstanceId'] == instance_id:
                        inst_ready = True
            self.ssm_client.send_command(InstanceIds=[instance_id], DocumentName='AWS-RunShellScript', Parameters={'commands': ['cd / && sudo python3 server.py 80']})
            log.info('Restarted the Python web server on instance %s.', instance_id)
        except ClientError as err:
            raise AutoScalerError(f"Couldn't replace instance profile for association {profile_association_id}: {err}")

    def delete_instance_profile(self, profile_name, role_name):
        if False:
            for i in range(10):
                print('nop')
        '\n        Detaches a role from an instance profile, detaches policies from the role,\n        and deletes all the resources.\n\n        :param profile_name: The name of the profile to delete.\n        :param role_name: The name of the role to delete.\n        '
        try:
            self.iam_client.remove_role_from_instance_profile(InstanceProfileName=profile_name, RoleName=role_name)
            self.iam_client.delete_instance_profile(InstanceProfileName=profile_name)
            log.info('Deleted instance profile %s.', profile_name)
            attached_policies = self.iam_client.list_attached_role_policies(RoleName=role_name)
            for pol in attached_policies['AttachedPolicies']:
                self.iam_client.detach_role_policy(RoleName=role_name, PolicyArn=pol['PolicyArn'])
                if not pol['PolicyArn'].startswith('arn:aws:iam::aws'):
                    self.iam_client.delete_policy(PolicyArn=pol['PolicyArn'])
                log.info('Detached and deleted policy %s.', pol['PolicyName'])
            self.iam_client.delete_role(RoleName=role_name)
            log.info('Deleted role %s.', role_name)
        except ClientError as err:
            if err.response['Error']['Code'] == 'NoSuchEntity':
                log.info("Instance profile %s doesn't exist, nothing to do.", profile_name)
            else:
                raise AutoScalerError(f"Couldn't delete instance profile {profile_name} or detach policies and delete role {role_name}: {err}")

    def create_key_pair(self, key_pair_name):
        if False:
            return 10
        '\n        Creates a new key pair.\n\n        :param key_pair_name: The name of the key pair to create.\n        :return: The newly created key pair.\n        '
        try:
            response = self.ec2_client.create_key_pair(KeyName=key_pair_name)
            with open(f'{key_pair_name}.pem', 'w') as file:
                file.write(response['KeyMaterial'])
            chmod(f'{key_pair_name}.pem', 384)
            log.info('Created key pair %s.', key_pair_name)
        except ClientError as err:
            raise AutoScalerError(f"Couldn't create key pair {key_pair_name}: {err}")

    def delete_key_pair(self):
        if False:
            print('Hello World!')
        '\n        Deletes a key pair.\n\n        :param key_pair_name: The name of the key pair to delete.\n        '
        try:
            self.ec2_client.delete_key_pair(KeyName=self.key_pair_name)
            remove(f'{self.key_pair_name}.pem')
            log.info('Deleted key pair %s.', self.key_pair_name)
        except ClientError as err:
            raise AutoScalerError(f"Couldn't delete key pair {self.key_pair_name}: {err}")
        except FileNotFoundError:
            log.info("Key pair %s doesn't exist, nothing to do.", self.key_pair_name)
        except PermissionError:
            log.info('Inadequate permissions to delete key pair %s.', self.key_pair_name)
        except Exception as err:
            raise AutoScalerError(f"Couldn't delete key pair {self.key_pair_name}: {err}")

    def create_template(self, server_startup_script_file, instance_policy_file):
        if False:
            while True:
                i = 10
        '\n        Creates an Amazon EC2 launch template to use with Amazon EC2 Auto Scaling. The\n        launch template specifies a Bash script in its user data field that runs after\n        the instance is started. This script installs Python packages and starts a\n        Python web server on the instance.\n\n        :param server_startup_script_file: The path to a Bash script file that is run\n                                           when an instance starts.\n        :param instance_policy_file: The path to a file that defines a permissions policy\n                                     to create and attach to the instance profile.\n        :return: Information about the newly created template.\n        '
        template = {}
        try:
            self.create_key_pair(self.key_pair_name)
            self.create_instance_profile(instance_policy_file, self.instance_policy_name, self.instance_role_name, self.instance_profile_name)
            with open(server_startup_script_file) as file:
                start_server_script = file.read()
            ami_latest = self.ssm_client.get_parameter(Name=self.ami_param)
            ami_id = ami_latest['Parameter']['Value']
            lt_response = self.ec2_client.create_launch_template(LaunchTemplateName=self.launch_template_name, LaunchTemplateData={'InstanceType': self.inst_type, 'ImageId': ami_id, 'IamInstanceProfile': {'Name': self.instance_profile_name}, 'UserData': base64.b64encode(start_server_script.encode(encoding='utf-8')).decode(encoding='utf-8'), 'KeyName': self.key_pair_name})
            template = lt_response['LaunchTemplate']
            log.info('Created launch template %s for AMI %s on %s.', self.launch_template_name, ami_id, self.inst_type)
        except ClientError as err:
            if err.response['Error']['Code'] == 'InvalidLaunchTemplateName.AlreadyExistsException':
                log.info('Launch template %s already exists, nothing to do.', self.launch_template_name)
            else:
                raise AutoScalerError(f"Couldn't create launch template {self.launch_template_name}: {err}.")
        return template

    def delete_template(self):
        if False:
            print('Hello World!')
        '\n        Deletes a launch template.\n        '
        try:
            self.ec2_client.delete_launch_template(LaunchTemplateName=self.launch_template_name)
            self.delete_instance_profile(self.instance_profile_name, self.instance_role_name)
            log.info('Launch template %s deleted.', self.launch_template_name)
        except ClientError as err:
            if err.response['Error']['Code'] == 'InvalidLaunchTemplateName.NotFoundException':
                log.info('Launch template %s does not exist, nothing to do.', self.launch_template_name)
            else:
                raise AutoScalerError(f"Couldn't delete launch template {self.launch_template_name}: {err}.")

    def get_availability_zones(self):
        if False:
            print('Hello World!')
        '\n        Gets a list of Availability Zones in the AWS Region of the Amazon EC2 client.\n\n        :return: The list of Availability Zones for the client Region.\n        '
        try:
            response = self.ec2_client.describe_availability_zones()
            zones = [zone['ZoneName'] for zone in response['AvailabilityZones']]
        except ClientError as err:
            raise AutoScalerError(f"Couldn't get availability zones: {err}.")
        else:
            return zones

    def create_group(self, group_size):
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates an EC2 Auto Scaling group with the specified size.\n\n        :param group_size: The number of instances to set for the minimum and maximum in\n                           the group.\n        :return: The list of Availability Zones specified for the group.\n        '
        zones = []
        try:
            zones = self.get_availability_zones()
            self.autoscaling_client.create_auto_scaling_group(AutoScalingGroupName=self.group_name, AvailabilityZones=zones, LaunchTemplate={'LaunchTemplateName': self.launch_template_name, 'Version': '$Default'}, MinSize=group_size, MaxSize=group_size)
            log.info('Created EC2 Auto Scaling group %s with availability zones %s.', self.launch_template_name, zones)
        except ClientError as err:
            if err.response['Error']['Code'] == 'AlreadyExists':
                log.info('EC2 Auto Scaling group %s already exists, nothing to do.', self.group_name)
            else:
                raise AutoScalerError(f"Couldn't create EC2 Auto Scaling group {self.group_name}: {err}")
        return zones

    def get_instances(self):
        if False:
            while True:
                i = 10
        '\n        Gets data about the instances in the EC2 Auto Scaling group.\n\n        :return: Data about the instances.\n        '
        try:
            as_response = self.autoscaling_client.describe_auto_scaling_groups(AutoScalingGroupNames=[self.group_name])
            instance_ids = [i['InstanceId'] for i in as_response['AutoScalingGroups'][0]['Instances']]
        except ClientError as err:
            raise AutoScalerError(f"Couldn't get instances for Auto Scaling group {self.group_name}: {err}")
        else:
            return instance_ids

    def terminate_instance(self, instance_id):
        if False:
            print('Hello World!')
        '\n        Terminates and instances in an EC2 Auto Scaling group. After an instance is\n        terminated, it can no longer be accessed.\n\n        :param instance_id: The ID of the instance to terminate.\n        '
        try:
            self.autoscaling_client.terminate_instance_in_auto_scaling_group(InstanceId=instance_id, ShouldDecrementDesiredCapacity=False)
            log.info('Terminated instance %s.', instance_id)
        except ClientError as err:
            raise AutoScalerError(f"Couldn't terminate instance {instance_id}: {err}")

    def attach_load_balancer_target_group(self, lb_target_group):
        if False:
            while True:
                i = 10
        '\n        Attaches an Elastic Load Balancing (ELB) target group to this EC2 Auto Scaling group.\n        The target group specifies how the load balancer forward requests to the instances\n        in the group.\n\n        :param lb_target_group: Data about the ELB target group to attach.\n        '
        try:
            self.autoscaling_client.attach_load_balancer_target_groups(AutoScalingGroupName=self.group_name, TargetGroupARNs=[lb_target_group['TargetGroupArn']])
            log.info('Attached load balancer target group %s to auto scaling group %s.', lb_target_group['TargetGroupName'], self.group_name)
        except ClientError as err:
            raise AutoScalerError(f"Couldn't attach load balancer target group {lb_target_group['TargetGroupName']}\nto auto scaling group {self.group_name}")

    def _try_terminate_instance(self, inst_id):
        if False:
            for i in range(10):
                print('nop')
        stopping = False
        log.info(f'Stopping {inst_id}.')
        while not stopping:
            try:
                self.autoscaling_client.terminate_instance_in_auto_scaling_group(InstanceId=inst_id, ShouldDecrementDesiredCapacity=True)
                stopping = True
            except ClientError as err:
                if err.response['Error']['Code'] == 'ScalingActivityInProgress':
                    log.info('Scaling activity in progress for %s. Waiting...', inst_id)
                    time.sleep(10)
                else:
                    raise AutoScalerError(f"Couldn't stop instance {inst_id}: {err}.")

    def _try_delete_group(self):
        if False:
            i = 10
            return i + 15
        '\n        Tries to delete the EC2 Auto Scaling group. If the group is in use or in progress,\n        the function waits and retries until the group is successfully deleted.\n        '
        stopped = False
        while not stopped:
            try:
                self.autoscaling_client.delete_auto_scaling_group(AutoScalingGroupName=self.group_name)
                stopped = True
                log.info('Deleted EC2 Auto Scaling group %s.', self.group_name)
            except ClientError as err:
                if err.response['Error']['Code'] == 'ResourceInUse' or err.response['Error']['Code'] == 'ScalingActivityInProgress':
                    log.info('Some instances are still running. Waiting for them to stop...')
                    time.sleep(10)
                else:
                    raise AutoScalerError(f"Couldn't delete group {self.group_name}: {err}.")

    def delete_group(self):
        if False:
            print('Hello World!')
        '\n        Terminates all instances in the group, deletes the EC2 Auto Scaling group.\n        '
        try:
            response = self.autoscaling_client.describe_auto_scaling_groups(AutoScalingGroupNames=[self.group_name])
            groups = response.get('AutoScalingGroups', [])
            if len(groups) > 0:
                self.autoscaling_client.update_auto_scaling_group(AutoScalingGroupName=self.group_name, MinSize=0)
                instance_ids = [inst['InstanceId'] for inst in groups[0]['Instances']]
                for inst_id in instance_ids:
                    self._try_terminate_instance(inst_id)
                self._try_delete_group()
            else:
                log.info('No groups found named %s, nothing to do.', self.group_name)
        except ClientError as err:
            raise AutoScalerError(f"Couldn't delete group {self.group_name}: {err}.")

    def get_default_vpc(self):
        if False:
            i = 10
            return i + 15
        '\n        Gets the default VPC for the account.\n\n        :return: Data about the default VPC.\n        '
        try:
            response = self.ec2_client.describe_vpcs(Filters=[{'Name': 'is-default', 'Values': ['true']}])
        except ClientError as err:
            raise AutoScalerError(f"Couldn't get default VPC: {err}")
        else:
            return response['Vpcs'][0]

    def verify_inbound_port(self, vpc, port, ip_address):
        if False:
            for i in range(10):
                print('nop')
        "\n        Verify the default security group of the specified VPC allows ingress from this\n        computer. This can be done by allowing ingress from this computer's IP\n        address. In some situations, such as connecting from a corporate network, you\n        must instead specify a prefix list ID. You can also temporarily open the port to\n        any IP address while running this example. If you do, be sure to remove public\n        access when you're done.\n\n        :param vpc: The VPC used by this example.\n        :param port: The port to verify.\n        :param ip_address: This computer's IP address.\n        :return: The default security group of the specific VPC, and a value that indicates\n                 whether the specified port is open.\n        "
        try:
            response = self.ec2_client.describe_security_groups(Filters=[{'Name': 'group-name', 'Values': ['default']}, {'Name': 'vpc-id', 'Values': [vpc['VpcId']]}])
            sec_group = response['SecurityGroups'][0]
            port_is_open = False
            log.info('Found default security group %s.', sec_group['GroupId'])
            for ip_perm in sec_group['IpPermissions']:
                if ip_perm.get('FromPort', 0) == port:
                    log.info('Found inbound rule: %s', ip_perm)
                    for ip_range in ip_perm['IpRanges']:
                        cidr = ip_range.get('CidrIp', '')
                        if cidr.startswith(ip_address) or cidr == '0.0.0.0/0':
                            port_is_open = True
                    if ip_perm['PrefixListIds']:
                        port_is_open = True
                    if not port_is_open:
                        log.info("The inbound rule does not appear to be open to either this computer's IP\naddress of %s, to all IP addresses (0.0.0.0/0), or to a prefix list ID.", ip_address)
                    else:
                        break
        except ClientError as err:
            raise AutoScalerError(f"Couldn't verify inbound rule for port {port} for VPC {vpc['VpcId']}: {err}")
        else:
            return (sec_group, port_is_open)

    def open_inbound_port(self, sec_group_id, port, ip_address):
        if False:
            return 10
        '\n        Add an ingress rule to the specified security group that allows access on the\n        specified port from the specified IP address.\n\n        :param sec_group_id: The ID of the security group to modify.\n        :param port: The port to open.\n        :param ip_address: The IP address that is granted access.\n        '
        try:
            self.ec2_client.authorize_security_group_ingress(GroupId=sec_group_id, CidrIp=f'{ip_address}/32', FromPort=port, ToPort=port, IpProtocol='tcp')
            log.info('Authorized ingress to %s on port %s from %s.', sec_group_id, port, ip_address)
        except ClientError as err:
            raise AutoScalerError(f"Couldn't authorize ingress to {sec_group_id} on port {port} from {ip_address}: {err}")

    def get_subnets(self, vpc_id, zones):
        if False:
            for i in range(10):
                print('nop')
        '\n        Gets the default subnets in a VPC for a specified list of Availability Zones.\n\n        :param vpc_id: The ID of the VPC to look up.\n        :param zones: The list of Availability Zones to look up.\n        :return: The list of subnets found.\n        '
        try:
            response = self.ec2_client.describe_subnets(Filters=[{'Name': 'vpc-id', 'Values': [vpc_id]}, {'Name': 'availability-zone', 'Values': zones}, {'Name': 'default-for-az', 'Values': ['true']}])
            subnets = response['Subnets']
            log.info('Found %s subnets for the specified zones.', len(subnets))
        except ClientError as err:
            raise AutoScalerError(f"Couldn't get subnets: {err}")
        else:
            return subnets