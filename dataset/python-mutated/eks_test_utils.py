from __future__ import annotations
import re
from copy import deepcopy
from typing import TYPE_CHECKING, Pattern, Type, Union
from ..utils.eks_test_constants import STATUS, ClusterAttributes, ClusterInputs, FargateProfileAttributes, FargateProfileInputs, NodegroupAttributes, NodegroupInputs, ResponseAttributes
if TYPE_CHECKING:
    import datetime
    from airflow.providers.amazon.aws.hooks.eks import EksHook
InputTypes = Union[Type[ClusterInputs], Type[NodegroupInputs], Type[FargateProfileInputs]]

def attributes_to_test(inputs: InputTypes, cluster_name: str, fargate_profile_name: str | None=None, nodegroup_name: str | None=None) -> list[tuple]:
    if False:
        return 10
    '\n    Assembles the list of tuples which will be used to validate test results.\n    The format of the tuple is (attribute name, expected value)\n\n    :param inputs: A class containing lists of tuples to use for verifying the output\n    of cluster or nodegroup creation tests.\n    :param cluster_name: The name of the cluster under test.\n    :param fargate_profile_name: The name of the Fargate profile under test if applicable.\n    :param nodegroup_name: The name of the nodegroup under test if applicable.\n    :return: Returns a list of tuples containing the keys and values to be validated in testing.\n    '
    result: list[tuple] = deepcopy(inputs.REQUIRED + inputs.OPTIONAL + [STATUS])
    if inputs == ClusterInputs:
        result += [(ClusterAttributes.NAME, cluster_name)]
    elif inputs == FargateProfileInputs:
        result += [(FargateProfileAttributes.FARGATE_PROFILE_NAME, fargate_profile_name)]
    elif inputs == NodegroupInputs:
        required_tag: dict = {f'kubernetes.io/cluster/{cluster_name}': 'owned'}
        final_tag_set: dict = required_tag
        for (key, value) in result:
            if key == 'tags':
                final_tag_set = {**value, **final_tag_set}
        result = [(key, value) if key != NodegroupAttributes.TAGS else (NodegroupAttributes.TAGS, final_tag_set) for (key, value) in result]
        result += [(NodegroupAttributes.NODEGROUP_NAME, nodegroup_name)]
    return result

def generate_clusters(eks_hook: EksHook, num_clusters: int, minimal: bool) -> list[str]:
    if False:
        i = 10
        return i + 15
    '\n    Generates a number of EKS Clusters with data and adds them to the mocked backend.\n\n    :param eks_hook: An EksHook object used to call the EKS API.\n    :param num_clusters: Number of clusters to generate.\n    :param minimal: If True, only the required values are generated; if False all values are generated.\n    :return: Returns a list of the names of the generated clusters.\n    '
    return [eks_hook.create_cluster(name=f'cluster{count}', **_input_builder(ClusterInputs, minimal))[ResponseAttributes.CLUSTER][ClusterAttributes.NAME] for count in range(num_clusters)]

def generate_fargate_profiles(eks_hook: EksHook, cluster_name: str, num_profiles: int, minimal: bool) -> list[str]:
    if False:
        i = 10
        return i + 15
    '\n    Generates a number of EKS Fargate profiles with data and adds them to the mocked backend.\n\n    :param eks_hook: An EksHook object used to call the EKS API.\n    :param cluster_name: The name of the EKS Cluster to attach the nodegroups to.\n    :param num_profiles: Number of Fargate profiles to generate.\n    :param minimal: If True, only the required values are generated; if False all values are generated.\n    :return: Returns a list of the names of the generated nodegroups.\n    '
    return [eks_hook.create_fargate_profile(fargateProfileName=f'profile{count}', clusterName=cluster_name, **_input_builder(FargateProfileInputs, minimal))[ResponseAttributes.FARGATE_PROFILE][FargateProfileAttributes.FARGATE_PROFILE_NAME] for count in range(num_profiles)]

def generate_nodegroups(eks_hook: EksHook, cluster_name: str, num_nodegroups: int, minimal: bool) -> list[str]:
    if False:
        while True:
            i = 10
    '\n    Generates a number of EKS Managed Nodegroups with data and adds them to the mocked backend.\n\n    :param eks_hook: An EksHook object used to call the EKS API.\n    :param cluster_name: The name of the EKS Cluster to attach the nodegroups to.\n    :param num_nodegroups: Number of clusters to generate.\n    :param minimal: If True, only the required values are generated; if False all values are generated.\n    :return: Returns a list of the names of the generated nodegroups.\n    '
    return [eks_hook.create_nodegroup(nodegroupName=f'nodegroup{count}', clusterName=cluster_name, **_input_builder(NodegroupInputs, minimal))[ResponseAttributes.NODEGROUP][NodegroupAttributes.NODEGROUP_NAME] for count in range(num_nodegroups)]

def region_matches_partition(region: str, partition: str) -> bool:
    if False:
        i = 10
        return i + 15
    '\n    Returns True if the provided region and partition are a valid pair.\n\n    :param region: AWS region code to test.\n    :param partition: AWS partition code to test.\n    :return: Returns True if the provided region and partition are a valid pair.\n    '
    valid_matches: list[tuple[str, str]] = [('cn-', 'aws-cn'), ('us-gov-', 'aws-us-gov'), ('us-gov-iso-', 'aws-iso'), ('us-gov-iso-b-', 'aws-iso-b')]
    for (prefix, expected_partition) in valid_matches:
        if region.startswith(prefix):
            return partition == expected_partition
    return partition == 'aws'

def _input_builder(options: InputTypes, minimal: bool) -> dict:
    if False:
        return 10
    '\n    Assembles the inputs which will be used to generate test object into a dictionary.\n\n    :param options: A class containing lists of tuples to use for to create\n    the cluster or nodegroup used in testing.\n    :param minimal: If True, only the required values are generated; if False all values are generated.\n    :return: Returns a dict containing the keys and values to be validated in testing.\n    '
    values: list[tuple] = deepcopy(options.REQUIRED)
    if not minimal:
        values.extend(deepcopy(options.OPTIONAL))
    return dict(values)

def string_to_regex(value: str) -> Pattern[str]:
    if False:
        print('Hello World!')
    '\n    Converts a string template into a regex template for pattern matching.\n\n    :param value: The template string to convert.\n    :returns: Returns a regex pattern\n    '
    return re.compile(re.sub('[{](.*?)[}]', '(?P<\\1>.+)', value))

def convert_keys(original: dict) -> dict:
    if False:
        for i in range(10):
            print('nop')
    '\n    API Input and Output keys are formatted differently.  The EKS Hooks map\n    as closely as possible to the API calls, which use camelCase variable\n    names, but the Operators match python conventions and use snake_case.\n    This method converts the keys of a dict which are in snake_case (input\n    format) to camelCase (output format) while leaving the dict values unchanged.\n\n    :param original: Dict which needs the keys converted.\n    :value original: Dict\n    '
    if 'nodegroup_name' in original.keys():
        conversion_map = {'cluster_name': 'clusterName', 'cluster_role_arn': 'roleArn', 'nodegroup_subnets': 'subnets', 'subnets': 'subnets', 'nodegroup_name': 'nodegroupName', 'nodegroup_role_arn': 'nodeRole'}
    elif 'fargate_profile_name' in original.keys():
        conversion_map = {'cluster_name': 'clusterName', 'fargate_profile_name': 'fargateProfileName', 'subnets': 'subnets', 'pod_execution_role_arn': 'podExecutionRoleArn', 'fargate_pod_execution_role_arn': 'podExecutionRoleArn', 'selectors': 'selectors', 'fargate_selectors': 'selectors'}
    else:
        conversion_map = {'cluster_name': 'name', 'cluster_role_arn': 'roleArn', 'resources_vpc_config': 'resourcesVpcConfig'}
    return {conversion_map[k] if k in conversion_map else k: v for (k, v) in deepcopy(original).items()}

def iso_date(input_datetime: datetime.datetime) -> str:
    if False:
        return 10
    return f'{input_datetime:%Y-%m-%dT%H:%M:%S}Z'

def generate_dict(prefix, count) -> dict:
    if False:
        while True:
            i = 10
    return {f'{prefix}_{_count}': str(_count) for _count in range(count)}