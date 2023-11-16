import copy
from unittest.mock import Mock, patch
import pytest
from click.exceptions import ClickException
import ray.tests.aws.utils.helpers as helpers
import ray.tests.aws.utils.stubs as stubs
from ray.autoscaler._private.aws.config import DEFAULT_AMI, _configure_subnet, _get_subnets_or_die, bootstrap_aws, log_to_cli
from ray.autoscaler._private.aws.node_provider import AWSNodeProvider
from ray.autoscaler._private.providers import _get_node_provider
from ray.tests.aws.utils.constants import AUX_SG, AUX_SUBNET, CUSTOM_IN_BOUND_RULES, DEFAULT_CLUSTER_NAME, DEFAULT_INSTANCE_PROFILE, DEFAULT_KEY_PAIR, DEFAULT_LT, DEFAULT_SG, DEFAULT_SG_AUX_SUBNET, DEFAULT_SG_DUAL_GROUP_RULES, DEFAULT_SG_WITH_NAME, DEFAULT_SG_WITH_NAME_AND_RULES, DEFAULT_SG_WITH_RULES, DEFAULT_SG_WITH_RULES_AUX_SUBNET, DEFAULT_SUBNET

def test_use_subnets_in_only_one_vpc(iam_client_stub, ec2_client_stub):
    if False:
        while True:
            i = 10
    '\n    This test validates that when bootstrap_aws populates the SubnetIds field,\n    all of the subnets used belong to the same VPC, and that a SecurityGroup\n    in that VPC is correctly configured.\n\n    Also validates that head IAM role is correctly filled.\n    '
    stubs.configure_iam_role_default(iam_client_stub)
    stubs.configure_key_pair_default(ec2_client_stub)
    stubs.describe_a_thousand_subnets_in_different_vpcs(ec2_client_stub)
    stubs.describe_subnets_echo(ec2_client_stub, [DEFAULT_SUBNET])
    stubs.describe_no_security_groups(ec2_client_stub)
    stubs.create_sg_echo(ec2_client_stub, DEFAULT_SG)
    stubs.describe_sgs_on_vpc(ec2_client_stub, [DEFAULT_SUBNET['VpcId']], [DEFAULT_SG])
    stubs.authorize_sg_ingress(ec2_client_stub, DEFAULT_SG_WITH_RULES)
    stubs.describe_an_sg_2(ec2_client_stub, DEFAULT_SG_WITH_RULES)
    config = helpers.bootstrap_aws_example_config_file('example-full.yaml')
    _get_subnets_or_die.cache_clear()
    for node_type in config['available_node_types'].values():
        node_config = node_type['node_config']
        assert node_config['SubnetIds'] == [DEFAULT_SUBNET['SubnetId']]
        assert node_config['SecurityGroupIds'] == [DEFAULT_SG['GroupId']]

@pytest.mark.parametrize('correct_az', [True, False])
def test_create_sg_different_vpc_same_rules(iam_client_stub, ec2_client_stub, correct_az: bool):
    if False:
        return 10
    stubs.skip_to_configure_sg(ec2_client_stub, iam_client_stub)
    default_subnet = copy.deepcopy(DEFAULT_SUBNET)
    if not correct_az:
        default_subnet['AvailabilityZone'] = 'us-west-2b'
    stubs.describe_subnets_echo(ec2_client_stub, [default_subnet])
    stubs.describe_subnets_echo(ec2_client_stub, [AUX_SUBNET])
    stubs.describe_no_security_groups(ec2_client_stub)
    stubs.create_sg_echo(ec2_client_stub, DEFAULT_SG_AUX_SUBNET)
    stubs.describe_sgs_on_vpc(ec2_client_stub, [AUX_SUBNET['VpcId']], [DEFAULT_SG_AUX_SUBNET])
    stubs.create_sg_echo(ec2_client_stub, DEFAULT_SG)
    stubs.describe_sgs_on_vpc(ec2_client_stub, [DEFAULT_SUBNET['VpcId']], [DEFAULT_SG])
    stubs.authorize_sg_ingress(ec2_client_stub, DEFAULT_SG_DUAL_GROUP_RULES)
    stubs.authorize_sg_ingress(ec2_client_stub, DEFAULT_SG_WITH_RULES_AUX_SUBNET)
    error = None
    try:
        config = helpers.bootstrap_aws_example_config_file('example-subnets.yaml')
    except ClickException as e:
        error = e
    _get_subnets_or_die.cache_clear()
    if not correct_az:
        assert isinstance(error, ClickException), 'Did not get a ClickException!'
        iam_client_stub._queue.clear()
        ec2_client_stub._queue.clear()
        return
    for (node_type_key, node_type) in config['available_node_types'].items():
        node_config = node_type['node_config']
        security_group_ids = node_config['SecurityGroupIds']
        subnet_ids = node_config['SubnetIds']
        if node_type_key == config['head_node_type']:
            assert security_group_ids == [DEFAULT_SG['GroupId']]
            assert subnet_ids == [DEFAULT_SUBNET['SubnetId']]
        else:
            assert security_group_ids == [AUX_SG['GroupId']]
            assert subnet_ids == [AUX_SUBNET['SubnetId']]
    iam_client_stub.assert_no_pending_responses()
    ec2_client_stub.assert_no_pending_responses()

def test_create_sg_with_custom_inbound_rules_and_name(iam_client_stub, ec2_client_stub):
    if False:
        for i in range(10):
            print('nop')
    stubs.skip_to_configure_sg(ec2_client_stub, iam_client_stub)
    stubs.describe_subnets_echo(ec2_client_stub, [DEFAULT_SUBNET])
    stubs.describe_no_security_groups(ec2_client_stub)
    stubs.create_sg_echo(ec2_client_stub, DEFAULT_SG_WITH_NAME)
    stubs.describe_sgs_on_vpc(ec2_client_stub, [DEFAULT_SUBNET['VpcId']], [DEFAULT_SG_WITH_NAME])
    stubs.authorize_sg_ingress(ec2_client_stub, DEFAULT_SG_WITH_NAME_AND_RULES)
    stubs.describe_sg_echo(ec2_client_stub, DEFAULT_SG_WITH_NAME_AND_RULES)
    _get_subnets_or_die.cache_clear()
    config = helpers.bootstrap_aws_example_config_file('example-security-group.yaml')
    assert config['provider']['security_group']['GroupName'] == DEFAULT_SG_WITH_NAME_AND_RULES['GroupName']
    assert config['provider']['security_group']['IpPermissions'] == CUSTOM_IN_BOUND_RULES
    iam_client_stub.assert_no_pending_responses()
    ec2_client_stub.assert_no_pending_responses()

def test_subnet_given_head_and_worker_sg(iam_client_stub, ec2_client_stub):
    if False:
        i = 10
        return i + 15
    stubs.configure_iam_role_default(iam_client_stub)
    stubs.configure_key_pair_default(ec2_client_stub)
    stubs.describe_a_security_group(ec2_client_stub, DEFAULT_SG)
    stubs.describe_a_thousand_subnets_in_different_vpcs(ec2_client_stub)
    config = helpers.bootstrap_aws_example_config_file('example-head-and-worker-security-group.yaml')
    for node_type in config['available_node_types'].values():
        node_config = node_type['node_config']
        assert node_config['SubnetIds'] == [DEFAULT_SUBNET['SubnetId']]
    iam_client_stub.assert_no_pending_responses()
    ec2_client_stub.assert_no_pending_responses()

@pytest.mark.parametrize('iam_client_stub,ec2_client_stub,region', [3 * (region,) for region in DEFAULT_AMI], indirect=['iam_client_stub', 'ec2_client_stub'])
def test_fills_out_amis_and_iam(iam_client_stub, ec2_client_stub, region):
    if False:
        return 10
    region_key_pair = DEFAULT_KEY_PAIR.copy()
    region_key_pair['KeyName'] = DEFAULT_KEY_PAIR['KeyName'].replace('us-west-2', region)
    stubs.configure_iam_role_default(iam_client_stub)
    stubs.configure_key_pair_default(ec2_client_stub, region=region, expected_key_pair=region_key_pair)
    stubs.describe_a_security_group(ec2_client_stub, DEFAULT_SG)
    stubs.configure_subnet_default(ec2_client_stub)
    config = helpers.load_aws_example_config_file('example-full.yaml')
    head_node_config = config['available_node_types']['ray.head.default']['node_config']
    worker_node_config = config['available_node_types']['ray.worker.default']['node_config']
    del head_node_config['ImageId']
    del worker_node_config['ImageId']
    head_node_config['SecurityGroupIds'] = ['sg-1234abcd']
    worker_node_config['SecurityGroupIds'] = ['sg-1234abcd']
    config['provider']['region'] = region
    defaults_filled = bootstrap_aws(config)
    ami = DEFAULT_AMI.get(defaults_filled.get('provider', {}).get('region'))
    for node_type in defaults_filled['available_node_types'].values():
        node_config = node_type['node_config']
        assert node_config.get('ImageId') == ami
    assert defaults_filled['head_node']['IamInstanceProfile'] == {'Arn': DEFAULT_INSTANCE_PROFILE['Arn']}
    head_type = config['head_node_type']
    assert 'IamInstanceProfile' not in defaults_filled['available_node_types'][head_type]
    iam_client_stub.assert_no_pending_responses()
    ec2_client_stub.assert_no_pending_responses()

def test_iam_already_configured(iam_client_stub, ec2_client_stub):
    if False:
        return 10
    '\n    Checks that things work as expected when IAM role is supplied by user.\n    '
    stubs.configure_key_pair_default(ec2_client_stub)
    stubs.describe_a_security_group(ec2_client_stub, DEFAULT_SG)
    stubs.configure_subnet_default(ec2_client_stub)
    config = helpers.load_aws_example_config_file('example-full.yaml')
    head_node_config = config['available_node_types']['ray.head.default']['node_config']
    worker_node_config = config['available_node_types']['ray.worker.default']['node_config']
    head_node_config['IamInstanceProfile'] = 'mock_profile'
    head_node_config['SecurityGroupIds'] = ['sg-1234abcd']
    worker_node_config['SecurityGroupIds'] = ['sg-1234abcd']
    defaults_filled = bootstrap_aws(config)
    filled_head = defaults_filled['available_node_types']['ray.head.default']['node_config']
    assert filled_head['IamInstanceProfile'] == 'mock_profile'
    assert 'IamInstanceProfile' not in defaults_filled['head_node']
    iam_client_stub.assert_no_pending_responses()
    ec2_client_stub.assert_no_pending_responses()

def test_create_sg_multinode(iam_client_stub, ec2_client_stub):
    if False:
        i = 10
        return i + 15
    '\n    Test AWS Bootstrap logic when config being bootstrapped has the\n    following properties:\n\n    (1) auth config does not specify ssh key path\n    (2) available_node_types is provided\n    (3) security group name and ip permissions set in provider field\n    (4) Available node types have SubnetIds field set and this\n        field is of form SubnetIds: [subnet-xxxxx].\n        Both node types specify the same subnet-xxxxx.\n\n    Tests creation of a security group and key pair under these conditions.\n    '
    subnet_id = DEFAULT_SUBNET['SubnetId']
    provider_data = helpers.load_aws_example_config_file('example-security-group.yaml')['provider']
    base_config = helpers.load_aws_example_config_file('example-full.yaml')
    config = copy.deepcopy(base_config)
    config['provider'] = provider_data
    head_node_config = config['available_node_types']['ray.head.default']['node_config']
    worker_node_config = config['available_node_types']['ray.worker.default']['node_config']
    head_node_config['SubnetIds'] = [subnet_id]
    worker_node_config['SubnetIds'] = [subnet_id]
    stubs.configure_iam_role_default(iam_client_stub)
    stubs.configure_key_pair_default(ec2_client_stub)
    stubs.describe_a_thousand_subnets_in_different_vpcs(ec2_client_stub)
    stubs.describe_subnets_echo(ec2_client_stub, [DEFAULT_SUBNET])
    stubs.describe_no_security_groups(ec2_client_stub)
    stubs.create_sg_echo(ec2_client_stub, DEFAULT_SG_WITH_NAME)
    stubs.describe_sgs_on_vpc(ec2_client_stub, [DEFAULT_SUBNET['VpcId']], [DEFAULT_SG_WITH_NAME])
    stubs.authorize_sg_ingress(ec2_client_stub, DEFAULT_SG_WITH_NAME_AND_RULES)
    stubs.describe_sg_echo(ec2_client_stub, DEFAULT_SG_WITH_NAME_AND_RULES)
    _get_subnets_or_die.cache_clear()
    bootstrapped_config = helpers.bootstrap_aws_config(config)
    assert bootstrapped_config['provider']['security_group']['GroupName'] == DEFAULT_SG_WITH_NAME_AND_RULES['GroupName']
    assert bootstrapped_config['provider']['security_group']['IpPermissions'] == CUSTOM_IN_BOUND_RULES
    sg_id = DEFAULT_SG['GroupId']
    for node_type in bootstrapped_config['available_node_types'].values():
        node_config = node_type['node_config']
        assert node_config['SecurityGroupIds'] == [sg_id]
    for node_type in bootstrapped_config['available_node_types'].values():
        node_config = node_type['node_config']
        assert node_config['KeyName'] == DEFAULT_KEY_PAIR['KeyName']
    bootstrapped_head_type = bootstrapped_config['head_node_type']
    bootstrapped_types = bootstrapped_config['available_node_types']
    bootstrapped_head_config = bootstrapped_types[bootstrapped_head_type]['node_config']
    assert DEFAULT_SG['VpcId'] == DEFAULT_SUBNET['VpcId']
    assert DEFAULT_SUBNET['SubnetId'] == bootstrapped_head_config['SubnetIds'][0]
    assert 'ssh_private_key' in bootstrapped_config['auth']
    iam_client_stub.assert_no_pending_responses()
    ec2_client_stub.assert_no_pending_responses()

def test_missing_keyname(iam_client_stub, ec2_client_stub):
    if False:
        i = 10
        return i + 15
    config = helpers.load_aws_example_config_file('example-full.yaml')
    config['auth']['ssh_private_key'] = '/path/to/private/key'
    head_node_config = config['available_node_types']['ray.head.default']['node_config']
    worker_node_config = config['available_node_types']['ray.worker.default']['node_config']
    stubs.configure_iam_role_default(iam_client_stub)
    missing_user_data_config = copy.deepcopy(config)
    with pytest.raises(AssertionError):
        bootstrap_aws(missing_user_data_config)
    head_node_config['SecurityGroupIds'] = ['sg-1234abcd']
    worker_node_config['SecurityGroupIds'] = ['sg-1234abcd']
    head_node_config['UserData'] = {'someKey': 'someValue'}
    worker_node_config['UserData'] = {'someKey': 'someValue'}
    stubs.configure_iam_role_default(iam_client_stub)
    stubs.describe_a_security_group(ec2_client_stub, DEFAULT_SG)
    stubs.configure_subnet_default(ec2_client_stub)
    bootstrap_aws(config)
    iam_client_stub.assert_no_pending_responses()
    ec2_client_stub.assert_no_pending_responses()

def test_log_to_cli(iam_client_stub, ec2_client_stub):
    if False:
        while True:
            i = 10
    config = helpers.load_aws_example_config_file('example-full.yaml')
    head_node_config = config['available_node_types']['ray.head.default']['node_config']
    worker_node_config = config['available_node_types']['ray.worker.default']['node_config']
    head_node_config['SecurityGroupIds'] = ['sg-1234abcd']
    worker_node_config['SecurityGroupIds'] = ['sg-1234abcd']
    stubs.configure_iam_role_default(iam_client_stub)
    stubs.configure_key_pair_default(ec2_client_stub)
    stubs.describe_a_security_group(ec2_client_stub, DEFAULT_SG)
    stubs.configure_subnet_default(ec2_client_stub)
    config = helpers.bootstrap_aws_config(config)
    log_to_cli(config)
    iam_client_stub.assert_no_pending_responses()
    ec2_client_stub.assert_no_pending_responses()

def test_network_interfaces(ec2_client_stub, iam_client_stub, ec2_client_stub_fail_fast, ec2_client_stub_max_retries):
    if False:
        return 10
    stubs.configure_iam_role_default(iam_client_stub)
    stubs.configure_key_pair_default(ec2_client_stub)
    sgids = ['sg-00000000', 'sg-11111111', 'sg-22222222', 'sg-33333333']
    security_groups = []
    suffix = 0
    for sgid in sgids:
        sg = copy.deepcopy(DEFAULT_SG)
        sg['GroupName'] += f'-{suffix}'
        sg['GroupId'] = sgid
        security_groups.append(sg)
        suffix += 1
    stubs.describe_sgs_by_id(ec2_client_stub, sgids, security_groups)
    stubs.configure_subnet_default(ec2_client_stub)
    stubs.describe_subnets_echo(ec2_client_stub, [DEFAULT_SUBNET, {**DEFAULT_SUBNET, 'SubnetId': 'subnet-11111111'}])
    stubs.describe_subnets_echo(ec2_client_stub, [{**DEFAULT_SUBNET, 'SubnetId': 'subnet-22222222'}])
    stubs.describe_subnets_echo(ec2_client_stub, [{**DEFAULT_SUBNET, 'SubnetId': 'subnet-33333333'}])
    config = helpers.bootstrap_aws_example_config_file('example-network-interfaces.yaml')
    new_provider = _get_node_provider(config['provider'], DEFAULT_CLUSTER_NAME, False)
    for (name, node_type) in config['available_node_types'].items():
        node_cfg = node_type['node_config']
        tags = helpers.node_provider_tags(config, name)
        stubs.describe_instances_with_any_filter_consumer(ec2_client_stub_max_retries)
        stubs.run_instances_with_network_interfaces_consumer(ec2_client_stub_fail_fast, node_cfg['NetworkInterfaces'])
        new_provider.create_node(node_cfg, tags, 1)
    iam_client_stub.assert_no_pending_responses()
    ec2_client_stub.assert_no_pending_responses()
    ec2_client_stub_fail_fast.assert_no_pending_responses()
    ec2_client_stub_max_retries.assert_no_pending_responses()

def test_network_interface_conflict_keys():
    if False:
        return 10
    conflict_kv_pairs = [('SubnetId', 'subnet-0000000'), ('SubnetIds', ['subnet-0000000', 'subnet-1111111']), ('SecurityGroupIds', ['sg-1234abcd', 'sg-dcba4321'])]
    expected_error_msg = 'If NetworkInterfaces are defined, subnets and security groups must ONLY be given in each NetworkInterface.'
    for conflict_kv_pair in conflict_kv_pairs:
        config = helpers.load_aws_example_config_file('example-network-interfaces.yaml')
        head_name = config['head_node_type']
        head_node_cfg = config['available_node_types'][head_name]['node_config']
        head_node_cfg[conflict_kv_pair[0]] = conflict_kv_pair[1]
        with pytest.raises(ValueError, match=expected_error_msg):
            helpers.bootstrap_aws_config(config)

def test_network_interface_missing_subnet():
    if False:
        print('Hello World!')
    expected_error_msg = 'NetworkInterfaces are defined but at least one is missing a subnet. Please ensure all interfaces have a subnet assigned.'
    config = helpers.load_aws_example_config_file('example-network-interfaces.yaml')
    for (name, node_type) in config['available_node_types'].items():
        node_cfg = node_type['node_config']
        for network_interface_cfg in node_cfg['NetworkInterfaces']:
            network_interface_cfg.pop('SubnetId')
            with pytest.raises(ValueError, match=expected_error_msg):
                helpers.bootstrap_aws_config(config)

def test_network_interface_missing_security_group():
    if False:
        while True:
            i = 10
    expected_error_msg = 'NetworkInterfaces are defined but at least one is missing a security group. Please ensure all interfaces have a security group assigned.'
    config = helpers.load_aws_example_config_file('example-network-interfaces.yaml')
    for (name, node_type) in config['available_node_types'].items():
        node_cfg = node_type['node_config']
        for network_interface_cfg in node_cfg['NetworkInterfaces']:
            network_interface_cfg.pop('Groups')
            with pytest.raises(ValueError, match=expected_error_msg):
                helpers.bootstrap_aws_config(config)

def test_launch_templates(ec2_client_stub, ec2_client_stub_fail_fast, ec2_client_stub_max_retries):
    if False:
        while True:
            i = 10
    stubs.describe_launch_template_versions_by_id_default(ec2_client_stub, ['$Latest'])
    stubs.describe_launch_template_versions_by_name_default(ec2_client_stub, ['2'])
    stubs.configure_key_pair_default(ec2_client_stub)
    sgids = [DEFAULT_SG['GroupId']]
    security_groups = [DEFAULT_SG]
    stubs.describe_sgs_by_id(ec2_client_stub, sgids, security_groups)
    stubs.configure_subnet_default(ec2_client_stub)
    config = helpers.bootstrap_aws_example_config_file('example-launch-templates.yaml')
    new_provider = _get_node_provider(config['provider'], DEFAULT_CLUSTER_NAME, False)
    max_count = 1
    for (name, node_type) in config['available_node_types'].items():
        stubs.describe_instances_with_any_filter_consumer(ec2_client_stub_max_retries)
        node_cfg = node_type['node_config']
        stubs.run_instances_with_launch_template_consumer(ec2_client_stub_fail_fast, config, node_cfg, name, DEFAULT_LT['LaunchTemplateData'], max_count)
        tags = helpers.node_provider_tags(config, name)
        new_provider.create_node(node_cfg, tags, max_count)
    ec2_client_stub.assert_no_pending_responses()
    ec2_client_stub_fail_fast.assert_no_pending_responses()
    ec2_client_stub_max_retries.assert_no_pending_responses()

@pytest.mark.parametrize('num_on_demand_nodes', [0, 1001, 9999])
@pytest.mark.parametrize('num_spot_nodes', [0, 1001, 9999])
@pytest.mark.parametrize('stop', [True, False])
def test_terminate_nodes(num_on_demand_nodes, num_spot_nodes, stop):
    if False:
        return 10
    on_demand_nodes = {'i-{:017d}'.format(i) for i in range(num_on_demand_nodes)}
    spot_nodes = {'i-{:017d}'.format(i + num_on_demand_nodes) for i in range(num_spot_nodes)}
    node_ids = list(on_demand_nodes.union(spot_nodes))
    with patch('ray.autoscaler._private.aws.node_provider.make_ec2_resource'):
        provider = AWSNodeProvider(provider_config={'region': 'nowhere', 'cache_stopped_nodes': stop}, cluster_name='default')

    def mock_get_cached_node(node_id):
        if False:
            return 10
        result = Mock()
        result.spot_instance_request_id = 'sir-08b93456' if node_id in spot_nodes else ''
        return result
    provider._get_cached_node = mock_get_cached_node
    provider.terminate_nodes(node_ids)
    stop_calls = provider.ec2.meta.client.stop_instances.call_args_list
    terminate_calls = provider.ec2.meta.client.terminate_instances.call_args_list
    nodes_to_stop = set()
    nodes_to_terminate = spot_nodes
    if stop:
        nodes_to_stop.update(on_demand_nodes)
    else:
        nodes_to_terminate.update(on_demand_nodes)
    for (calls, nodes_to_include_in_call) in ((stop_calls, nodes_to_stop), (terminate_calls, nodes_to_terminate)):
        nodes_included_in_call = set()
        for call in calls:
            assert len(call[1]['InstanceIds']) <= provider.max_terminate_nodes
            nodes_included_in_call.update(call[1]['InstanceIds'])
        assert nodes_to_include_in_call == nodes_included_in_call

def test_use_subnets_ordered_by_az(ec2_client_stub):
    if False:
        return 10
    '\n    This test validates that when bootstrap_aws populates the SubnetIds field,\n    the subnets are ordered the same way as availability zones.\n\n    '
    stubs.describe_twenty_subnets_in_different_azs(ec2_client_stub)
    base_config = helpers.load_aws_example_config_file('example-full.yaml')
    base_config['provider']['availability_zone'] = 'us-west-2c,us-west-2d,us-west-2a'
    config = _configure_subnet(base_config)
    for node_type in config['available_node_types'].values():
        node_config = node_type['node_config']
        assert len(node_config['SubnetIds']) == 15
        offsets = [int(s.split('-')[1]) % 4 for s in node_config['SubnetIds']]
        assert set(offsets[:5]) == {2}, 'First 5 should be in us-west-2c'
        assert set(offsets[5:10]) == {3}, 'Next 5 should be in us-west-2d'
        assert set(offsets[10:15]) == {0}, 'Last 5 should be in us-west-2a'

def test_cloudwatch_dashboard_creation(cloudwatch_client_stub, ssm_client_stub):
    if False:
        while True:
            i = 10
    node_id = 'i-abc'
    cloudwatch_helper = helpers.get_cloudwatch_helper(node_id)
    stubs.put_cluster_dashboard_success(cloudwatch_client_stub, cloudwatch_helper)
    cloudwatch_helper._put_cloudwatch_dashboard()
    cloudwatch_client_stub.assert_no_pending_responses()

def test_cloudwatch_alarm_creation(cloudwatch_client_stub, ssm_client_stub):
    if False:
        for i in range(10):
            print('nop')
    node_id = 'i-abc'
    cloudwatch_helper = helpers.get_cloudwatch_helper(node_id)
    cw_ssm_param_name = helpers.get_ssm_param_name(cloudwatch_helper.cluster_name, 'alarm')
    stubs.get_param_ssm_same(ssm_client_stub, cw_ssm_param_name, cloudwatch_helper, 'alarm')
    stubs.put_cluster_alarms_success(cloudwatch_client_stub, cloudwatch_helper)
    cloudwatch_helper._put_cloudwatch_alarm()
    cloudwatch_client_stub.assert_no_pending_responses()

def test_cloudwatch_agent_update_without_change_head_node(ssm_client_stub, ec2_client_stub):
    if False:
        i = 10
        return i + 15
    node_id = 'i-abc'
    is_head_node = True
    cloudwatch_helper = helpers.get_cloudwatch_helper(node_id)
    stubs.get_ec2_cwa_installed_tag_true(ec2_client_stub, node_id)
    cw_ssm_param_name = helpers.get_ssm_param_name(cloudwatch_helper.cluster_name, 'agent')
    stubs.get_param_ssm_same(ssm_client_stub, cw_ssm_param_name, cloudwatch_helper, 'agent')
    cloudwatch_helper._update_cloudwatch_config('agent', is_head_node)

def test_cloudwatch_agent_update_with_change_head_node(ec2_client_stub, ssm_client_stub):
    if False:
        for i in range(10):
            print('nop')
    node_id = 'i-abc'
    is_head_node = True
    cloudwatch_helper = helpers.get_cloudwatch_helper(node_id)
    stubs.get_ec2_cwa_installed_tag_true(ec2_client_stub, node_id)
    cw_ssm_param_name = helpers.get_ssm_param_name(cloudwatch_helper.cluster_name, 'agent')
    stubs.get_param_ssm_different(ssm_client_stub, cw_ssm_param_name)
    cmd_id = stubs.put_parameter_cloudwatch_config(ssm_client_stub, cloudwatch_helper.cluster_name, 'agent')
    stubs.update_hash_tag_success(ec2_client_stub, node_id, 'agent', cloudwatch_helper)
    cmd_id = stubs.send_command_stop_cwa(ssm_client_stub, node_id)
    stubs.list_command_invocations_success(ssm_client_stub, node_id, cmd_id)
    cmd_id = stubs.send_command_start_cwa(ssm_client_stub, node_id, cw_ssm_param_name)
    stubs.list_command_invocations_success(ssm_client_stub, node_id, cmd_id)
    cloudwatch_helper._update_cloudwatch_config('agent', is_head_node)
    ec2_client_stub.assert_no_pending_responses()
    ssm_client_stub.assert_no_pending_responses()

def test_cloudwatch_agent_update_with_change_worker_node(ec2_client_stub, ssm_client_stub):
    if False:
        i = 10
        return i + 15
    node_id = 'i-abc'
    is_head_node = False
    cloudwatch_helper = helpers.get_cloudwatch_helper(node_id)
    stubs.get_ec2_cwa_installed_tag_true(ec2_client_stub, node_id)
    stubs.get_head_node_config_hash_different(ec2_client_stub, 'agent', cloudwatch_helper, node_id)
    stubs.get_cur_node_config_hash_different(ec2_client_stub, 'agent', node_id)
    stubs.update_hash_tag_success(ec2_client_stub, node_id, 'agent', cloudwatch_helper)
    cmd_id = stubs.send_command_stop_cwa(ssm_client_stub, node_id)
    stubs.list_command_invocations_success(ssm_client_stub, node_id, cmd_id)
    cw_ssm_param_name = helpers.get_ssm_param_name(cloudwatch_helper.cluster_name, 'agent')
    cmd_id = stubs.send_command_start_cwa(ssm_client_stub, node_id, cw_ssm_param_name)
    stubs.list_command_invocations_success(ssm_client_stub, node_id, cmd_id)
    cloudwatch_helper._update_cloudwatch_config('agent', is_head_node)
    ec2_client_stub.assert_no_pending_responses()
    ssm_client_stub.assert_no_pending_responses()

def test_cloudwatch_dashboard_update_head_node(ec2_client_stub, ssm_client_stub, cloudwatch_client_stub):
    if False:
        print('Hello World!')
    node_id = 'i-abc'
    is_head_node = True
    cloudwatch_helper = helpers.get_cloudwatch_helper(node_id)
    stubs.get_ec2_cwa_installed_tag_true(ec2_client_stub, node_id)
    cw_ssm_param_name = helpers.get_ssm_param_name(cloudwatch_helper.cluster_name, 'dashboard')
    stubs.get_param_ssm_different(ssm_client_stub, cw_ssm_param_name)
    stubs.put_parameter_cloudwatch_config(ssm_client_stub, cloudwatch_helper.cluster_name, 'dashboard')
    stubs.update_hash_tag_success(ec2_client_stub, node_id, 'dashboard', cloudwatch_helper)
    stubs.put_cluster_dashboard_success(cloudwatch_client_stub, cloudwatch_helper)
    cloudwatch_helper._update_cloudwatch_config('dashboard', is_head_node)
    ec2_client_stub.assert_no_pending_responses()
    ssm_client_stub.assert_no_pending_responses()

def test_cloudwatch_dashboard_update_worker_node(ec2_client_stub, ssm_client_stub, cloudwatch_client_stub):
    if False:
        while True:
            i = 10
    node_id = 'i-abc'
    is_head_node = False
    cloudwatch_helper = helpers.get_cloudwatch_helper(node_id)
    stubs.get_ec2_cwa_installed_tag_true(ec2_client_stub, node_id)
    stubs.get_head_node_config_hash_different(ec2_client_stub, 'dashboard', cloudwatch_helper, node_id)
    stubs.get_cur_node_config_hash_different(ec2_client_stub, 'dashboard', node_id)
    stubs.update_hash_tag_success(ec2_client_stub, node_id, 'dashboard', cloudwatch_helper)
    cloudwatch_helper._update_cloudwatch_config('dashboard', is_head_node)
    ec2_client_stub.assert_no_pending_responses()
    ssm_client_stub.assert_no_pending_responses()

def test_cloudwatch_alarm_update_head_node(ec2_client_stub, ssm_client_stub, cloudwatch_client_stub):
    if False:
        return 10
    node_id = 'i-abc'
    is_head_node = True
    cloudwatch_helper = helpers.get_cloudwatch_helper(node_id)
    stubs.get_ec2_cwa_installed_tag_true(ec2_client_stub, node_id)
    cw_ssm_param_name = helpers.get_ssm_param_name(cloudwatch_helper.cluster_name, 'alarm')
    stubs.get_param_ssm_different(ssm_client_stub, cw_ssm_param_name)
    stubs.put_parameter_cloudwatch_config(ssm_client_stub, cloudwatch_helper.cluster_name, 'alarm')
    stubs.update_hash_tag_success(ec2_client_stub, node_id, 'alarm', cloudwatch_helper)
    stubs.get_param_ssm_same(ssm_client_stub, cw_ssm_param_name, cloudwatch_helper, 'alarm')
    stubs.put_cluster_alarms_success(cloudwatch_client_stub, cloudwatch_helper)
    cloudwatch_helper._update_cloudwatch_config('alarm', is_head_node)
    ec2_client_stub.assert_no_pending_responses()
    ssm_client_stub.assert_no_pending_responses()

def test_cloudwatch_alarm_update_worker_node(ec2_client_stub, ssm_client_stub, cloudwatch_client_stub):
    if False:
        for i in range(10):
            print('nop')
    node_id = 'i-abc'
    is_head_node = False
    cloudwatch_helper = helpers.get_cloudwatch_helper(node_id)
    stubs.get_ec2_cwa_installed_tag_true(ec2_client_stub, node_id)
    cw_ssm_param_name = helpers.get_ssm_param_name(cloudwatch_helper.cluster_name, 'alarm')
    stubs.get_head_node_config_hash_different(ec2_client_stub, 'alarm', cloudwatch_helper, node_id)
    stubs.get_cur_node_config_hash_different(ec2_client_stub, 'alarm', node_id)
    stubs.update_hash_tag_success(ec2_client_stub, node_id, 'alarm', cloudwatch_helper)
    stubs.get_param_ssm_same(ssm_client_stub, cw_ssm_param_name, cloudwatch_helper, 'alarm')
    stubs.put_cluster_alarms_success(cloudwatch_client_stub, cloudwatch_helper)
    cloudwatch_helper._update_cloudwatch_config('alarm', is_head_node)
    ec2_client_stub.assert_no_pending_responses()
    ssm_client_stub.assert_no_pending_responses()
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', __file__]))