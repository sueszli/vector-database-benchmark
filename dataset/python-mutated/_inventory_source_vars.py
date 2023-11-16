import json
import re
import logging
from django.utils.translation import gettext_lazy as _
from django.utils.encoding import iri_to_uri
FrozenInjectors = dict()
logger = logging.getLogger('awx.main.migrations')

class PluginFileInjector(object):
    plugin_name = None
    namespace = None
    collection = None

    def inventory_as_dict(self, inventory_source, private_data_dir):
        if False:
            return 10
        'Default implementation of inventory plugin file contents.\n        There are some valid cases when all parameters can be obtained from\n        the environment variables, example "plugin: linode" is valid\n        ideally, however, some options should be filled from the inventory source data\n        '
        if self.plugin_name is None:
            raise NotImplementedError('At minimum the plugin name is needed for inventory plugin use.')
        proper_name = f'{self.namespace}.{self.collection}.{self.plugin_name}'
        return {'plugin': proper_name}

class azure_rm(PluginFileInjector):
    plugin_name = 'azure_rm'
    namespace = 'azure'
    collection = 'azcollection'

    def inventory_as_dict(self, inventory_source, private_data_dir):
        if False:
            while True:
                i = 10
        ret = super(azure_rm, self).inventory_as_dict(inventory_source, private_data_dir)
        source_vars = inventory_source.source_vars_dict
        ret['fail_on_template_errors'] = False
        group_by_hostvar = {'location': {'prefix': '', 'separator': '', 'key': 'location'}, 'tag': {'prefix': '', 'separator': '', 'key': 'tags.keys() | list if tags else []'}, 'security_group': {'prefix': '', 'separator': '', 'key': 'security_group'}, 'resource_group': {'prefix': '', 'separator': '', 'key': 'resource_group'}, 'os_family': {'prefix': '', 'separator': '', 'key': 'os_disk.operating_system_type'}}
        group_by = [grouping_name for grouping_name in group_by_hostvar if source_vars.get('group_by_{}'.format(grouping_name), True)]
        ret['keyed_groups'] = [group_by_hostvar[grouping_name] for grouping_name in group_by]
        if 'tag' in group_by:
            ret['keyed_groups'].append({'prefix': '', 'separator': '', 'key': 'dict(tags.keys() | map("regex_replace", "^(.*)$", "\\1_") | list | zip(tags.values() | list)) if tags else []'})
        ret['use_contrib_script_compatible_sanitization'] = True
        ret['plain_host_names'] = True
        ret['default_host_filters'] = []
        user_filters = []
        old_filterables = [('resource_groups', 'resource_group'), ('tags', 'tags')]
        for (key, loc) in old_filterables:
            value = source_vars.get(key, None)
            if value and isinstance(value, str):
                if key == 'tags':
                    for kvpair in value.split(','):
                        kv = kvpair.split(':')
                        user_filters.append('"{}" not in tags.keys()'.format(kv[0].strip()))
                        if len(kv) > 1:
                            user_filters.append('tags["{}"] != "{}"'.format(kv[0].strip(), kv[1].strip()))
                else:
                    user_filters.append('{} not in {}'.format(loc, value.split(',')))
        if user_filters:
            ret.setdefault('exclude_host_filters', [])
            ret['exclude_host_filters'].extend(user_filters)
        ret['conditional_groups'] = {'azure': True}
        ret['hostvar_expressions'] = {'provisioning_state': 'provisioning_state | title', 'computer_name': 'name', 'type': 'resource_type', 'private_ip': 'private_ipv4_addresses[0] if private_ipv4_addresses else None', 'public_ip': 'public_ipv4_addresses[0] if public_ipv4_addresses else None', 'public_ip_name': 'public_ip_name if public_ip_name is defined else None', 'public_ip_id': 'public_ip_id if public_ip_id is defined else None', 'tags': 'tags if tags else None'}
        if source_vars.get('use_private_ip', False):
            ret['hostvar_expressions']['ansible_host'] = 'private_ipv4_addresses[0]'
        if inventory_source.source_regions and 'all' not in inventory_source.source_regions:
            ret.setdefault('exclude_host_filters', [])
            python_regions = [x.strip() for x in inventory_source.source_regions.split(',')]
            ret['exclude_host_filters'].append('location not in {}'.format(repr(python_regions)))
        return ret

class ec2(PluginFileInjector):
    plugin_name = 'aws_ec2'
    namespace = 'amazon'
    collection = 'aws'

    def _get_ec2_group_by_choices(self):
        if False:
            for i in range(10):
                print('nop')
        return [('ami_id', _('Image ID')), ('availability_zone', _('Availability Zone')), ('aws_account', _('Account')), ('instance_id', _('Instance ID')), ('instance_state', _('Instance State')), ('platform', _('Platform')), ('instance_type', _('Instance Type')), ('key_pair', _('Key Name')), ('region', _('Region')), ('security_group', _('Security Group')), ('tag_keys', _('Tags')), ('tag_none', _('Tag None')), ('vpc_id', _('VPC ID'))]

    def _compat_compose_vars(self):
        if False:
            for i in range(10):
                print('nop')
        return {'ec2_block_devices': "dict(block_device_mappings | map(attribute='device_name') | list | zip(block_device_mappings | map(attribute='ebs.volume_id') | list))", 'ec2_dns_name': 'public_dns_name', 'ec2_group_name': 'placement.group_name', 'ec2_instance_profile': 'iam_instance_profile | default("")', 'ec2_ip_address': 'public_ip_address', 'ec2_kernel': 'kernel_id | default("")', 'ec2_monitored': "monitoring.state in ['enabled', 'pending']", 'ec2_monitoring_state': 'monitoring.state', 'ec2_placement': 'placement.availability_zone', 'ec2_ramdisk': 'ramdisk_id | default("")', 'ec2_reason': 'state_transition_reason', 'ec2_security_group_ids': "security_groups | map(attribute='group_id') | list |  join(',')", 'ec2_security_group_names': "security_groups | map(attribute='group_name') | list |  join(',')", 'ec2_tag_Name': 'tags.Name', 'ec2_state': 'state.name', 'ec2_state_code': 'state.code', 'ec2_state_reason': 'state_reason.message if state_reason is defined else ""', 'ec2_sourceDestCheck': 'source_dest_check | default(false) | lower | string', 'ec2_account_id': 'owner_id', 'ec2_ami_launch_index': 'ami_launch_index | string', 'ec2_architecture': 'architecture', 'ec2_client_token': 'client_token', 'ec2_ebs_optimized': 'ebs_optimized', 'ec2_hypervisor': 'hypervisor', 'ec2_image_id': 'image_id', 'ec2_instance_type': 'instance_type', 'ec2_key_name': 'key_name', 'ec2_launch_time': 'launch_time | regex_replace(" ", "T") | regex_replace("(\\+)(\\d\\d):(\\d)(\\d)$", ".\\g<2>\\g<3>Z")', 'ec2_platform': 'platform | default("")', 'ec2_private_dns_name': 'private_dns_name', 'ec2_private_ip_address': 'private_ip_address', 'ec2_public_dns_name': 'public_dns_name', 'ec2_region': 'placement.region', 'ec2_root_device_name': 'root_device_name', 'ec2_root_device_type': 'root_device_type', 'ec2_spot_instance_request_id': 'spot_instance_request_id | default("")', 'ec2_subnet_id': 'subnet_id | default("")', 'ec2_virtualization_type': 'virtualization_type', 'ec2_vpc_id': 'vpc_id | default("")', 'ansible_host': 'public_ip_address', 'ec2_eventsSet': 'events | default("")', 'ec2_persistent': 'persistent | default(false)', 'ec2_requester_id': 'requester_id | default("")'}

    def inventory_as_dict(self, inventory_source, private_data_dir):
        if False:
            return 10
        ret = super(ec2, self).inventory_as_dict(inventory_source, private_data_dir)
        keyed_groups = []
        group_by_hostvar = {'ami_id': {'prefix': '', 'separator': '', 'key': 'image_id', 'parent_group': 'images'}, 'availability_zone': {'prefix': '', 'separator': '', 'key': 'placement.availability_zone', 'parent_group': 'zones'}, 'aws_account': {'prefix': '', 'separator': '', 'key': 'ec2_account_id', 'parent_group': 'accounts'}, 'instance_id': {'prefix': '', 'separator': '', 'key': 'instance_id', 'parent_group': 'instances'}, 'instance_state': {'prefix': 'instance_state', 'key': 'ec2_state', 'parent_group': 'instance_states'}, 'platform': {'prefix': 'platform', 'key': 'platform | default("undefined")', 'parent_group': 'platforms'}, 'instance_type': {'prefix': 'type', 'key': 'instance_type', 'parent_group': 'types'}, 'key_pair': {'prefix': 'key', 'key': 'key_name', 'parent_group': 'keys'}, 'region': {'prefix': '', 'separator': '', 'key': 'placement.region', 'parent_group': 'regions'}, 'security_group': {'prefix': 'security_group', 'key': 'security_groups | map(attribute="group_name")', 'parent_group': 'security_groups'}, 'tag_keys': [{'prefix': 'tag', 'key': 'tags', 'parent_group': 'tags'}, {'prefix': 'tag', 'key': 'tags.keys()', 'parent_group': 'tags'}], 'vpc_id': {'prefix': 'vpc_id', 'key': 'vpc_id', 'parent_group': 'vpcs'}}
        group_by = [x.strip().lower() for x in inventory_source.group_by.split(',') if x.strip()]
        for choice in self._get_ec2_group_by_choices():
            value = bool(group_by and choice[0] in group_by or (not group_by and choice[0] != 'instance_id'))
            if value:
                this_keyed_group = group_by_hostvar.get(choice[0], None)
                if this_keyed_group is not None:
                    if isinstance(this_keyed_group, list):
                        keyed_groups.extend(this_keyed_group)
                    else:
                        keyed_groups.append(this_keyed_group)
        if not group_by or ('region' in group_by and 'availability_zone' in group_by):
            keyed_groups.append({'prefix': '', 'separator': '', 'key': 'placement.availability_zone', 'parent_group': '{{ placement.region }}'})
        source_vars = inventory_source.source_vars_dict
        replace_dash = bool(source_vars.get('replace_dash_in_groups', True))
        legacy_regex = {True: '[^A-Za-z0-9\\_]', False: '[^A-Za-z0-9\\_\\-]'}[replace_dash]
        list_replacer = 'map("regex_replace", "{rx}", "_") | list'.format(rx=legacy_regex)
        ret['use_contrib_script_compatible_sanitization'] = True
        for grouping_data in keyed_groups:
            if grouping_data['key'] in ('placement.region', 'placement.availability_zone'):
                continue
            if grouping_data['key'] == 'tags':
                grouping_data['key'] = 'dict(tags.keys() | {replacer} | zip(tags.values() | {replacer}))'.format(replacer=list_replacer)
            elif grouping_data['key'] == 'tags.keys()' or grouping_data['prefix'] == 'security_group':
                grouping_data['key'] += ' | {replacer}'.format(replacer=list_replacer)
            else:
                grouping_data['key'] += ' | regex_replace("{rx}", "_")'.format(rx=legacy_regex)
        if source_vars.get('iam_role_arn', None):
            ret['iam_role_arn'] = source_vars['iam_role_arn']
        if source_vars.get('boto_profile', None):
            ret['boto_profile'] = source_vars['boto_profile']
        elif not replace_dash:
            ret['use_contrib_script_compatible_sanitization'] = True
        if source_vars.get('nested_groups') is False:
            for this_keyed_group in keyed_groups:
                this_keyed_group.pop('parent_group', None)
        if keyed_groups:
            ret['keyed_groups'] = keyed_groups
        compose_dict = {'ec2_id': 'instance_id'}
        inst_filters = {}
        compose_dict.update(self._compat_compose_vars())
        ret['groups'] = {'ec2': True}
        if source_vars.get('hostname_variable') is not None:
            hnames = []
            for expr in source_vars.get('hostname_variable').split(','):
                if expr == 'public_dns_name':
                    hnames.append('dns-name')
                elif not expr.startswith('tag:') and '_' in expr:
                    hnames.append(expr.replace('_', '-'))
                else:
                    hnames.append(expr)
            ret['hostnames'] = hnames
        else:
            ret['hostnames'] = ['network-interface.addresses.association.public-ip', 'dns-name', 'private-dns-name']
        inst_filters['instance-state-name'] = ['running']
        if source_vars.get('destination_variable') or source_vars.get('vpc_destination_variable'):
            for fd in ('destination_variable', 'vpc_destination_variable'):
                if source_vars.get(fd):
                    compose_dict['ansible_host'] = source_vars.get(fd)
                    break
        if compose_dict:
            ret['compose'] = compose_dict
        if inventory_source.instance_filters:
            filter_sets = [f for f in inventory_source.instance_filters.split(',') if f]
            for instance_filter in filter_sets:
                instance_filter = instance_filter.strip()
                if not instance_filter or '=' not in instance_filter:
                    continue
                (filter_key, filter_value) = [x.strip() for x in instance_filter.split('=', 1)]
                if not filter_key:
                    continue
                inst_filters[filter_key] = filter_value
        if inst_filters:
            ret['filters'] = inst_filters
        if inventory_source.source_regions and 'all' not in inventory_source.source_regions:
            ret['regions'] = inventory_source.source_regions.split(',')
        return ret

class gce(PluginFileInjector):
    plugin_name = 'gcp_compute'
    namespace = 'google'
    collection = 'cloud'

    def _compat_compose_vars(self):
        if False:
            i = 10
            return i + 15
        return {'gce_description': 'description if description else None', 'gce_machine_type': 'machineType', 'gce_name': 'name', 'gce_network': 'networkInterfaces[0].network.name', 'gce_private_ip': 'networkInterfaces[0].networkIP', 'gce_public_ip': 'networkInterfaces[0].accessConfigs[0].natIP | default(None)', 'gce_status': 'status', 'gce_subnetwork': 'networkInterfaces[0].subnetwork.name', 'gce_tags': 'tags.get("items", [])', 'gce_zone': 'zone', 'gce_metadata': 'metadata.get("items", []) | items2dict(key_name="key", value_name="value")', 'gce_image': 'image', 'ansible_ssh_host': 'networkInterfaces[0].accessConfigs[0].natIP | default(networkInterfaces[0].networkIP)'}

    def inventory_as_dict(self, inventory_source, private_data_dir):
        if False:
            return 10
        ret = super(gce, self).inventory_as_dict(inventory_source, private_data_dir)
        ret['auth_kind'] = 'serviceaccount'
        filters = []
        keyed_groups = [{'prefix': 'network', 'key': 'gce_subnetwork'}, {'prefix': '', 'separator': '', 'key': 'gce_private_ip'}, {'prefix': '', 'separator': '', 'key': 'gce_public_ip'}, {'prefix': '', 'separator': '', 'key': 'machineType'}, {'prefix': '', 'separator': '', 'key': 'zone'}, {'prefix': 'tag', 'key': 'gce_tags'}, {'prefix': 'status', 'key': 'status | lower'}, {'prefix': '', 'separator': '', 'key': 'image'}]
        compose_dict = {'gce_id': 'id'}
        ret['use_contrib_script_compatible_sanitization'] = True
        ret['retrieve_image_info'] = True
        compose_dict.update(self._compat_compose_vars())
        ret['hostnames'] = ['name', 'public_ip', 'private_ip']
        if keyed_groups:
            ret['keyed_groups'] = keyed_groups
        if filters:
            ret['filters'] = filters
        if compose_dict:
            ret['compose'] = compose_dict
        if inventory_source.source_regions and 'all' not in inventory_source.source_regions:
            ret['zones'] = inventory_source.source_regions.split(',')
        return ret

class vmware(PluginFileInjector):
    plugin_name = 'vmware_vm_inventory'
    namespace = 'community'
    collection = 'vmware'

    def inventory_as_dict(self, inventory_source, private_data_dir):
        if False:
            i = 10
            return i + 15
        ret = super(vmware, self).inventory_as_dict(inventory_source, private_data_dir)
        ret['strict'] = False
        UPPERCASE_PROPS = ['availableField', 'configIssue', 'configStatus', 'customValue', 'datastore', 'effectiveRole', 'guestHeartbeatStatus', 'layout', 'layoutEx', 'name', 'network', 'overallStatus', 'parentVApp', 'permission', 'recentTask', 'resourcePool', 'rootSnapshot', 'snapshot', 'triggeredAlarmState', 'value']
        NESTED_PROPS = ['capability', 'config', 'guest', 'runtime', 'storage', 'summary']
        ret['properties'] = UPPERCASE_PROPS + NESTED_PROPS
        ret['compose'] = {'ansible_host': 'guest.ipAddress'}
        ret['compose']['ansible_ssh_host'] = ret['compose']['ansible_host']
        ret['compose']['ansible_uuid'] = '99999999 | random | to_uuid'
        for prop in UPPERCASE_PROPS:
            if prop == prop.lower():
                continue
            ret['compose'][prop.lower()] = prop
        ret['with_nested_properties'] = True
        vmware_opts = dict(inventory_source.source_vars_dict.items())
        if inventory_source.instance_filters:
            vmware_opts.setdefault('host_filters', inventory_source.instance_filters)
        if inventory_source.group_by:
            vmware_opts.setdefault('groupby_patterns', inventory_source.group_by)
        alias_pattern = vmware_opts.get('alias_pattern')
        if alias_pattern:
            ret.setdefault('hostnames', [])
            for alias in alias_pattern.split(','):
                striped_alias = alias.replace('{', '').replace('}', '').strip()
                if not striped_alias:
                    continue
                ret['hostnames'].append(striped_alias)
        host_pattern = vmware_opts.get('host_pattern')
        if host_pattern:
            stripped_hp = host_pattern.replace('{', '').replace('}', '').strip()
            ret['compose']['ansible_host'] = stripped_hp
            ret['compose']['ansible_ssh_host'] = stripped_hp
        host_filters = vmware_opts.get('host_filters')
        if host_filters:
            ret.setdefault('filters', [])
            for hf in host_filters.split(','):
                striped_hf = hf.replace('{', '').replace('}', '').strip()
                if not striped_hf:
                    continue
                ret['filters'].append(striped_hf)
        else:
            ret['filters'] = ['runtime.powerState == "poweredOn"']
        groupby_patterns = vmware_opts.get('groupby_patterns')
        ret.setdefault('keyed_groups', [])
        if groupby_patterns:
            for pattern in groupby_patterns.split(','):
                stripped_pattern = pattern.replace('{', '').replace('}', '').strip()
                ret['keyed_groups'].append({'prefix': '', 'separator': '', 'key': stripped_pattern})
        else:
            for entry in ('config.guestId', '"templates" if config.template else "guests"'):
                ret['keyed_groups'].append({'prefix': '', 'separator': '', 'key': entry})
        return ret

class openstack(PluginFileInjector):
    plugin_name = 'openstack'
    namespace = 'openstack'
    collection = 'cloud'

    def inventory_as_dict(self, inventory_source, private_data_dir):
        if False:
            i = 10
            return i + 15

        def use_host_name_for_name(a_bool_maybe):
            if False:
                print('Hello World!')
            if not isinstance(a_bool_maybe, bool):
                return a_bool_maybe
            elif a_bool_maybe:
                return 'name'
            else:
                return 'uuid'
        ret = super(openstack, self).inventory_as_dict(inventory_source, private_data_dir)
        ret['fail_on_errors'] = True
        ret['expand_hostvars'] = True
        ret['inventory_hostname'] = use_host_name_for_name(False)
        source_vars = inventory_source.source_vars_dict
        for var_name in ['expand_hostvars', 'fail_on_errors']:
            if var_name in source_vars:
                ret[var_name] = source_vars[var_name]
        if 'use_hostnames' in source_vars:
            ret['inventory_hostname'] = use_host_name_for_name(source_vars['use_hostnames'])
        return ret

class rhv(PluginFileInjector):
    """ovirt uses the custom credential templating, and that is all"""
    plugin_name = 'ovirt'
    initial_version = '2.9'
    namespace = 'ovirt'
    collection = 'ovirt'

    def inventory_as_dict(self, inventory_source, private_data_dir):
        if False:
            for i in range(10):
                print('nop')
        ret = super(rhv, self).inventory_as_dict(inventory_source, private_data_dir)
        ret['ovirt_insecure'] = False
        ret['compose'] = {'ansible_host': '(devices.values() | list)[0][0] if devices else None'}
        ret['keyed_groups'] = []
        for key in ('cluster', 'status'):
            ret['keyed_groups'].append({'prefix': key, 'separator': '_', 'key': key})
        ret['keyed_groups'].append({'prefix': 'tag', 'separator': '_', 'key': 'tags'})
        ret['ovirt_hostname_preference'] = ['name', 'fqdn']
        source_vars = inventory_source.source_vars_dict
        for (key, value) in source_vars.items():
            if key == 'plugin':
                continue
            ret[key] = value
        return ret

class satellite6(PluginFileInjector):
    plugin_name = 'foreman'
    namespace = 'theforeman'
    collection = 'foreman'

    def inventory_as_dict(self, inventory_source, private_data_dir):
        if False:
            i = 10
            return i + 15
        ret = super(satellite6, self).inventory_as_dict(inventory_source, private_data_dir)
        ret['validate_certs'] = False
        group_patterns = '[]'
        group_prefix = 'foreman_'
        want_hostcollections = False
        want_ansible_ssh_host = False
        want_facts = True
        foreman_opts = inventory_source.source_vars_dict.copy()
        for (k, v) in foreman_opts.items():
            if k == 'satellite6_group_patterns' and isinstance(v, str):
                group_patterns = v
            elif k == 'satellite6_group_prefix' and isinstance(v, str):
                group_prefix = v
            elif k == 'satellite6_want_hostcollections' and isinstance(v, bool):
                want_hostcollections = v
            elif k == 'satellite6_want_ansible_ssh_host' and isinstance(v, bool):
                want_ansible_ssh_host = v
            elif k == 'satellite6_want_facts' and isinstance(v, bool):
                want_facts = v
            elif k == 'ssl_verify' and isinstance(v, bool):
                ret['validate_certs'] = v
            else:
                ret[k] = str(v)
        group_by_hostvar = {'environment': {'prefix': '{}environment_'.format(group_prefix), 'separator': '', 'key': "foreman['environment_name'] | lower | regex_replace(' ', '') | regex_replace('[^A-Za-z0-9_]', '_') | regex_replace('none', '')"}, 'location': {'prefix': '{}location_'.format(group_prefix), 'separator': '', 'key': "foreman['location_name'] | lower | regex_replace(' ', '') | regex_replace('[^A-Za-z0-9_]', '_')"}, 'organization': {'prefix': '{}organization_'.format(group_prefix), 'separator': '', 'key': "foreman['organization_name'] | lower | regex_replace(' ', '') | regex_replace('[^A-Za-z0-9_]', '_')"}, 'lifecycle_environment': {'prefix': '{}lifecycle_environment_'.format(group_prefix), 'separator': '', 'key': "foreman['content_facet_attributes']['lifecycle_environment_name'] | lower | regex_replace(' ', '') | regex_replace('[^A-Za-z0-9_]', '_')"}, 'content_view': {'prefix': '{}content_view_'.format(group_prefix), 'separator': '', 'key': "foreman['content_facet_attributes']['content_view_name'] | lower | regex_replace(' ', '') | regex_replace('[^A-Za-z0-9_]', '_')"}}
        ret['legacy_hostvars'] = True
        ret['want_params'] = True
        ret['group_prefix'] = group_prefix
        ret['want_hostcollections'] = want_hostcollections
        ret['want_facts'] = want_facts
        if want_ansible_ssh_host:
            ret['compose'] = {'ansible_ssh_host': "foreman['ip6'] | default(foreman['ip'], true)"}
        ret['keyed_groups'] = [group_by_hostvar[grouping_name] for grouping_name in group_by_hostvar]

        def form_keyed_group(group_pattern):
            if False:
                return 10
            '\n            Converts foreman group_pattern to\n            inventory plugin keyed_group\n\n            e.g. {app_param}-{tier_param}-{dc_param}\n                 becomes\n                 "%s-%s-%s" | format(app_param, tier_param, dc_param)\n            '
            if type(group_pattern) is not str:
                return None
            params = re.findall('{[^}]*}', group_pattern)
            if len(params) == 0:
                return None
            param_names = []
            for p in params:
                param_names.append(p[1:-1].strip())
            key = group_pattern
            for p in params:
                key = key.replace(p, '%s', 1)
            key = '"{}" | format({})'.format(key, ', '.join(param_names))
            keyed_group = {'key': key, 'separator': ''}
            return keyed_group
        try:
            group_patterns = json.loads(group_patterns)
            if type(group_patterns) is list:
                for group_pattern in group_patterns:
                    keyed_group = form_keyed_group(group_pattern)
                    if keyed_group:
                        ret['keyed_groups'].append(keyed_group)
        except json.JSONDecodeError:
            logger.warning('Could not parse group_patterns. Expected JSON-formatted string, found: {}'.format(group_patterns))
        return ret

class tower(PluginFileInjector):
    plugin_name = 'tower'
    namespace = 'awx'
    collection = 'awx'

    def inventory_as_dict(self, inventory_source, private_data_dir):
        if False:
            print('Hello World!')
        ret = super(tower, self).inventory_as_dict(inventory_source, private_data_dir)
        try:
            identifier = int(inventory_source.instance_filters)
        except ValueError:
            identifier = iri_to_uri(inventory_source.instance_filters)
        ret['inventory_id'] = identifier
        ret['include_metadata'] = True
        return ret
for cls in PluginFileInjector.__subclasses__():
    FrozenInjectors[cls.__name__] = cls