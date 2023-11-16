"""
Management of Zabbix hosts.

:codeauthor: Jiri Kotlin <jiri.kotlin@ultimum.io>


"""
from collections.abc import Mapping
from copy import deepcopy
import salt.utils.dictdiffer
import salt.utils.json
__deprecated__ = (3009, 'zabbix', 'https://github.com/salt-extensions/saltext-zabbix')

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Only make these states available if Zabbix module is available.\n    '
    if 'zabbix.host_create' in __salt__:
        return True
    return (False, 'zabbix module could not be loaded')

def present(host, groups, interfaces, **kwargs):
    if False:
        print('Hello World!')
    "\n    Ensures that the host exists, eventually creates new host.\n    NOTE: please use argument visible_name instead of name to not mess with name from salt sls. This function accepts\n    all standard host properties: keyword argument names differ depending on your zabbix version, see:\n    https://www.zabbix.com/documentation/2.4/manual/api/reference/host/object#host\n\n    .. versionadded:: 2016.3.0\n\n    :param host: technical name of the host\n    :param groups: groupids of host groups to add the host to\n    :param interfaces: interfaces to be created for the host\n    :param proxy_host: Optional proxy name or proxyid to monitor host\n    :param inventory: Optional list or dictionary of inventory names and values\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n    :param visible_name: Optional - string with visible name of the host, use 'visible_name' instead of 'name'\n        parameter to not mess with value supplied from Salt sls file.\n    :param inventory_clean: Optional - Boolean value that selects if the current inventory will be cleaned and\n        overwritten by the declared inventory list (True); or if the inventory will be kept and only updated with\n        inventory list contents (False). Defaults to True\n\n    .. code-block:: yaml\n\n        create_test_host:\n            zabbix_host.present:\n                - host: TestHostWithInterfaces\n                - proxy_host: 12345\n                - groups:\n                    - 5\n                    - 6\n                    - 7\n                - interfaces:\n                    - test1.example.com:\n                        - ip: '192.168.1.8'\n                        - type: 'Agent'\n                        - port: 92\n                    - testing2_create:\n                        - ip: '192.168.1.9'\n                        - dns: 'test2.example.com'\n                        - type: 'agent'\n                        - main: false\n                    - testovaci1_ipmi:\n                        - ip: '192.168.100.111'\n                        - type: 'ipmi'\n                - inventory:\n                    - alias: some alias\n                    - asset_tag: jlm3937\n\n\n    "
    connection_args = {}
    if '_connection_user' in kwargs:
        connection_args['_connection_user'] = kwargs.pop('_connection_user')
    if '_connection_password' in kwargs:
        connection_args['_connection_password'] = kwargs.pop('_connection_password')
    if '_connection_url' in kwargs:
        connection_args['_connection_url'] = kwargs.pop('_connection_url')
    ret = {'name': host, 'changes': {}, 'result': False, 'comment': ''}
    comment_host_created = f'Host {host} created.'
    comment_host_updated = f'Host {host} updated.'
    comment_host_notcreated = f'Unable to create host: {host}. '
    comment_host_exists = f'Host {host} already exists.'
    changes_host_created = {host: {'old': f'Host {host} does not exist.', 'new': f'Host {host} created.'}}

    def _interface_format(interfaces_data):
        if False:
            print('Hello World!')
        '\n        Formats interfaces from SLS file into valid JSON usable for zabbix API.\n        Completes JSON with default values.\n\n        :param interfaces_data: list of interfaces data from SLS file\n\n        '
        if not interfaces_data:
            return list()
        interface_attrs = ('ip', 'dns', 'main', 'type', 'useip', 'port', 'details')
        interfaces_json = salt.utils.json.loads(salt.utils.json.dumps(interfaces_data))
        interfaces_dict = dict()
        for interface in interfaces_json:
            for intf in interface:
                intf_name = intf
                interfaces_dict[intf_name] = dict()
                for intf_val in interface[intf]:
                    for (key, value) in intf_val.items():
                        if key in interface_attrs:
                            interfaces_dict[intf_name][key] = value
        interfaces_list = list()
        interface_ports = {'agent': ['1', '10050'], 'snmp': ['2', '161'], 'ipmi': ['3', '623'], 'jmx': ['4', '12345']}
        for (key, value) in interfaces_dict.items():
            interface_type = interface_ports[value['type'].lower()][0]
            main = '1' if str(value.get('main', 'true')).lower() == 'true' else '0'
            useip = '1' if str(value.get('useip', 'true')).lower() == 'true' else '0'
            interface_ip = value.get('ip', '')
            dns = value.get('dns', key)
            port = str(value.get('port', interface_ports[value['type'].lower()][1]))
            if interface_type == '2':
                if not value.get('details', False):
                    details_version = '2'
                    details_bulk = '1'
                    details_community = '{$SNMP_COMMUNITY}'
                else:
                    val_details = {}
                    for detail in value.get('details'):
                        val_details.update(detail)
                    details_version = val_details.get('version', '2')
                    details_bulk = val_details.get('bulk', '1')
                    details_community = val_details.get('community', '{$SNMP_COMMUNITY}')
                details = {'version': details_version, 'bulk': details_bulk, 'community': details_community}
                if details_version == '3':
                    details_securitylevel = val_details.get('securitylevel', '0')
                    details_securityname = val_details.get('securityname', '')
                    details_contextname = val_details.get('contextname', '')
                    details['securitylevel'] = details_securitylevel
                    details['securityname'] = details_securityname
                    details['contextname'] = details_contextname
                    if int(details_securitylevel) > 0:
                        details_authpassphrase = val_details.get('authpassphrase', '')
                        details_authprotocol = val_details.get('authprotocol', '0')
                        details['authpassphrase'] = details_authpassphrase
                        details['authprotocol'] = details_authprotocol
                        if int(details_securitylevel) > 1:
                            details_privpassphrase = val_details.get('privpassphrase', '')
                            details_privprotocol = val_details.get('privprotocol', '0')
                            details['privpassphrase'] = details_privpassphrase
                            details['privprotocol'] = details_privprotocol
            else:
                details = []
            interfaces_list.append({'type': interface_type, 'main': main, 'useip': useip, 'ip': interface_ip, 'dns': dns, 'port': port, 'details': details})
        interfaces_list_sorted = sorted(interfaces_list, key=lambda k: k['main'], reverse=True)
        return interfaces_list_sorted
    interfaces_formated = _interface_format(interfaces)
    groupids = []
    for group in groups:
        if isinstance(group, str):
            groupid = __salt__['zabbix.hostgroup_get'](name=group, **connection_args)
            try:
                groupids.append(int(groupid[0]['groupid']))
            except TypeError:
                ret['comment'] = f'Invalid group {group}'
                return ret
        else:
            groupids.append(group)
    groups = groupids
    proxy_hostid = '0'
    if 'proxy_host' in kwargs:
        proxy_host = kwargs.pop('proxy_host')
        if isinstance(proxy_host, str):
            try:
                proxy_hostid = __salt__['zabbix.run_query']('proxy.get', {'output': 'proxyid', 'selectInterface': 'extend', 'filter': {'host': f'{proxy_host}'}}, **connection_args)[0]['proxyid']
            except TypeError:
                ret['comment'] = f'Invalid proxy_host {proxy_host}'
                return ret
        else:
            try:
                proxy_hostid = __salt__['zabbix.run_query']('proxy.get', {'proxyids': f'{proxy_host}', 'output': 'proxyid'}, **connection_args)[0]['proxyid']
            except TypeError:
                ret['comment'] = f'Invalid proxy_host {proxy_host}'
                return ret
    inventory_clean = kwargs.pop('inventory_clean', True)
    inventory = kwargs.pop('inventory', None)
    new_inventory = {}
    if isinstance(inventory, Mapping):
        new_inventory = dict(inventory)
    elif inventory is not None:
        for inv_item in inventory:
            for (k, v) in inv_item.items():
                new_inventory[k] = str(v)
    visible_name = kwargs.pop('visible_name', None)
    host_extra_properties = {}
    if kwargs:
        host_properties_definition = ['description', 'inventory_mode', 'ipmi_authtype', 'ipmi_password', 'ipmi_privilege', 'ipmi_username', 'status', 'tls_connect', 'tls_accept', 'tls_issuer', 'tls_subject', 'tls_psk_identity', 'tls_psk']
        for param in host_properties_definition:
            if param in kwargs:
                host_extra_properties[param] = kwargs.pop(param)
    host_exists = __salt__['zabbix.host_exists'](host, **connection_args)
    if host_exists:
        host = __salt__['zabbix.host_get'](host=host, **connection_args)[0]
        hostid = host['hostid']
        update_host = False
        update_proxy = False
        update_hostgroups = False
        update_interfaces = False
        update_inventory = False
        host_updated_params = {}
        for param in host_extra_properties:
            if param in host:
                if host[param] == host_extra_properties[param]:
                    continue
            host_updated_params[param] = host_extra_properties[param]
        if host_updated_params:
            update_host = True
        host_inventory_mode = host['inventory_mode']
        inventory_mode = host_extra_properties.get('inventory_mode', '0' if host_inventory_mode == '-1' else host_inventory_mode)
        cur_proxy_hostid = host['proxy_hostid']
        if proxy_hostid != cur_proxy_hostid:
            update_proxy = True
        hostgroups = __salt__['zabbix.hostgroup_get'](hostids=hostid, **connection_args)
        cur_hostgroups = list()
        for hostgroup in hostgroups:
            cur_hostgroups.append(int(hostgroup['groupid']))
        if set(groups) != set(cur_hostgroups):
            update_hostgroups = True
        hostinterfaces = __salt__['zabbix.hostinterface_get'](hostids=hostid, **connection_args)
        if hostinterfaces:
            hostinterfaces = sorted(hostinterfaces, key=lambda k: k['main'])
            hostinterfaces_copy = deepcopy(hostinterfaces)
            for hostintf in hostinterfaces_copy:
                hostintf.pop('interfaceid')
                hostintf.pop('hostid')
                if 'bulk' in hostintf:
                    hostintf.pop('bulk')
                    if hostintf['type'] == '2':
                        hostintf['details'] = {'version': '2', 'bulk': '1', 'community': '{$SNMP_COMMUNITY}'}
                    else:
                        hostintf['details'] = []
            interface_diff = [x for x in interfaces_formated if x not in hostinterfaces_copy] + [y for y in hostinterfaces_copy if y not in interfaces_formated]
            if interface_diff:
                update_interfaces = True
        elif not hostinterfaces and interfaces:
            update_interfaces = True
        if inventory is not None and inventory_mode != '-1':
            cur_inventory = __salt__['zabbix.host_inventory_get'](hostids=hostid, **connection_args)
            inventory_diff = salt.utils.dictdiffer.diff(cur_inventory, new_inventory)
            if inventory_diff.changed():
                update_inventory = True
    if __opts__['test']:
        if host_exists:
            if update_host or update_hostgroups or update_interfaces or update_proxy or update_inventory:
                ret['result'] = None
                ret['comment'] = comment_host_updated
            else:
                ret['result'] = True
                ret['comment'] = comment_host_exists
        else:
            ret['result'] = None
            ret['comment'] = comment_host_created
            ret['changes'] = changes_host_created
        return ret
    error = []
    if host_exists:
        ret['result'] = True
        if update_host or update_hostgroups or update_interfaces or update_proxy or update_inventory:
            if update_host:
                sum_kwargs = deepcopy(host_updated_params)
                sum_kwargs.update(connection_args)
                hostupdate = __salt__['zabbix.host_update'](hostid, **sum_kwargs)
                ret['changes']['host'] = str(host_updated_params)
                if 'error' in hostupdate:
                    error.append(hostupdate['error'])
            if update_inventory:
                sum_kwargs = deepcopy(new_inventory)
                sum_kwargs.update(connection_args)
                sum_kwargs['clear_old'] = inventory_clean
                sum_kwargs['inventory_mode'] = inventory_mode
                hostupdate = __salt__['zabbix.host_inventory_set'](hostid, **sum_kwargs)
                ret['changes']['inventory'] = str(new_inventory)
                if 'error' in hostupdate:
                    error.append(hostupdate['error'])
            if update_proxy:
                hostupdate = __salt__['zabbix.host_update'](hostid, proxy_hostid=proxy_hostid, **connection_args)
                ret['changes']['proxy_hostid'] = str(proxy_hostid)
                if 'error' in hostupdate:
                    error.append(hostupdate['error'])
            if update_hostgroups:
                hostupdate = __salt__['zabbix.host_update'](hostid, groups=groups, **connection_args)
                ret['changes']['groups'] = str(groups)
                if 'error' in hostupdate:
                    error.append(hostupdate['error'])
            if update_interfaces:
                interfaceid_by_type = {'1': [], '2': [], '3': [], '4': []}
                other_interfaces = []
                if hostinterfaces:
                    for interface in hostinterfaces:
                        if interface['main']:
                            interfaceid_by_type[interface['type']].insert(0, interface['interfaceid'])
                        else:
                            interfaceid_by_type[interface['type']].append(interface['interfaceid'])

                def _update_interfaces(interface):
                    if False:
                        while True:
                            i = 10
                    if not interfaceid_by_type[interface['type']]:
                        ret = __salt__['zabbix.hostinterface_create'](hostid, interface['ip'], dns=interface['dns'], main=interface['main'], if_type=interface['type'], useip=interface['useip'], port=interface['port'], details=interface['details'], **connection_args)
                    else:
                        interfaceid = interfaceid_by_type[interface['type']].pop(0)
                        ret = __salt__['zabbix.hostinterface_update'](interfaceid=interfaceid, ip=interface['ip'], dns=interface['dns'], main=interface['main'], type=interface['type'], useip=interface['useip'], port=interface['port'], details=interface['details'], **connection_args)
                    return ret
                for interface in interfaces_formated:
                    if interface['main']:
                        updatedint = _update_interfaces(interface)
                        if 'error' in updatedint:
                            error.append(updatedint['error'])
                    else:
                        other_interfaces.append(interface)
                for interface in other_interfaces:
                    updatedint = _update_interfaces(interface)
                    if 'error' in updatedint:
                        error.append(updatedint['error'])
                for interface_type in interfaceid_by_type:
                    for interfaceid in interfaceid_by_type[interface_type]:
                        __salt__['zabbix.hostinterface_delete'](interfaceids=interfaceid, **connection_args)
                ret['changes']['interfaces'] = str(interfaces_formated)
            ret['comment'] = comment_host_updated
        else:
            ret['comment'] = comment_host_exists
    else:
        sum_kwargs = host_extra_properties
        sum_kwargs.update(connection_args)
        host_create = __salt__['zabbix.host_create'](host, groups, interfaces_formated, proxy_hostid=proxy_hostid, inventory=new_inventory, visible_name=visible_name, **sum_kwargs)
        if 'error' not in host_create:
            ret['result'] = True
            ret['comment'] = comment_host_created
            ret['changes'] = changes_host_created
        else:
            ret['result'] = False
            ret['comment'] = comment_host_notcreated + str(host_create['error'])
    if error:
        ret['changes'] = {}
        ret['result'] = False
        ret['comment'] = str(error)
    return ret

def absent(name, **kwargs):
    if False:
        print('Hello World!')
    "\n    Ensures that the host does not exists, eventually deletes host.\n\n    .. versionadded:: 2016.3.0\n\n    :param: name: technical name of the host\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n\n    .. code-block:: yaml\n\n        TestHostWithInterfaces:\n            zabbix_host.absent\n\n    "
    ret = {'name': name, 'changes': {}, 'result': False, 'comment': ''}
    comment_host_deleted = f'Host {name} deleted.'
    comment_host_notdeleted = f'Unable to delete host: {name}. '
    comment_host_notexists = f'Host {name} does not exist.'
    changes_host_deleted = {name: {'old': f'Host {name} exists.', 'new': f'Host {name} deleted.'}}
    connection_args = {}
    if '_connection_user' in kwargs:
        connection_args['_connection_user'] = kwargs['_connection_user']
    if '_connection_password' in kwargs:
        connection_args['_connection_password'] = kwargs['_connection_password']
    if '_connection_url' in kwargs:
        connection_args['_connection_url'] = kwargs['_connection_url']
    host_exists = __salt__['zabbix.host_exists'](name, **connection_args)
    if __opts__['test']:
        if not host_exists:
            ret['result'] = True
            ret['comment'] = comment_host_notexists
        else:
            ret['result'] = None
            ret['comment'] = comment_host_deleted
        return ret
    host_get = __salt__['zabbix.host_get'](name, **connection_args)
    if not host_get:
        ret['result'] = True
        ret['comment'] = comment_host_notexists
    else:
        try:
            hostid = host_get[0]['hostid']
            host_delete = __salt__['zabbix.host_delete'](hostid, **connection_args)
        except KeyError:
            host_delete = False
        if host_delete and 'error' not in host_delete:
            ret['result'] = True
            ret['comment'] = comment_host_deleted
            ret['changes'] = changes_host_deleted
        else:
            ret['result'] = False
            ret['comment'] = comment_host_notdeleted + str(host_delete['error'])
    return ret

def assign_templates(host, templates, **kwargs):
    if False:
        return 10
    '\n    Ensures that templates are assigned to the host.\n\n    .. versionadded:: 2017.7.0\n\n    :param host: technical name of the host\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module\'s docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module\'s docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module\'s docstring)\n\n    .. code-block:: yaml\n\n        add_zabbix_templates_to_host:\n            zabbix_host.assign_templates:\n                - host: TestHost\n                - templates:\n                    - "Template OS Linux"\n                    - "Template App MySQL"\n\n    '
    connection_args = {}
    if '_connection_user' in kwargs:
        connection_args['_connection_user'] = kwargs['_connection_user']
    if '_connection_password' in kwargs:
        connection_args['_connection_password'] = kwargs['_connection_password']
    if '_connection_url' in kwargs:
        connection_args['_connection_url'] = kwargs['_connection_url']
    ret = {'name': host, 'changes': {}, 'result': False, 'comment': ''}
    comment_host_templates_updated = 'Templates updated.'
    comment_host_templ_notupdated = 'Unable to update templates on host: {}.'.format(host)
    comment_host_templates_in_sync = 'Templates already synced.'
    update_host_templates = False
    curr_template_ids = list()
    requested_template_ids = list()
    hostid = ''
    host_exists = __salt__['zabbix.host_exists'](host, **connection_args)
    if not host_exists:
        ret['result'] = False
        ret['comment'] = comment_host_templ_notupdated
        return ret
    host_info = __salt__['zabbix.host_get'](host=host, **connection_args)[0]
    hostid = host_info['hostid']
    if not templates:
        templates = list()
    host_templates = __salt__['zabbix.host_get'](hostids=hostid, output='[{"hostid"}]', selectParentTemplates='["templateid"]', **connection_args)
    for template_id in host_templates[0]['parentTemplates']:
        curr_template_ids.append(template_id['templateid'])
    for template in templates:
        try:
            template_id = __salt__['zabbix.template_get'](host=template, **connection_args)[0]['templateid']
            requested_template_ids.append(template_id)
        except TypeError:
            ret['result'] = False
            ret['comment'] = f'Unable to find template: {template}.'
            return ret
    requested_template_ids = list(set(requested_template_ids))
    if set(curr_template_ids) != set(requested_template_ids):
        update_host_templates = True
    changes_host_templates_modified = {host: {'old': 'Host templates: ' + ', '.join(curr_template_ids), 'new': 'Host templates: ' + ', '.join(requested_template_ids)}}
    if __opts__['test']:
        if update_host_templates:
            ret['result'] = None
            ret['comment'] = comment_host_templates_updated
        else:
            ret['result'] = True
            ret['comment'] = comment_host_templates_in_sync
        return ret
    ret['result'] = True
    if update_host_templates:
        update_output = __salt__['zabbix.host_update'](hostid, templates=requested_template_ids, **connection_args)
        if update_output is False:
            ret['result'] = False
            ret['comment'] = comment_host_templ_notupdated
            return ret
        ret['comment'] = comment_host_templates_updated
        ret['changes'] = changes_host_templates_modified
    else:
        ret['comment'] = comment_host_templates_in_sync
    return ret