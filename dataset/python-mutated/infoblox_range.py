"""
Infoblox host record management.

functions accept api_opts:

    api_verifyssl: verify SSL [default to True or pillar value]
    api_url: server to connect to [default to pillar value]
    api_username:  [default to pillar value]
    api_password:  [default to pillar value]
"""

def present(name=None, start_addr=None, end_addr=None, data=None, **api_opts):
    if False:
        i = 10
        return i + 15
    "\n    Ensure range record is present.\n\n    infoblox_range.present:\n        start_addr: '129.97.150.160',\n        end_addr: '129.97.150.170',\n\n    Verbose state example:\n\n    .. code-block:: yaml\n\n        infoblox_range.present:\n            data: {\n                'always_update_dns': False,\n                'authority': False,\n                'comment': 'range of IP addresses used for salt.. was used for ghost images deployment',\n                'ddns_generate_hostname': True,\n                'deny_all_clients': False,\n                'deny_bootp': False,\n                'disable': False,\n                'email_list': [],\n                'enable_ddns': False,\n                'enable_dhcp_thresholds': False,\n                'enable_email_warnings': False,\n                'enable_ifmap_publishing': False,\n                'enable_snmp_warnings': False,\n                'end_addr': '129.97.150.169',\n                'exclude': [],\n                'extattrs': {},\n                'fingerprint_filter_rules': [],\n                'high_water_mark': 95,\n                'high_water_mark_reset': 85,\n                'ignore_dhcp_option_list_request': False,\n                'lease_scavenge_time': -1,\n                'logic_filter_rules': [],\n                'low_water_mark': 0,\n                'low_water_mark_reset': 10,\n                'mac_filter_rules': [],\n                'member': {'_struct': 'dhcpmember',\n                        'ipv4addr': '129.97.128.9',\n                        'name': 'cn-dhcp-mc.example.ca'},\n                'ms_options': [],\n                'nac_filter_rules': [],\n                'name': 'ghost-range',\n                'network': '129.97.150.0/24',\n                'network_view': 'default',\n                'option_filter_rules': [],\n                'options': [{'name': 'dhcp-lease-time',\n                            'num': 51,\n                            'use_option': False,\n                            'value': '43200',\n                            'vendor_class': 'DHCP'}],\n                'recycle_leases': True,\n                'relay_agent_filter_rules': [],\n                'server_association_type': 'MEMBER',\n                'start_addr': '129.97.150.160',\n                'update_dns_on_lease_renewal': False,\n                'use_authority': False,\n                'use_bootfile': False,\n                'use_bootserver': False,\n                'use_ddns_domainname': False,\n                'use_ddns_generate_hostname': True,\n                'use_deny_bootp': False,\n                'use_email_list': False,\n                'use_enable_ddns': False,\n                'use_enable_dhcp_thresholds': False,\n                'use_enable_ifmap_publishing': False,\n                'use_ignore_dhcp_option_list_request': False,\n                'use_known_clients': False,\n                'use_lease_scavenge_time': False,\n                'use_nextserver': False,\n                'use_options': False,\n                'use_recycle_leases': False,\n                'use_unknown_clients': False,\n                'use_update_dns_on_lease_renewal': False\n            }\n    "
    ret = {'name': name, 'result': False, 'comment': '', 'changes': {}}
    if not data:
        data = {}
    if 'name' not in data:
        data.update({'name': name})
    if 'start_addr' not in data:
        data.update({'start_addr': start_addr})
    if 'end_addr' not in data:
        data.update({'end_addr': end_addr})
    obj = __salt__['infoblox.get_ipv4_range'](data['start_addr'], data['end_addr'], **api_opts)
    if obj is None:
        obj = __salt__['infoblox.get_ipv4_range'](start_addr=data['start_addr'], end_addr=None, **api_opts)
        if obj is None:
            obj = __salt__['infoblox.get_ipv4_range'](start_addr=None, end_addr=data['end_addr'], **api_opts)
    if obj:
        diff = __salt__['infoblox.diff_objects'](data, obj)
        if not diff:
            ret['result'] = True
            ret['comment'] = 'supplied fields in correct state'
            return ret
        if diff:
            if __opts__['test']:
                ret['result'] = None
                ret['comment'] = 'would attempt to update record'
                return ret
            new_obj = __salt__['infoblox.update_object'](obj['_ref'], data=data, **api_opts)
            ret['result'] = True
            ret['comment'] = 'record fields updated'
            ret['changes'] = {'diff': diff}
            return ret
    if __opts__['test']:
        ret['result'] = None
        ret['comment'] = 'would attempt to create record {}'.format(name)
        return ret
    new_obj_ref = __salt__['infoblox.create_ipv4_range'](data, **api_opts)
    new_obj = __salt__['infoblox.get_ipv4_range'](data['start_addr'], data['end_addr'], **api_opts)
    ret['result'] = True
    ret['comment'] = 'record created'
    ret['changes'] = {'old': 'None', 'new': {'_ref': new_obj_ref, 'data': new_obj}}
    return ret

def absent(name=None, start_addr=None, end_addr=None, data=None, **api_opts):
    if False:
        for i in range(10):
            print('nop')
    "\n    Ensure the range is removed\n\n    Supplying the end of the range is optional.\n\n    State example:\n\n    .. code-block:: yaml\n\n        infoblox_range.absent:\n            - name: 'vlan10'\n\n        infoblox_range.absent:\n            - name:\n            - start_addr: 127.0.1.20\n    "
    ret = {'name': name, 'result': False, 'comment': '', 'changes': {}}
    if not data:
        data = {}
    if 'name' not in data:
        data.update({'name': name})
    if 'start_addr' not in data:
        data.update({'start_addr': start_addr})
    if 'end_addr' not in data:
        data.update({'end_addr': end_addr})
    obj = __salt__['infoblox.get_ipv4_range'](data['start_addr'], data['end_addr'], **api_opts)
    if obj is None:
        obj = __salt__['infoblox.get_ipv4_range'](start_addr=data['start_addr'], end_addr=None, **api_opts)
        if obj is None:
            obj = __salt__['infoblox.get_ipv4_range'](start_addr=None, end_addr=data['end_addr'], **api_opts)
    if not obj:
        ret['result'] = True
        ret['comment'] = 'already deleted'
        return ret
    if __opts__['test']:
        ret['result'] = None
        ret['comment'] = 'would attempt to delete range'
        return ret
    if __salt__['infoblox.delete_object'](objref=obj['_ref']):
        ret['result'] = True
        ret['changes'] = {'old': 'Found {} - {}'.format(start_addr, end_addr), 'new': 'Removed'}
    return ret