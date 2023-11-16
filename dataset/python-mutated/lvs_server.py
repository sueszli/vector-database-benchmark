"""
Management of LVS (Linux Virtual Server) Real Server
====================================================
"""

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Only load if the lvs module is available in __salt__\n    '
    if 'lvs.get_rules' in __salt__:
        return 'lvs_server'
    return (False, 'lvs module could not be loaded')

def present(name, protocol=None, service_address=None, server_address=None, packet_forward_method='dr', weight=1):
    if False:
        while True:
            i = 10
    '\n    Ensure that the named service is present.\n\n    name\n        The LVS server name\n\n    protocol\n        The service protocol\n\n    service_address\n        The LVS service address\n\n    server_address\n        The real server address.\n\n    packet_forward_method\n        The LVS packet forwarding method(``dr`` for direct routing, ``tunnel`` for tunneling, ``nat`` for network access translation).\n\n    weight\n        The capacity  of a server relative to the others in the pool.\n\n\n    .. code-block:: yaml\n\n        lvsrs:\n          lvs_server.present:\n            - protocol: tcp\n            - service_address: 1.1.1.1:80\n            - server_address: 192.168.0.11:8080\n            - packet_forward_method: dr\n            - weight: 10\n    '
    ret = {'name': name, 'changes': {}, 'result': True, 'comment': ''}
    server_check = __salt__['lvs.check_server'](protocol=protocol, service_address=service_address, server_address=server_address)
    if server_check is True:
        server_rule_check = __salt__['lvs.check_server'](protocol=protocol, service_address=service_address, server_address=server_address, packet_forward_method=packet_forward_method, weight=weight)
        if server_rule_check is True:
            ret['comment'] = 'LVS Server {} in service {}({}) is present'.format(name, service_address, protocol)
            return ret
        elif __opts__['test']:
            ret['result'] = None
            ret['comment'] = 'LVS Server {} in service {}({}) is present but some options should update'.format(name, service_address, protocol)
            return ret
        else:
            server_edit = __salt__['lvs.edit_server'](protocol=protocol, service_address=service_address, server_address=server_address, packet_forward_method=packet_forward_method, weight=weight)
            if server_edit is True:
                ret['comment'] = 'LVS Server {} in service {}({}) has been updated'.format(name, service_address, protocol)
                ret['changes'][name] = 'Update'
                return ret
            else:
                ret['result'] = False
                ret['comment'] = 'LVS Server {} in service {}({}) update failed({})'.format(name, service_address, protocol, server_edit)
                return ret
    elif __opts__['test']:
        ret['comment'] = 'LVS Server {} in service {}({}) is not present and needs to be created'.format(name, service_address, protocol)
        ret['result'] = None
        return ret
    else:
        server_add = __salt__['lvs.add_server'](protocol=protocol, service_address=service_address, server_address=server_address, packet_forward_method=packet_forward_method, weight=weight)
        if server_add is True:
            ret['comment'] = 'LVS Server {} in service {}({}) has been created'.format(name, service_address, protocol)
            ret['changes'][name] = 'Present'
            return ret
        else:
            ret['comment'] = 'LVS Service {} in service {}({}) create failed({})'.format(name, service_address, protocol, server_add)
            ret['result'] = False
            return ret

def absent(name, protocol=None, service_address=None, server_address=None):
    if False:
        i = 10
        return i + 15
    '\n    Ensure the LVS Real Server in specified service is absent.\n\n    name\n        The name of the LVS server.\n\n    protocol\n        The service protocol(only support ``tcp``, ``udp`` and ``fwmark`` service).\n\n    service_address\n        The LVS service address.\n\n    server_address\n        The LVS real server address.\n    '
    ret = {'name': name, 'changes': {}, 'result': True, 'comment': ''}
    server_check = __salt__['lvs.check_server'](protocol=protocol, service_address=service_address, server_address=server_address)
    if server_check is True:
        if __opts__['test']:
            ret['result'] = None
            ret['comment'] = 'LVS Server {} in service {}({}) is present and needs to be removed'.format(name, service_address, protocol)
            return ret
        server_delete = __salt__['lvs.delete_server'](protocol=protocol, service_address=service_address, server_address=server_address)
        if server_delete is True:
            ret['comment'] = 'LVS Server {} in service {}({}) has been removed'.format(name, service_address, protocol)
            ret['changes'][name] = 'Absent'
            return ret
        else:
            ret['comment'] = 'LVS Server {} in service {}({}) removed failed({})'.format(name, service_address, protocol, server_delete)
            ret['result'] = False
            return ret
    else:
        ret['comment'] = 'LVS Server {} in service {}({}) is not present, so it cannot be removed'.format(name, service_address, protocol)
    return ret