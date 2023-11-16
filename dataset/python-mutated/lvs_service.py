"""
Management of LVS (Linux Virtual Server) Service
================================================
"""

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Only load if the lvs module is available in __salt__\n    '
    if 'lvs.get_rules' in __salt__:
        return 'lvs_service'
    return (False, 'lvs module could not be loaded')

def present(name, protocol=None, service_address=None, scheduler='wlc'):
    if False:
        print('Hello World!')
    '\n    Ensure that the named service is present.\n\n    name\n        The LVS service name\n\n    protocol\n        The service protocol\n\n    service_address\n        The LVS service address\n\n    scheduler\n        Algorithm for allocating TCP connections and UDP datagrams to real servers.\n\n    .. code-block:: yaml\n\n        lvstest:\n          lvs_service.present:\n            - service_address: 1.1.1.1:80\n            - protocol: tcp\n            - scheduler: rr\n    '
    ret = {'name': name, 'changes': {}, 'result': True, 'comment': ''}
    service_check = __salt__['lvs.check_service'](protocol=protocol, service_address=service_address)
    if service_check is True:
        service_rule_check = __salt__['lvs.check_service'](protocol=protocol, service_address=service_address, scheduler=scheduler)
        if service_rule_check is True:
            ret['comment'] = 'LVS Service {} is present'.format(name)
            return ret
        elif __opts__['test']:
            ret['result'] = None
            ret['comment'] = 'LVS Service {} is present but some options should update'.format(name)
            return ret
        else:
            service_edit = __salt__['lvs.edit_service'](protocol=protocol, service_address=service_address, scheduler=scheduler)
            if service_edit is True:
                ret['comment'] = 'LVS Service {} has been updated'.format(name)
                ret['changes'][name] = 'Update'
                return ret
            else:
                ret['result'] = False
                ret['comment'] = 'LVS Service {} update failed'.format(name)
                return ret
    elif __opts__['test']:
        ret['comment'] = 'LVS Service {} is not present and needs to be created'.format(name)
        ret['result'] = None
        return ret
    else:
        service_add = __salt__['lvs.add_service'](protocol=protocol, service_address=service_address, scheduler=scheduler)
        if service_add is True:
            ret['comment'] = 'LVS Service {} has been created'.format(name)
            ret['changes'][name] = 'Present'
            return ret
        else:
            ret['comment'] = 'LVS Service {} create failed({})'.format(name, service_add)
            ret['result'] = False
            return ret

def absent(name, protocol=None, service_address=None):
    if False:
        print('Hello World!')
    '\n    Ensure the LVS service is absent.\n\n    name\n        The name of the LVS service\n\n    protocol\n        The service protocol\n\n    service_address\n        The LVS service address\n    '
    ret = {'name': name, 'changes': {}, 'result': True, 'comment': ''}
    service_check = __salt__['lvs.check_service'](protocol=protocol, service_address=service_address)
    if service_check is True:
        if __opts__['test']:
            ret['result'] = None
            ret['comment'] = 'LVS Service {} is present and needs to be removed'.format(name)
            return ret
        service_delete = __salt__['lvs.delete_service'](protocol=protocol, service_address=service_address)
        if service_delete is True:
            ret['comment'] = 'LVS Service {} has been removed'.format(name)
            ret['changes'][name] = 'Absent'
            return ret
        else:
            ret['comment'] = 'LVS Service {} removed failed({})'.format(name, service_delete)
            ret['result'] = False
            return ret
    else:
        ret['comment'] = 'LVS Service {} is not present, so it cannot be removed'.format(name)
    return ret