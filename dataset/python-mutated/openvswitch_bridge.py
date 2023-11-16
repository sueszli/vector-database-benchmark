"""
Management of Open vSwitch bridges.
"""

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Only make these states available if Open vSwitch module is available.\n    '
    if 'openvswitch.bridge_create' in __salt__:
        return True
    return (False, 'openvswitch module could not be loaded')

def present(name, parent=None, vlan=None):
    if False:
        print('Hello World!')
    '\n    Ensures that the named bridge exists, eventually creates it.\n\n    Args:\n        name : string\n            name of the bridge\n        parent : string\n            name of the parent bridge (if the bridge shall be created as a fake\n            bridge). If specified, vlan must also be specified.\n        .. versionadded:: 3006.0\n        vlan: int\n            VLAN ID of the bridge (if the bridge shall be created as a fake\n            bridge). If specified, parent must also be specified.\n        .. versionadded:: 3006.0\n\n    '
    ret = {'name': name, 'changes': {}, 'result': False, 'comment': ''}
    comment_bridge_created = f'Bridge {name} created.'
    comment_bridge_notcreated = f'Unable to create bridge: {name}.'
    comment_bridge_exists = f'Bridge {name} already exists.'
    comment_bridge_mismatch = 'Bridge {} already exists, but has a different parent or VLAN ID.'.format(name)
    changes_bridge_created = {name: {'old': f'Bridge {name} does not exist.', 'new': f'Bridge {name} created'}}
    bridge_exists = __salt__['openvswitch.bridge_exists'](name)
    if bridge_exists:
        current_parent = __salt__['openvswitch.bridge_to_parent'](name)
        if current_parent == name:
            current_parent = None
        current_vlan = __salt__['openvswitch.bridge_to_vlan'](name)
        if current_vlan == 0:
            current_vlan = None
    if __opts__['test']:
        if bridge_exists:
            if current_parent == parent and current_vlan == vlan:
                ret['result'] = True
                ret['comment'] = comment_bridge_exists
            else:
                ret['result'] = False
                ret['comment'] = comment_bridge_mismatch
        else:
            ret['result'] = None
            ret['comment'] = comment_bridge_created
        return ret
    if bridge_exists:
        if current_parent == parent and current_vlan == vlan:
            ret['result'] = True
            ret['comment'] = comment_bridge_exists
        else:
            ret['result'] = False
            ret['comment'] = comment_bridge_mismatch
    else:
        bridge_create = __salt__['openvswitch.bridge_create'](name, parent=parent, vlan=vlan)
        if bridge_create:
            ret['result'] = True
            ret['comment'] = comment_bridge_created
            ret['changes'] = changes_bridge_created
        else:
            ret['result'] = False
            ret['comment'] = comment_bridge_notcreated
    return ret

def absent(name):
    if False:
        return 10
    '\n    Ensures that the named bridge does not exist, eventually deletes it.\n\n    Args:\n        name: The name of the bridge.\n\n    '
    ret = {'name': name, 'changes': {}, 'result': False, 'comment': ''}
    comment_bridge_deleted = f'Bridge {name} deleted.'
    comment_bridge_notdeleted = f'Unable to delete bridge: {name}.'
    comment_bridge_notexists = f'Bridge {name} does not exist.'
    changes_bridge_deleted = {name: {'old': f'Bridge {name} exists.', 'new': f'Bridge {name} deleted.'}}
    bridge_exists = __salt__['openvswitch.bridge_exists'](name)
    if __opts__['test']:
        if not bridge_exists:
            ret['result'] = True
            ret['comment'] = comment_bridge_notexists
        else:
            ret['result'] = None
            ret['comment'] = comment_bridge_deleted
        return ret
    if not bridge_exists:
        ret['result'] = True
        ret['comment'] = comment_bridge_notexists
    else:
        bridge_delete = __salt__['openvswitch.bridge_delete'](name)
        if bridge_delete:
            ret['result'] = True
            ret['comment'] = comment_bridge_deleted
            ret['changes'] = changes_bridge_deleted
        else:
            ret['result'] = False
            ret['comment'] = comment_bridge_notdeleted
    return ret