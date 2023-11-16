"""
Manage PagerDuty schedules.

Example:

.. code-block:: yaml

    ensure test schedule:
        pagerduty_schedule.present:
            - name: 'bruce test schedule level1'
            - schedule:
                name: 'bruce test schedule level1'
                time_zone: 'Pacific Time (US & Canada)'
                schedule_layers:
                    - name: 'Schedule Layer 1'
                      start: '2015-01-01T00:00:00'
                      users:
                        - user:
                            'id': 'Bruce TestUser1'
                          member_order: 1
                        - user:
                            'id': 'Bruce TestUser2'
                          member_order: 2
                        - user:
                            'id': 'bruce+test3@lyft.com'
                          member_order: 3
                        - user:
                            'id': 'bruce+test4@lyft.com'
                          member_order: 4
                      rotation_virtual_start: '2015-01-01T00:00:00'
                      priority: 1
                      rotation_turn_length_seconds: 604800

"""

def __virtual__():
    if False:
        return 10
    '\n    Only load if the pygerduty module is available in __salt__\n    '
    if 'pagerduty_util.get_resource' in __salt__:
        return 'pagerduty_schedule'
    return (False, 'pagerduty_util module could not be loaded')

def present(profile='pagerduty', subdomain=None, api_key=None, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Ensure that a pagerduty schedule exists.\n    This method accepts as args everything defined in\n    https://developer.pagerduty.com/documentation/rest/schedules/create.\n    This means that most arguments are in a dict called "schedule."\n\n    User id\'s can be pagerduty id, or name, or email address.\n    '
    kwargs['schedule']['name'] = kwargs['name']
    for schedule_layer in kwargs['schedule']['schedule_layers']:
        for user in schedule_layer['users']:
            u = __salt__['pagerduty_util.get_resource']('users', user['user']['id'], ['email', 'name', 'id'], profile=profile, subdomain=subdomain, api_key=api_key)
            if u is None:
                raise Exception('unknown user: {}'.format(user))
            user['user']['id'] = u['id']
    r = __salt__['pagerduty_util.resource_present']('schedules', ['name', 'id'], _diff, profile, subdomain, api_key, **kwargs)
    return r

def absent(profile='pagerduty', subdomain=None, api_key=None, **kwargs):
    if False:
        print('Hello World!')
    '\n    Ensure that a pagerduty schedule does not exist.\n    Name can be pagerduty schedule id or pagerduty schedule name.\n    '
    r = __salt__['pagerduty_util.resource_absent']('schedules', ['name', 'id'], profile, subdomain, api_key, **kwargs)
    return r

def _diff(state_data, resource_object):
    if False:
        return 10
    'helper method to compare salt state info with the PagerDuty API json structure,\n    and determine if we need to update.\n\n    returns the dict to pass to the PD API to perform the update, or empty dict if no update.\n    '
    state_data['id'] = resource_object['schedule']['id']
    objects_differ = None
    for (k, v) in state_data['schedule'].items():
        if k == 'schedule_layers':
            continue
        if v != resource_object['schedule'][k]:
            objects_differ = '{} {} {}'.format(k, v, resource_object['schedule'][k])
            break
    if not objects_differ:
        for layer in state_data['schedule']['schedule_layers']:
            resource_layer = None
            for resource_layer in resource_object['schedule']['schedule_layers']:
                found = False
                if layer['name'] == resource_layer['name']:
                    found = True
                    break
            if not found:
                objects_differ = 'layer {} missing'.format(layer['name'])
                break
            layer['id'] = resource_layer['id']
            for (k, v) in layer.items():
                if k == 'users':
                    continue
                if k == 'start':
                    continue
                if v != resource_layer[k]:
                    objects_differ = 'layer {} key {} {} != {}'.format(layer['name'], k, v, resource_layer[k])
                    break
            if objects_differ:
                break
            if len(layer['users']) != len(resource_layer['users']):
                objects_differ = 'num users in layer {} {} != {}'.format(layer['name'], len(layer['users']), len(resource_layer['users']))
                break
            for user1 in layer['users']:
                found = False
                user2 = None
                for user2 in resource_layer['users']:
                    if user1['member_order'] == user2['member_order'] - 1:
                        found = True
                        break
                if not found:
                    objects_differ = 'layer {} no one with member_order {}'.format(layer['name'], user1['member_order'])
                    break
                if user1['user']['id'] != user2['user']['id']:
                    objects_differ = 'layer {} user at member_order {} {} != {}'.format(layer['name'], user1['member_order'], user1['user']['id'], user2['user']['id'])
                    break
    if objects_differ:
        return state_data
    else:
        return {}