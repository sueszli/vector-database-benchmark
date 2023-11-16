"""
Manage PagerDuty services

Escalation policies can be referenced by pagerduty ID or by namea.

For example:

.. code-block:: yaml

    ensure test service
        pagerduty_service.present:
            - name: 'my service'
            - escalation_policy_id: 'my escalation policy'
            - type: nagios

"""

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    Only load if the pygerduty module is available in __salt__\n    '
    if 'pagerduty_util.get_resource' in __salt__:
        return 'pagerduty_service'
    return (False, 'pagerduty_util module could not be loaded')

def present(profile='pagerduty', subdomain=None, api_key=None, **kwargs):
    if False:
        while True:
            i = 10
    '\n    Ensure pagerduty service exists.\n    This method accepts as arguments everything defined in\n    https://developer.pagerduty.com/documentation/rest/services/create\n\n    Note that many arguments are mutually exclusive, depending on the "type" argument.\n\n    Examples:\n\n    .. code-block:: yaml\n\n        # create a PagerDuty email service at test-email@DOMAIN.pagerduty.com\n        ensure generic email service exists:\n            pagerduty_service.present:\n                - name: my email service\n                - service:\n                    description: "email service controlled by salt"\n                    escalation_policy_id: "my escalation policy"\n                    type: "generic_email"\n                    service_key: "test-email"\n\n    .. code-block:: yaml\n\n        # create a pagerduty service using cloudwatch integration\n        ensure my cloudwatch service exists:\n            pagerduty_service.present:\n                - name: my cloudwatch service\n                - service:\n                    escalation_policy_id: "my escalation policy"\n                    type: aws_cloudwatch\n                    description: "my cloudwatch service controlled by salt"\n\n    '
    kwargs['service']['name'] = kwargs['name']
    escalation_policy_id = kwargs['service']['escalation_policy_id']
    escalation_policy = __salt__['pagerduty_util.get_resource']('escalation_policies', escalation_policy_id, ['name', 'id'], profile=profile, subdomain=subdomain, api_key=api_key)
    if escalation_policy:
        kwargs['service']['escalation_policy_id'] = escalation_policy['id']
    r = __salt__['pagerduty_util.resource_present']('services', ['name', 'id'], _diff, profile, subdomain, api_key, **kwargs)
    return r

def absent(profile='pagerduty', subdomain=None, api_key=None, **kwargs):
    if False:
        print('Hello World!')
    '\n    Ensure a pagerduty service does not exist.\n    Name can be the service name or pagerduty service id.\n    '
    r = __salt__['pagerduty_util.resource_absent']('services', ['name', 'id'], profile, subdomain, api_key, **kwargs)
    return r

def _diff(state_data, resource_object):
    if False:
        return 10
    'helper method to compare salt state info with the PagerDuty API json structure,\n    and determine if we need to update.\n\n    returns the dict to pass to the PD API to perform the update, or empty dict if no update.\n    '
    objects_differ = None
    for (k, v) in state_data['service'].items():
        if k == 'escalation_policy_id':
            resource_value = resource_object['escalation_policy']['id']
        elif k == 'service_key':
            resource_value = resource_object['service_key']
            if '@' in resource_value:
                resource_value = resource_value[0:resource_value.find('@')]
        else:
            resource_value = resource_object[k]
        if v != resource_value:
            objects_differ = '{} {} {}'.format(k, v, resource_value)
            break
    if objects_differ:
        return state_data
    else:
        return {}