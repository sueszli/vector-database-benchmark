"""
Manage Telemetry alert configurations
=====================================

.. versionadded:: 2016.3.0

Create, Update and destroy Mongo Telemetry alert configurations.

This module uses requests, which can be installed via package, or pip.

This module accepts explicit credential (telemetry api key)
or can also read api key credentials from a pillar.
Example:

.. code-block:: yaml

    ensure telemetry alert X is defined on deployment Y:
        telemetry_alert.present:
            - deployment_id: "rs-XXXXXX"
            - metric_name: "testMetric"
            - alert_config:
               max: 1
               filter:  SERVER_ROLE_MONGOD_PRIMARY
               escalate_to: "example@pagerduty.com"
            - name: "**MANAGED BY ORCA DO NOT EDIT BY HAND** manages alarm on testMetric"
"""

def __virtual__():
    if False:
        print('Hello World!')
    if 'telemetry.get_alert_config' in __salt__:
        return 'telemetry_alert'
    return (False, 'telemetry module could not be loaded')

def present(name, deployment_id, metric_name, alert_config, api_key=None, profile='telemetry'):
    if False:
        i = 10
        return i + 15
    '\n    Ensure the telemetry alert exists.\n\n    name\n        An optional description of the alarm (not currently supported by telemetry API)\n\n    deployment_id\n        Specifies the ID of the root deployment resource\n        (replica set cluster or sharded cluster) to which this alert definition is attached\n\n    metric_name\n        Specifies the unique ID of the metric to whose values these thresholds will be applied\n\n    alert_config: Is a list of dictionaries where each dict contains the following fields:\n        filter\n            By default the alert will apply to the deployment and all its constituent resources.\n            If the alert only applies to a subset of those resources, a filter may be specified to narrow this scope.\n\n        min\n            the smallest "ok" value the metric may take on; if missing or null, no minimum is enforced.\n\n        max\n            the largest "ok" value the metric may take on; if missing or null, no maximum is enforced.\n\n        notify_all\n            Used to indicate if you want to alert both onCallEngineer and apiNotifications\n\n    api_key\n        Telemetry api key for the user\n\n    profile\n        A dict of telemetry config information.  If present, will be used instead of\n        api_key.\n\n    '
    ret = {'name': metric_name, 'result': True, 'comment': '', 'changes': {}}
    saved_alert_config = __salt__['telemetry.get_alert_config'](deployment_id, metric_name, api_key, profile)
    post_body = {'deployment': deployment_id, 'filter': alert_config.get('filter'), 'notificationChannel': __salt__['telemetry.get_notification_channel_id'](alert_config.get('escalate_to')).split(), 'condition': {'metric': metric_name, 'max': alert_config.get('max'), 'min': alert_config.get('min')}}
    difference = []
    if saved_alert_config:
        for (k, v) in post_body.items():
            if k not in saved_alert_config:
                difference.append('{}={} (new)'.format(k, v))
                continue
            v2 = saved_alert_config[k]
            if v == v2:
                continue
            if isinstance(v, str) and v == str(v2):
                continue
            if isinstance(v, float) and v == float(v2):
                continue
            if isinstance(v, int) and v == int(v2):
                continue
            difference.append("{}='{}' was: '{}'".format(k, v, v2))
    else:
        difference.append('new alert config')
    create_or_update_args = (deployment_id, metric_name, alert_config, api_key, profile)
    if saved_alert_config:
        if len(difference) == 0:
            ret['comment'] = 'alert config {} present and matching'.format(metric_name)
            return ret
        if __opts__['test']:
            msg = 'alert config {} is to be updated.'.format(metric_name)
            ret['comment'] = msg
            ret['result'] = '\n'.join(difference)
            return ret
        (result, msg) = __salt__['telemetry.update_alarm'](*create_or_update_args)
        if result:
            ret['changes']['diff'] = difference
            ret['comment'] = 'Alert updated.'
        else:
            ret['result'] = False
            ret['comment'] = 'Failed to update {} alert config: {}'.format(metric_name, msg)
    else:
        if __opts__['test']:
            msg = 'alert config {} is to be created.'.format(metric_name)
            ret['comment'] = msg
            ret['result'] = None
            return ret
        (result, msg) = __salt__['telemetry.create_alarm'](*create_or_update_args)
        if result:
            ret['changes']['new'] = msg
        else:
            ret['result'] = False
            ret['comment'] = 'Failed to create {} alert config: {}'.format(metric_name, msg)
    return ret

def absent(name, deployment_id, metric_name, api_key=None, profile='telemetry'):
    if False:
        return 10
    '\n    Ensure the telemetry alert config is deleted\n\n    name\n        An optional description of the alarms (not currently supported by telemetry API)\n\n    deployment_id\n        Specifies the ID of the root deployment resource\n        (replica set cluster or sharded cluster) to which this alert definition is attached\n\n    metric_name\n        Specifies the unique ID of the metric to whose values these thresholds will be applied\n\n    api_key\n        Telemetry api key for the user\n\n    profile\n        A dict with telemetry config data. If present, will be used instead of\n        api_key.\n    '
    ret = {'name': metric_name, 'result': True, 'comment': '', 'changes': {}}
    is_present = __salt__['telemetry.get_alert_config'](deployment_id, metric_name, api_key, profile)
    if is_present:
        alert_id = is_present.get('_id')
        if __opts__['test']:
            ret['comment'] = 'alert {} is set to be removed from deployment: {}.'.format(metric_name, deployment_id)
            ret['result'] = None
            return ret
        (deleted, msg) = __salt__['telemetry.delete_alarms'](deployment_id, alert_id, is_present.get('condition', {}).get('metric'), api_key, profile)
        if deleted:
            ret['changes']['old'] = metric_name
            ret['changes']['new'] = None
        else:
            ret['result'] = False
            ret['comment'] = 'Failed to delete alert {} from deployment: {}'.format(metric_name, msg)
    else:
        ret['comment'] = 'alarm on {} does not exist within {}.'.format(metric_name, deployment_id)
    return ret