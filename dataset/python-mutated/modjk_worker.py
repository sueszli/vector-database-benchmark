"""
Manage modjk workers
====================

Send commands to a :strong:`modjk` load balancer via the peer system.

This module can be used with the :ref:`prereq <requisites-prereq>`
requisite to remove/add the worker from the load balancer before
deploying/restarting service.

Mandatory Settings:

- The minion needs to have permission to publish the :strong:`modjk.*`
  functions (see :ref:`here <peer>` for information on configuring
  peer publishing permissions)

- The modjk load balancer must be configured as stated in the :strong:`modjk`
  execution module :mod:`documentation <salt.modules.modjk>`
"""

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    Check if we have peer access ?\n    '
    return True

def _send_command(cmd, worker, lbn, target, profile='default', tgt_type='glob'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Send a command to the modjk loadbalancer\n    The minion need to be able to publish the commands to the load balancer\n\n    cmd:\n        worker_stop - won't get any traffic from the lbn\n        worker_activate - activate the worker\n        worker_disable - will get traffic only for current sessions\n    "
    ret = {'code': False, 'msg': 'OK', 'minions': []}
    func = 'modjk.{}'.format(cmd)
    args = [worker, lbn, profile]
    response = __salt__['publish.publish'](target, func, args, tgt_type)
    errors = []
    minions = []
    for minion in response:
        minions.append(minion)
        if not response[minion]:
            errors.append(minion)
    if not response:
        ret['msg'] = 'no servers answered the published command {}'.format(cmd)
        return ret
    elif len(errors) > 0:
        ret['msg'] = 'the following minions return False'
        ret['minions'] = errors
        return ret
    else:
        ret['code'] = True
        ret['msg'] = 'the commad was published successfully'
        ret['minions'] = minions
        return ret

def _worker_status(target, worker, activation, profile='default', tgt_type='glob'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Check if the worker is in `activation` state in the targeted load balancers\n\n    The function will return the following dictionary:\n        result - False if no server returned from the published command\n        errors - list of servers that couldn't find the worker\n        wrong_state - list of servers that the worker was in the wrong state\n                      (not activation)\n    "
    ret = {'result': True, 'errors': [], 'wrong_state': []}
    args = [worker, profile]
    status = __salt__['publish.publish'](target, 'modjk.worker_status', args, tgt_type)
    if not status:
        ret['result'] = False
        return ret
    for balancer in status:
        if not status[balancer]:
            ret['errors'].append(balancer)
        elif status[balancer]['activation'] != activation:
            ret['wrong_state'].append(balancer)
    return ret

def _talk2modjk(name, lbn, target, action, profile='default', tgt_type='glob'):
    if False:
        i = 10
        return i + 15
    '\n    Wrapper function for the stop/disable/activate functions\n    '
    ret = {'name': name, 'result': True, 'changes': {}, 'comment': ''}
    action_map = {'worker_stop': 'STP', 'worker_disable': 'DIS', 'worker_activate': 'ACT'}
    status = _worker_status(target, name, action_map[action], profile, tgt_type)
    if not status['result']:
        ret['result'] = False
        ret['comment'] = 'no servers answered the published command modjk.worker_status'
        return ret
    if status['errors']:
        ret['result'] = False
        ret['comment'] = 'the following balancers could not find the worker {}: {}'.format(name, status['errors'])
        return ret
    if not status['wrong_state']:
        ret['comment'] = 'the worker is in the desired activation state on all the balancers'
        return ret
    else:
        ret['comment'] = 'the action {} will be sent to the balancers {}'.format(action, status['wrong_state'])
        ret['changes'] = {action: status['wrong_state']}
    if __opts__['test']:
        ret['result'] = None
        return ret
    response = _send_command(action, name, lbn, target, profile, tgt_type)
    ret['comment'] = response['msg']
    ret['result'] = response['code']
    return ret

def stop(name, lbn, target, profile='default', tgt_type='glob'):
    if False:
        print('Hello World!')
    "\n    .. versionchanged:: 2017.7.0\n        The ``expr_form`` argument has been renamed to ``tgt_type``, earlier\n        releases must use ``expr_form``.\n\n    Stop the named worker from the lbn load balancers at the targeted minions\n    The worker won't get any traffic from the lbn\n\n    Example:\n\n    .. code-block:: yaml\n\n        disable-before-deploy:\n          modjk_worker.stop:\n            - name: {{ grains['id'] }}\n            - lbn: application\n            - target: 'roles:balancer'\n            - tgt_type: grain\n    "
    return _talk2modjk(name, lbn, target, 'worker_stop', profile, tgt_type)

def activate(name, lbn, target, profile='default', tgt_type='glob'):
    if False:
        i = 10
        return i + 15
    "\n    .. versionchanged:: 2017.7.0\n        The ``expr_form`` argument has been renamed to ``tgt_type``, earlier\n        releases must use ``expr_form``.\n\n    Activate the named worker from the lbn load balancers at the targeted\n    minions\n\n    Example:\n\n    .. code-block:: yaml\n\n        disable-before-deploy:\n          modjk_worker.activate:\n            - name: {{ grains['id'] }}\n            - lbn: application\n            - target: 'roles:balancer'\n            - tgt_type: grain\n    "
    return _talk2modjk(name, lbn, target, 'worker_activate', profile, tgt_type)

def disable(name, lbn, target, profile='default', tgt_type='glob'):
    if False:
        print('Hello World!')
    "\n    .. versionchanged:: 2017.7.0\n        The ``expr_form`` argument has been renamed to ``tgt_type``, earlier\n        releases must use ``expr_form``.\n\n    Disable the named worker from the lbn load balancers at the targeted\n    minions. The worker will get traffic only for current sessions and won't\n    get new ones.\n\n    Example:\n\n    .. code-block:: yaml\n\n        disable-before-deploy:\n          modjk_worker.disable:\n            - name: {{ grains['id'] }}\n            - lbn: application\n            - target: 'roles:balancer'\n            - tgt_type: grain\n    "
    return _talk2modjk(name, lbn, target, 'worker_disable', profile, tgt_type)