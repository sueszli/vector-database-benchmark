"""
Control Modjk via the Apache Tomcat "Status" worker
(http://tomcat.apache.org/connectors-doc/reference/status.html)

Below is an example of the configuration needed for this module. This
configuration data can be placed either in :ref:`grains
<targeting-grains>` or :ref:`pillar <salt-pillars>`.

If using grains, this can be accomplished :ref:`statically
<static-custom-grains>` or via a :ref:`grain module <writing-grains>`.

If using pillar, the yaml configuration can be placed directly into a pillar
SLS file, making this both the easier and more dynamic method of configuring
this module.

.. code-block:: yaml

    modjk:
      default:
        url: http://localhost/jkstatus
        user: modjk
        pass: secret
        realm: authentication realm for digest passwords
        timeout: 5
      otherVhost:
        url: http://otherVhost/jkstatus
        user: modjk
        pass: secret2
        realm: authentication realm2 for digest passwords
        timeout: 600
"""
import urllib.parse
import urllib.request

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Always load\n    '
    return True

def _auth(url, user, passwd, realm):
    if False:
        print('Hello World!')
    '\n    returns a authentication handler.\n    '
    basic = urllib.request.HTTPBasicAuthHandler()
    basic.add_password(realm=realm, uri=url, user=user, passwd=passwd)
    digest = urllib.request.HTTPDigestAuthHandler()
    digest.add_password(realm=realm, uri=url, user=user, passwd=passwd)
    return urllib.request.build_opener(basic, digest)

def _do_http(opts, profile='default'):
    if False:
        for i in range(10):
            print('nop')
    '\n    Make the http request and return the data\n    '
    ret = {}
    url = __salt__['config.get']('modjk:{}:url'.format(profile), '')
    user = __salt__['config.get']('modjk:{}:user'.format(profile), '')
    passwd = __salt__['config.get']('modjk:{}:pass'.format(profile), '')
    realm = __salt__['config.get']('modjk:{}:realm'.format(profile), '')
    timeout = __salt__['config.get']('modjk:{}:timeout'.format(profile), '')
    if not url:
        raise Exception('missing url in profile {}'.format(profile))
    if user and passwd:
        auth = _auth(url=url, realm=realm, user=user, passwd=passwd)
        urllib.request.install_opener(auth)
    url += '?{}'.format(urllib.parse.urlencode(opts))
    for line in urllib.request.urlopen(url, timeout=timeout).read().splitlines():
        splt = line.split('=', 1)
        if splt[0] in ret:
            ret[splt[0]] += ',{}'.format(splt[1])
        else:
            ret[splt[0]] = splt[1]
    return ret

def _worker_ctl(worker, lbn, vwa, profile='default'):
    if False:
        while True:
            i = 10
    '\n    enable/disable/stop a worker\n    '
    cmd = {'cmd': 'update', 'mime': 'prop', 'w': lbn, 'sw': worker, 'vwa': vwa}
    return _do_http(cmd, profile)['worker.result.type'] == 'OK'

def version(profile='default'):
    if False:
        while True:
            i = 10
    "\n    Return the modjk version\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' modjk.version\n        salt '*' modjk.version other-profile\n    "
    cmd = {'cmd': 'version', 'mime': 'prop'}
    return _do_http(cmd, profile)['worker.jk_version'].split('/')[-1]

def get_running(profile='default'):
    if False:
        i = 10
        return i + 15
    "\n    Get the current running config (not from disk)\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' modjk.get_running\n        salt '*' modjk.get_running other-profile\n    "
    cmd = {'cmd': 'list', 'mime': 'prop'}
    return _do_http(cmd, profile)

def dump_config(profile='default'):
    if False:
        i = 10
        return i + 15
    "\n    Dump the original configuration that was loaded from disk\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' modjk.dump_config\n        salt '*' modjk.dump_config other-profile\n    "
    cmd = {'cmd': 'dump', 'mime': 'prop'}
    return _do_http(cmd, profile)

def list_configured_members(lbn, profile='default'):
    if False:
        while True:
            i = 10
    "\n    Return a list of member workers from the configuration files\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' modjk.list_configured_members loadbalancer1\n        salt '*' modjk.list_configured_members loadbalancer1 other-profile\n    "
    config = dump_config(profile)
    try:
        ret = config['worker.{}.balance_workers'.format(lbn)]
    except KeyError:
        return []
    return [_f for _f in ret.strip().split(',') if _f]

def workers(profile='default'):
    if False:
        i = 10
        return i + 15
    "\n    Return a list of member workers and their status\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' modjk.workers\n        salt '*' modjk.workers other-profile\n    "
    config = get_running(profile)
    lbn = config['worker.list'].split(',')
    worker_list = []
    ret = {}
    for lb in lbn:
        try:
            worker_list.extend(config['worker.{}.balance_workers'.format(lb)].split(','))
        except KeyError:
            pass
    worker_list = list(set(worker_list))
    for worker in worker_list:
        ret[worker] = {'activation': config['worker.{}.activation'.format(worker)], 'state': config['worker.{}.state'.format(worker)]}
    return ret

def recover_all(lbn, profile='default'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Set the all the workers in lbn to recover and activate them if they are not\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' modjk.recover_all loadbalancer1\n        salt '*' modjk.recover_all loadbalancer1 other-profile\n    "
    ret = {}
    config = get_running(profile)
    try:
        workers_ = config['worker.{}.balance_workers'.format(lbn)].split(',')
    except KeyError:
        return ret
    for worker in workers_:
        curr_state = worker_status(worker, profile)
        if curr_state['activation'] != 'ACT':
            worker_activate(worker, lbn, profile)
        if not curr_state['state'].startswith('OK'):
            worker_recover(worker, lbn, profile)
        ret[worker] = worker_status(worker, profile)
    return ret

def reset_stats(lbn, profile='default'):
    if False:
        while True:
            i = 10
    "\n    Reset all runtime statistics for the load balancer\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' modjk.reset_stats loadbalancer1\n        salt '*' modjk.reset_stats loadbalancer1 other-profile\n    "
    cmd = {'cmd': 'reset', 'mime': 'prop', 'w': lbn}
    return _do_http(cmd, profile)['worker.result.type'] == 'OK'

def lb_edit(lbn, settings, profile='default'):
    if False:
        for i in range(10):
            print('nop')
    '\n    Edit the loadbalancer settings\n\n    Note: http://tomcat.apache.org/connectors-doc/reference/status.html\n    Data Parameters for the standard Update Action\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt \'*\' modjk.lb_edit loadbalancer1 "{\'vlr\': 1, \'vlt\': 60}"\n        salt \'*\' modjk.lb_edit loadbalancer1 "{\'vlr\': 1, \'vlt\': 60}" other-profile\n    '
    settings['cmd'] = 'update'
    settings['mime'] = 'prop'
    settings['w'] = lbn
    return _do_http(settings, profile)['worker.result.type'] == 'OK'

def bulk_stop(workers, lbn, profile='default'):
    if False:
        i = 10
        return i + 15
    '\n    Stop all the given workers in the specific load balancer\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt \'*\' modjk.bulk_stop node1,node2,node3 loadbalancer1\n        salt \'*\' modjk.bulk_stop node1,node2,node3 loadbalancer1 other-profile\n\n        salt \'*\' modjk.bulk_stop ["node1","node2","node3"] loadbalancer1\n        salt \'*\' modjk.bulk_stop ["node1","node2","node3"] loadbalancer1 other-profile\n    '
    ret = {}
    if isinstance(workers, str):
        workers = workers.split(',')
    for worker in workers:
        try:
            ret[worker] = worker_stop(worker, lbn, profile)
        except Exception:
            ret[worker] = False
    return ret

def bulk_activate(workers, lbn, profile='default'):
    if False:
        for i in range(10):
            print('nop')
    '\n    Activate all the given workers in the specific load balancer\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt \'*\' modjk.bulk_activate node1,node2,node3 loadbalancer1\n        salt \'*\' modjk.bulk_activate node1,node2,node3 loadbalancer1 other-profile\n\n        salt \'*\' modjk.bulk_activate ["node1","node2","node3"] loadbalancer1\n        salt \'*\' modjk.bulk_activate ["node1","node2","node3"] loadbalancer1 other-profile\n    '
    ret = {}
    if isinstance(workers, str):
        workers = workers.split(',')
    for worker in workers:
        try:
            ret[worker] = worker_activate(worker, lbn, profile)
        except Exception:
            ret[worker] = False
    return ret

def bulk_disable(workers, lbn, profile='default'):
    if False:
        return 10
    '\n    Disable all the given workers in the specific load balancer\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt \'*\' modjk.bulk_disable node1,node2,node3 loadbalancer1\n        salt \'*\' modjk.bulk_disable node1,node2,node3 loadbalancer1 other-profile\n\n        salt \'*\' modjk.bulk_disable ["node1","node2","node3"] loadbalancer1\n        salt \'*\' modjk.bulk_disable ["node1","node2","node3"] loadbalancer1 other-profile\n    '
    ret = {}
    if isinstance(workers, str):
        workers = workers.split(',')
    for worker in workers:
        try:
            ret[worker] = worker_disable(worker, lbn, profile)
        except Exception:
            ret[worker] = False
    return ret

def bulk_recover(workers, lbn, profile='default'):
    if False:
        while True:
            i = 10
    '\n    Recover all the given workers in the specific load balancer\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt \'*\' modjk.bulk_recover node1,node2,node3 loadbalancer1\n        salt \'*\' modjk.bulk_recover node1,node2,node3 loadbalancer1 other-profile\n\n        salt \'*\' modjk.bulk_recover ["node1","node2","node3"] loadbalancer1\n        salt \'*\' modjk.bulk_recover ["node1","node2","node3"] loadbalancer1 other-profile\n    '
    ret = {}
    if isinstance(workers, str):
        workers = workers.split(',')
    for worker in workers:
        try:
            ret[worker] = worker_recover(worker, lbn, profile)
        except Exception:
            ret[worker] = False
    return ret

def worker_status(worker, profile='default'):
    if False:
        return 10
    "\n    Return the state of the worker\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' modjk.worker_status node1\n        salt '*' modjk.worker_status node1 other-profile\n    "
    config = get_running(profile)
    try:
        return {'activation': config['worker.{}.activation'.format(worker)], 'state': config['worker.{}.state'.format(worker)]}
    except KeyError:
        return False

def worker_recover(worker, lbn, profile='default'):
    if False:
        while True:
            i = 10
    "\n    Set the worker to recover\n    this module will fail if it is in OK state\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' modjk.worker_recover node1 loadbalancer1\n        salt '*' modjk.worker_recover node1 loadbalancer1 other-profile\n    "
    cmd = {'cmd': 'recover', 'mime': 'prop', 'w': lbn, 'sw': worker}
    return _do_http(cmd, profile)

def worker_disable(worker, lbn, profile='default'):
    if False:
        print('Hello World!')
    "\n    Set the worker to disable state in the lbn load balancer\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' modjk.worker_disable node1 loadbalancer1\n        salt '*' modjk.worker_disable node1 loadbalancer1 other-profile\n    "
    return _worker_ctl(worker, lbn, 'd', profile)

def worker_activate(worker, lbn, profile='default'):
    if False:
        while True:
            i = 10
    "\n    Set the worker to activate state in the lbn load balancer\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' modjk.worker_activate node1 loadbalancer1\n        salt '*' modjk.worker_activate node1 loadbalancer1 other-profile\n    "
    return _worker_ctl(worker, lbn, 'a', profile)

def worker_stop(worker, lbn, profile='default'):
    if False:
        print('Hello World!')
    "\n    Set the worker to stopped state in the lbn load balancer\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt '*' modjk.worker_activate node1 loadbalancer1\n        salt '*' modjk.worker_activate node1 loadbalancer1 other-profile\n    "
    return _worker_ctl(worker, lbn, 's', profile)

def worker_edit(worker, lbn, settings, profile='default'):
    if False:
        while True:
            i = 10
    '\n    Edit the worker settings\n\n    Note: http://tomcat.apache.org/connectors-doc/reference/status.html\n    Data Parameters for the standard Update Action\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt \'*\' modjk.worker_edit node1 loadbalancer1 "{\'vwf\': 500, \'vwd\': 60}"\n        salt \'*\' modjk.worker_edit node1 loadbalancer1 "{\'vwf\': 500, \'vwd\': 60}" other-profile\n    '
    settings['cmd'] = 'update'
    settings['mime'] = 'prop'
    settings['w'] = lbn
    settings['sw'] = worker
    return _do_http(settings, profile)['worker.result.type'] == 'OK'