"""
Module providing a simple management interface to a chronos cluster.

Currently this only works when run through a proxy minion.

.. versionadded:: 2015.8.2
"""
import logging
import salt.utils.http
import salt.utils.json
import salt.utils.platform
from salt.exceptions import get_error_message
__proxyenabled__ = ['chronos']
log = logging.getLogger(__file__)

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    return salt.utils.platform.is_proxy() and 'proxy' in __opts__

def _base_url():
    if False:
        return 10
    '\n    Return the proxy configured base url.\n    '
    base_url = 'http://locahost:4400'
    if 'proxy' in __opts__:
        base_url = __opts__['proxy'].get('base_url', base_url)
    return base_url

def _jobs():
    if False:
        print('Hello World!')
    '\n    Return the currently configured jobs.\n    '
    response = salt.utils.http.query('{}/scheduler/jobs'.format(_base_url()), decode_type='json', decode=True)
    jobs = {}
    for job in response['dict']:
        jobs[job.pop('name')] = job
    return jobs

def jobs():
    if False:
        while True:
            i = 10
    '\n    Return a list of the currently installed job names.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt chronos-minion-id chronos.jobs\n    '
    job_names = _jobs().keys()
    job_names.sort()
    return {'jobs': job_names}

def has_job(name):
    if False:
        return 10
    '\n    Return whether the given job is currently configured.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt chronos-minion-id chronos.has_job my-job\n    '
    return name in _jobs()

def job(name):
    if False:
        while True:
            i = 10
    '\n    Return the current server configuration for the specified job.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt chronos-minion-id chronos.job my-job\n    '
    jobs = _jobs()
    if name in jobs:
        return {'job': jobs[name]}
    return None

def update_job(name, config):
    if False:
        i = 10
        return i + 15
    "\n    Update the specified job with the given configuration.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt chronos-minion-id chronos.update_job my-job '<config yaml>'\n    "
    if 'name' not in config:
        config['name'] = name
    data = salt.utils.json.dumps(config)
    try:
        response = salt.utils.http.query('{}/scheduler/iso8601'.format(_base_url()), method='POST', data=data, header_dict={'Content-Type': 'application/json'})
        log.debug('update response: %s', response)
        return {'success': True}
    except Exception as ex:
        log.error('unable to update chronos job: %s', get_error_message(ex))
        return {'exception': {'message': get_error_message(ex)}}

def rm_job(name):
    if False:
        while True:
            i = 10
    '\n    Remove the specified job from the server.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt chronos-minion-id chronos.rm_job my-job\n    '
    response = salt.utils.http.query('{}/scheduler/job/{}'.format(_base_url(), name), method='DELETE')
    return True