"""
Configure Marathon apps via a salt proxy.

.. code-block:: yaml

    my_app:
      marathon_app.config:
        - config:
            cmd: "while [ true ] ; do echo 'Hello Marathon' ; sleep 5 ; done"
            cpus: 0.1
            mem: 10
            instances: 3

.. versionadded:: 2015.8.2
"""
import copy
import logging
import salt.utils.configcomparer
__proxyenabled__ = ['marathon']
log = logging.getLogger(__file__)

def config(name, config):
    if False:
        while True:
            i = 10
    '\n    Ensure that the marathon app with the given id is present and is configured\n    to match the given config values.\n\n    :param name: The app name/id\n    :param config: The configuration to apply (dict)\n    :return: A standard Salt changes dictionary\n    '
    ret = {'name': name, 'changes': {}, 'result': False, 'comment': ''}
    existing_config = None
    if __salt__['marathon.has_app'](name):
        existing_config = __salt__['marathon.app'](name)['app']
    if existing_config:
        update_config = copy.deepcopy(existing_config)
        salt.utils.configcomparer.compare_and_update_config(config, update_config, ret['changes'])
    else:
        ret['changes']['app'] = {'new': config, 'old': None}
        update_config = config
    if ret['changes']:
        if __opts__['test']:
            ret['result'] = None
            ret['comment'] = 'Marathon app {} is set to be updated'.format(name)
            return ret
        update_result = __salt__['marathon.update_app'](name, update_config)
        if 'exception' in update_result:
            ret['result'] = False
            ret['comment'] = 'Failed to update app config for {}: {}'.format(name, update_result['exception'])
            return ret
        else:
            ret['result'] = True
            ret['comment'] = 'Updated app config for {}'.format(name)
            return ret
    ret['result'] = True
    ret['comment'] = 'Marathon app {} configured correctly'.format(name)
    return ret

def absent(name):
    if False:
        for i in range(10):
            print('nop')
    '\n    Ensure that the marathon app with the given id is not present.\n\n    :param name: The app name/id\n    :return: A standard Salt changes dictionary\n    '
    ret = {'name': name, 'changes': {}, 'result': False, 'comment': ''}
    if not __salt__['marathon.has_app'](name):
        ret['result'] = True
        ret['comment'] = 'App {} already absent'.format(name)
        return ret
    if __opts__['test']:
        ret['result'] = None
        ret['comment'] = 'App {} is set to be removed'.format(name)
        return ret
    if __salt__['marathon.rm_app'](name):
        ret['changes'] = {'app': name}
        ret['result'] = True
        ret['comment'] = 'Removed app {}'.format(name)
        return ret
    else:
        ret['result'] = False
        ret['comment'] = 'Failed to remove app {}'.format(name)
        return ret

def running(name, restart=False, force=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Ensure that the marathon app with the given id is present and restart if set.\n\n    :param name: The app name/id\n    :param restart: Restart the app\n    :param force: Override the current deployment\n    :return: A standard Salt changes dictionary\n    '
    ret = {'name': name, 'changes': {}, 'result': False, 'comment': ''}
    if not __salt__['marathon.has_app'](name):
        ret['result'] = False
        ret['comment'] = 'App {} cannot be restarted because it is absent'.format(name)
        return ret
    if __opts__['test']:
        ret['result'] = None
        qualifier = 'is' if restart else 'is not'
        ret['comment'] = 'App {} {} set to be restarted'.format(name, qualifier)
        return ret
    restart_result = __salt__['marathon.restart_app'](name, restart, force)
    if 'exception' in restart_result:
        ret['result'] = False
        ret['comment'] = 'Failed to restart app {}: {}'.format(name, restart_result['exception'])
        return ret
    else:
        ret['changes'] = restart_result
        ret['result'] = True
        qualifier = 'Restarted' if restart else 'Did not restart'
        ret['comment'] = '{} app {}'.format(qualifier, name)
        return ret