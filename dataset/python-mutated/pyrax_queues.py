"""
Manage Rackspace Queues
=======================

.. versionadded:: 2015.5.0

Create and destroy Rackspace queues. Be aware that this interacts with
Rackspace's services, and so may incur charges.

This module uses ``pyrax``, which can be installed via package, or pip.
This module is greatly inspired by boto_* modules from SaltStack code source.

.. code-block:: yaml

    myqueue:
        pyrax_queues.present:
            - provider: my-pyrax

    myqueue:
        pyrax_queues.absent:
            - provider: my-pyrax
"""
import salt.utils.openstack.pyrax as suop

def __virtual__():
    if False:
        return 10
    '\n    Only load if pyrax is available.\n    '
    return suop.HAS_PYRAX

def present(name, provider):
    if False:
        return 10
    '\n    Ensure the RackSpace queue exists.\n\n    name\n        Name of the Rackspace queue.\n\n    provider\n        Salt Cloud Provider\n    '
    ret = {'name': name, 'result': True, 'comment': '', 'changes': {}}
    is_present = next(iter(__salt__['cloud.action']('queues_exists', provider=provider, name=name)[provider].values()))
    if not is_present:
        if __opts__['test']:
            msg = 'Rackspace queue {} is set to be created.'.format(name)
            ret['comment'] = msg
            ret['result'] = None
            return ret
        created = __salt__['cloud.action']('queues_create', provider=provider, name=name)
        if created:
            queue = __salt__['cloud.action']('queues_show', provider=provider, name=name)
            ret['changes']['old'] = {}
            ret['changes']['new'] = {'queue': queue}
        else:
            ret['result'] = False
            ret['comment'] = 'Failed to create {} Rackspace queue.'.format(name)
            return ret
    else:
        ret['comment'] = '{} present.'.format(name)
    return ret

def absent(name, provider):
    if False:
        print('Hello World!')
    '\n    Ensure the named Rackspace queue is deleted.\n\n    name\n        Name of the Rackspace queue.\n\n    provider\n        Salt Cloud provider\n    '
    ret = {'name': name, 'result': True, 'comment': '', 'changes': {}}
    is_present = next(iter(__salt__['cloud.action']('queues_exists', provider=provider, name=name)[provider].values()))
    if is_present:
        if __opts__['test']:
            ret['comment'] = 'Rackspace queue {} is set to be removed.'.format(name)
            ret['result'] = None
            return ret
        queue = __salt__['cloud.action']('queues_show', provider=provider, name=name)
        deleted = __salt__['cloud.action']('queues_delete', provider=provider, name=name)
        if deleted:
            ret['changes']['old'] = queue
            ret['changes']['new'] = {}
        else:
            ret['result'] = False
            ret['comment'] = 'Failed to delete {} Rackspace queue.'.format(name)
    else:
        ret['comment'] = '{} does not exist.'.format(name)
    return ret