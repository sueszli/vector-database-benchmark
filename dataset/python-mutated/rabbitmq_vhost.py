"""
Manage RabbitMQ Virtual Hosts
=============================

Example:

.. code-block:: yaml

    virtual_host:
      rabbitmq_vhost.present:
        - user: rabbit_user
        - conf: .*
        - write: .*
        - read: .*
"""
import logging
import salt.utils.path
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    Only load if RabbitMQ is installed.\n    '
    if salt.utils.path.which('rabbitmqctl'):
        return True
    return (False, 'Command not found: rabbitmqctl')

def present(name):
    if False:
        i = 10
        return i + 15
    '\n    Ensure the RabbitMQ VHost exists.\n\n    name\n        VHost name\n\n    user\n        Initial user permission to set on the VHost, if present\n\n        .. deprecated:: 2015.8.0\n    owner\n        Initial owner permission to set on the VHost, if present\n\n        .. deprecated:: 2015.8.0\n    conf\n        Initial conf string to apply to the VHost and user. Defaults to .*\n\n        .. deprecated:: 2015.8.0\n    write\n        Initial write permissions to apply to the VHost and user.\n        Defaults to .*\n\n        .. deprecated:: 2015.8.0\n    read\n        Initial read permissions to apply to the VHost and user.\n        Defaults to .*\n\n        .. deprecated:: 2015.8.0\n    runas\n        Name of the user to run the command\n\n        .. deprecated:: 2015.8.0\n    '
    ret = {'name': name, 'result': True, 'comment': '', 'changes': {}}
    vhost_exists = __salt__['rabbitmq.vhost_exists'](name)
    if vhost_exists:
        ret['comment'] = "Virtual Host '{}' already exists.".format(name)
        return ret
    if not __opts__['test']:
        result = __salt__['rabbitmq.add_vhost'](name)
        if 'Error' in result:
            ret['result'] = False
            ret['comment'] = result['Error']
            return ret
        elif 'Added' in result:
            ret['comment'] = result['Added']
    ret['changes'] = {'old': '', 'new': name}
    if __opts__['test']:
        ret['result'] = None
        ret['comment'] = "Virtual Host '{}' will be created.".format(name)
    return ret

def absent(name):
    if False:
        i = 10
        return i + 15
    '\n    Ensure the RabbitMQ Virtual Host is absent\n\n    name\n        Name of the Virtual Host to remove\n    runas\n        User to run the command\n\n        .. deprecated:: 2015.8.0\n    '
    ret = {'name': name, 'result': True, 'comment': '', 'changes': {}}
    vhost_exists = __salt__['rabbitmq.vhost_exists'](name)
    if not vhost_exists:
        ret['comment'] = "Virtual Host '{}' is not present.".format(name)
        return ret
    if not __opts__['test']:
        result = __salt__['rabbitmq.delete_vhost'](name)
        if 'Error' in result:
            ret['result'] = False
            ret['comment'] = result['Error']
            return ret
        elif 'Deleted' in result:
            ret['comment'] = result['Deleted']
    ret['changes'] = {'new': '', 'old': name}
    if __opts__['test']:
        ret['result'] = None
        ret['comment'] = "Virtual Host '{}' will be removed.".format(name)
    return ret