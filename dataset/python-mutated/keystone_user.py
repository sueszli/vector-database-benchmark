"""
Management of OpenStack Keystone Users
======================================

.. versionadded:: 2018.3.0

:depends: shade
:configuration: see :py:mod:`salt.modules.keystoneng` for setup instructions

Example States

.. code-block:: yaml

    create user:
      keystone_user.present:
        - name: user1

    delete user:
      keystone_user.absent:
        - name: user1

    create user with optional params:
      keystone_user.present:
        - name: user1
        - domain: domain1
        - enabled: False
        - password: password123
        - email: "user1@example.org"
        - description: 'my user'
"""
__virtualname__ = 'keystone_user'

def __virtual__():
    if False:
        i = 10
        return i + 15
    if 'keystoneng.user_get' in __salt__:
        return __virtualname__
    return (False, 'The keystoneng execution module failed to load: shade python module is not available')

def _common(kwargs):
    if False:
        print('Hello World!')
    "\n    Returns: None if user wasn't found, otherwise a user object\n    "
    search_kwargs = {'name': kwargs['name']}
    if 'domain' in kwargs:
        domain = __salt__['keystoneng.get_entity']('domain', name=kwargs.pop('domain'))
        domain_id = domain.id if hasattr(domain, 'id') else domain
        search_kwargs['domain_id'] = domain_id
        kwargs['domain_id'] = domain_id
    return __salt__['keystoneng.user_get'](**search_kwargs)

def present(name, auth=None, **kwargs):
    if False:
        while True:
            i = 10
    '\n    Ensure domain exists and is up-to-date\n\n    name\n        Name of the domain\n\n    domain\n        The name or id of the domain\n\n    enabled\n        Boolean to control if domain is enabled\n\n    description\n        An arbitrary description of the domain\n\n    password\n        The user password\n\n    email\n        The users email address\n    '
    ret = {'name': name, 'changes': {}, 'result': True, 'comment': ''}
    kwargs = __utils__['args.clean_kwargs'](**kwargs)
    __salt__['keystoneng.setup_clouds'](auth)
    kwargs['name'] = name
    user = _common(kwargs)
    if user is None:
        if __opts__['test'] is True:
            ret['result'] = None
            ret['changes'] = kwargs
            ret['comment'] = 'User will be created.'
            return ret
        user = __salt__['keystoneng.user_create'](**kwargs)
        ret['changes'] = user
        ret['comment'] = 'Created user'
        return ret
    changes = __salt__['keystoneng.compare_changes'](user, **kwargs)
    if changes:
        if __opts__['test'] is True:
            ret['result'] = None
            ret['changes'] = changes
            ret['comment'] = 'User will be updated.'
            return ret
        kwargs['name'] = user
        __salt__['keystoneng.user_update'](**kwargs)
        ret['changes'].update(changes)
        ret['comment'] = 'Updated user'
    return ret

def absent(name, auth=None, **kwargs):
    if False:
        return 10
    '\n    Ensure user does not exists\n\n    name\n        Name of the user\n\n    domain\n        The name or id of the domain\n    '
    ret = {'name': name, 'changes': {}, 'result': True, 'comment': ''}
    kwargs = __utils__['args.clean_kwargs'](**kwargs)
    __salt__['keystoneng.setup_clouds'](auth)
    kwargs['name'] = name
    user = _common(kwargs)
    if user:
        if __opts__['test'] is True:
            ret['result'] = None
            ret['changes'] = {'id': user.id}
            ret['comment'] = 'User will be deleted.'
            return ret
        __salt__['keystoneng.user_delete'](name=user)
        ret['changes']['id'] = user.id
        ret['comment'] = 'Deleted user'
    return ret