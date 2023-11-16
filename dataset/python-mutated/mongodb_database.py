"""
Management of MongoDB Databases
===============================

:depends:   - pymongo Python module

Only deletion is supported, creation doesn't make sense and can be done using
:py:func:`mongodb_user.present <salt.states.mongodb_user.present>`.
"""
__virtualname__ = 'mongodb_database'

def __virtual__():
    if False:
        print('Hello World!')
    if 'mongodb.db_exists' in __salt__:
        return __virtualname__
    return (False, 'mongodb module could not be loaded')

def absent(name, user=None, password=None, host=None, port=None, authdb=None):
    if False:
        print('Hello World!')
    "\n    Ensure that the named database is absent. Note that creation doesn't make\n    sense in MongoDB.\n\n    name\n        The name of the database to remove\n\n    user\n        The user to connect as (must be able to create the user)\n\n    password\n        The password of the user\n\n    host\n        The host to connect to\n\n    port\n        The port to connect to\n\n    authdb\n        The database in which to authenticate\n    "
    ret = {'name': name, 'changes': {}, 'result': True, 'comment': ''}
    if __salt__['mongodb.db_exists'](name, user, password, host, port, authdb=authdb):
        if __opts__['test']:
            ret['result'] = None
            ret['comment'] = 'Database {} is present and needs to be removed'.format(name)
            return ret
        if __salt__['mongodb.db_remove'](name, user, password, host, port, authdb=authdb):
            ret['comment'] = 'Database {} has been removed'.format(name)
            ret['changes'][name] = 'Absent'
            return ret
    ret['comment'] = 'Database {} is not present'.format(name)
    return ret