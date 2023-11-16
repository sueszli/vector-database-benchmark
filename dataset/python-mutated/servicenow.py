"""
Module for execution of ServiceNow CI (configuration items)

.. versionadded:: 2016.11.0

:depends: servicenow_rest python module

:configuration: Configure this module by specifying the name of a configuration
    profile in the minion config, minion pillar, or master config. The module
    will use the 'servicenow' key by default, if defined.

    For example:

    .. code-block:: yaml

        servicenow:
          instance_name: ''
          username: ''
          password: ''
"""
import logging
HAS_LIBS = False
try:
    from servicenow_rest.api import Client
    HAS_LIBS = True
except ImportError:
    pass
log = logging.getLogger(__name__)
__virtualname__ = 'servicenow'
SERVICE_NAME = 'servicenow'

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Only load this module if servicenow is installed on this minion.\n    '
    if HAS_LIBS:
        return __virtualname__
    return (False, 'The servicenow execution module failed to load: requires servicenow_rest python library to be installed.')

def _get_client():
    if False:
        while True:
            i = 10
    config = __salt__['config.option'](SERVICE_NAME)
    instance_name = config['instance_name']
    username = config['username']
    password = config['password']
    return Client(instance_name, username, password)

def set_change_request_state(change_id, state='approved'):
    if False:
        print('Hello World!')
    '\n    Set the approval state of a change request/record\n\n    :param change_id: The ID of the change request, e.g. CHG123545\n    :type  change_id: ``str``\n\n    :param state: The target state, e.g. approved\n    :type  state: ``str``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion servicenow.set_change_request_state CHG000123 declined\n        salt myminion servicenow.set_change_request_state CHG000123 approved\n    '
    client = _get_client()
    client.table = 'change_request'
    record = client.get({'number': change_id})
    if not record:
        log.error('Failed to fetch change record, maybe it does not exist?')
        return False
    sys_id = record[0]['sys_id']
    response = client.update({'approval': state}, sys_id)
    return response

def delete_record(table, sys_id):
    if False:
        while True:
            i = 10
    '\n    Delete an existing record\n\n    :param table: The table name, e.g. sys_user\n    :type  table: ``str``\n\n    :param sys_id: The unique ID of the record\n    :type  sys_id: ``str``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion servicenow.delete_record sys_computer 2134566\n    '
    client = _get_client()
    client.table = table
    response = client.delete(sys_id)
    return response

def non_structured_query(table, query=None, **kwargs):
    if False:
        print('Hello World!')
    "\n    Run a non-structed (not a dict) query on a servicenow table.\n    See http://wiki.servicenow.com/index.php?title=Encoded_Query_Strings#gsc.tab=0\n    for help on constructing a non-structured query string.\n\n    :param table: The table name, e.g. sys_user\n    :type  table: ``str``\n\n    :param query: The query to run (or use keyword arguments to filter data)\n    :type  query: ``str``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion servicenow.non_structured_query sys_computer 'role=web'\n        salt myminion servicenow.non_structured_query sys_computer role=web type=computer\n    "
    client = _get_client()
    client.table = table
    if query is None:
        query_parts = []
        for (key, value) in kwargs.items():
            query_parts.append('{}={}'.format(key, value))
        query = '^'.join(query_parts)
    query = str(query)
    response = client.get(query)
    return response

def update_record_field(table, sys_id, field, value):
    if False:
        while True:
            i = 10
    "\n    Update the value of a record's field in a servicenow table\n\n    :param table: The table name, e.g. sys_user\n    :type  table: ``str``\n\n    :param sys_id: The unique ID of the record\n    :type  sys_id: ``str``\n\n    :param field: The new value\n    :type  field: ``str``\n\n    :param value: The new value\n    :type  value: ``str``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion servicenow.update_record_field sys_user 2348234 first_name jimmy\n    "
    client = _get_client()
    client.table = table
    response = client.update({field: value}, sys_id)
    return response