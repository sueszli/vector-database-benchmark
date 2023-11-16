"""
Return salt data to Zabbix

The following Type: "Zabbix trapper" with "Type of information" Text items are required:

.. code-block:: cfg

    Key: salt.trap.info
    Key: salt.trap.warning
    Key: salt.trap.high

To use the Zabbix returner, append '--return zabbix' to the salt command. ex:

.. code-block:: bash

    salt '*' test.ping --return zabbix
"""
import os
__virtualname__ = 'zabbix'
__deprecated__ = (3009, 'zabbix', 'https://github.com/salt-extensions/saltext-zabbix')

def __virtual__():
    if False:
        return 10
    if zbx():
        return True
    return (False, 'Zabbix returner: No zabbix_sender and zabbix_agend.conf found.')

def zbx():
    if False:
        while True:
            i = 10
    if os.path.exists('/usr/local/zabbix/bin/zabbix_sender') and os.path.exists('/usr/local/zabbix/etc/zabbix_agentd.conf'):
        zabbix_sender = '/usr/local/zabbix/bin/zabbix_sender'
        zabbix_config = '/usr/local/zabbix/etc/zabbix_agentd.conf'
        return {'sender': zabbix_sender, 'config': zabbix_config}
    elif os.path.exists('/usr/bin/zabbix_sender') and os.path.exists('/etc/zabbix/zabbix_agentd.conf'):
        zabbix_sender = '/usr/bin/zabbix_sender'
        zabbix_config = '/etc/zabbix/zabbix_agentd.conf'
        return {'sender': zabbix_sender, 'config': zabbix_config}
    else:
        return False

def zabbix_send(key, output):
    if False:
        while True:
            i = 10
    cmd = zbx()['sender'] + ' -c ' + zbx()['config'] + ' -k ' + key + ' -o "' + output + '"'
    __salt__['cmd.shell'](cmd)

def save_load(jid, load, minions=None):
    if False:
        while True:
            i = 10
    '\n    Included for API consistency\n    '

def returner(ret):
    if False:
        print('Hello World!')
    changes = False
    errors = False
    job_minion_id = ret['id']
    if type(ret['return']) is dict:
        for (state, item) in ret['return'].items():
            if 'comment' in item and 'name' in item and (item['result'] is False):
                errors = True
                zabbix_send('salt.trap.high', 'SALT:\nname: {}\ncomment: {}'.format(item['name'], item['comment']))
            elif 'comment' in item and 'name' in item and item['changes']:
                changes = True
                zabbix_send('salt.trap.warning', 'SALT:\nname: {}\ncomment: {}'.format(item['name'], item['comment']))
    if not changes and (not errors):
        zabbix_send('salt.trap.info', f'SALT {job_minion_id} OK')