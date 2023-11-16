"""
Execute a command and read the output as JSON. The JSON data is then directly overlaid onto the minion's Pillar data.


Configuring the CMD_JSON ext_pillar
====================================

Set the following Salt config to setup cmd json result as external pillar source:

.. code-block:: yaml

  ext_pillar:
    - cmd_json: 'echo {"arg":"value"}'

This will run the command ``echo {arg: value}`` on the master.


Module Documentation
====================

"""
import logging
import salt.utils.json
log = logging.getLogger(__name__)

def ext_pillar(minion_id, pillar, command):
    if False:
        return 10
    '\n    Execute a command and read the output as JSON\n    '
    try:
        command = command.replace('%s', minion_id)
        return salt.utils.json.loads(__salt__['cmd.run'](command))
    except Exception:
        log.critical('JSON data from %s failed to parse', command)
        return {}