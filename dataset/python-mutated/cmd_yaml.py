"""
Execute a command and read the output as YAML. The YAML data is then directly overlaid onto the minion's Pillar data
"""
import logging
import salt.utils.yaml
log = logging.getLogger(__name__)

def ext_pillar(minion_id, pillar, command):
    if False:
        i = 10
        return i + 15
    '\n    Execute a command and read the output as YAML\n    '
    try:
        command = command.replace('%s', minion_id)
        output = __salt__['cmd.run_stdout'](command, python_shell=True)
        return salt.utils.yaml.safe_load(output)
    except Exception:
        log.critical("YAML data from '%s' failed to parse. Command output:\n%s", command, output)
        return {}