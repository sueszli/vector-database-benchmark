"""Describes a way to submit a key to a key server.
"""
from __future__ import absolute_import
from __future__ import division
import os
from pwnlib.args import args
from pwnlib.log import getLogger
from pwnlib.tubes.remote import remote
env_server = args.get('FLAG_HOST', 'flag-submission-server').strip()
env_port = args.get('FLAG_PORT', '31337').strip()
env_file = args.get('FLAG_FILE', '/does/not/exist').strip()
env_exploit_name = args.get('EXPLOIT_NAME', 'unnamed-exploit').strip()
env_target_host = args.get('TARGET_HOST', 'unknown-target').strip()
env_team_name = args.get('TEAM_NAME', 'unknown-team').strip()
log = getLogger(__name__)

def submit_flag(flag, exploit=env_exploit_name, target=env_target_host, server=env_server, port=env_port, team=env_team_name):
    if False:
        for i in range(10):
            print('nop')
    "\n    Submits a flag to the game server\n\n    Arguments:\n        flag(str): The flag to submit.\n        exploit(str): Exploit identifier, optional\n        target(str): Target identifier, optional\n        server(str): Flag server host name, optional\n        port(int): Flag server port, optional\n        team(str): Team identifier, optional\n\n    Optional arguments are inferred from the environment,\n    or omitted if none is set.\n\n    Returns:\n        A string indicating the status of the key submission,\n        or an error code.\n\n    Doctest:\n\n        >>> l = listen()\n        >>> _ = submit_flag('flag', server='localhost', port=l.lport)\n        >>> c = l.wait_for_connection()\n        >>> c.recvall().split()\n        [b'flag', b'unnamed-exploit', b'unknown-target', b'unknown-team']\n    "
    flag = flag.strip()
    log.success('Flag: %r' % flag)
    data = '\n'.join([flag, exploit, target, team, '']).encode('ascii')
    if os.path.exists(env_file):
        write(env_file, data)
        return
    try:
        with remote(server, int(port)) as r:
            r.send(data)
            return r.recvall(timeout=1)
    except Exception:
        log.warn('Could not submit flag %r to %s:%s', flag, server, port)