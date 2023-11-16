from __future__ import annotations
import subprocess
from ansible import constants as C
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.compat.paramiko import paramiko
from ansible.utils.display import Display
display = Display()
_HAS_CONTROLPERSIST = {}

def check_for_controlpersist(ssh_executable):
    if False:
        print('Hello World!')
    try:
        return _HAS_CONTROLPERSIST[ssh_executable]
    except KeyError:
        pass
    b_ssh_exec = to_bytes(ssh_executable, errors='surrogate_or_strict')
    has_cp = True
    try:
        cmd = subprocess.Popen([b_ssh_exec, '-o', 'ControlPersist'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (out, err) = cmd.communicate()
        if b'Bad configuration option' in err or b'Usage:' in err:
            has_cp = False
    except OSError:
        has_cp = False
    _HAS_CONTROLPERSIST[ssh_executable] = has_cp
    return has_cp

def set_default_transport():
    if False:
        for i in range(10):
            print('nop')
    if C.DEFAULT_TRANSPORT == 'smart':
        display.deprecated("The 'smart' option for connections is deprecated. Set the connection plugin directly instead.", version='2.20')
        if not check_for_controlpersist('ssh') and paramiko is not None:
            C.DEFAULT_TRANSPORT = 'paramiko'
        else:
            C.DEFAULT_TRANSPORT = 'ssh'