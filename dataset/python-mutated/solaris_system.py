"""
Support for reboot, shutdown, etc

This module is assumes we are using solaris-like shutdown

.. versionadded:: 2016.3.0
"""
import salt.utils.path
import salt.utils.platform
__virtualname__ = 'system'

def __virtual__():
    if False:
        return 10
    '\n    Only supported on Solaris-like systems\n    '
    if not salt.utils.platform.is_sunos() or not salt.utils.path.which('shutdown'):
        return (False, 'The system execution module failed to load: only available on Solaris-like ystems with shutdown command.')
    return __virtualname__

def halt():
    if False:
        return 10
    "\n    Halt a running system\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' system.halt\n    "
    return shutdown()

def init(state):
    if False:
        i = 10
        return i + 15
    "\n    Change the system runlevel on sysV compatible systems\n\n    CLI Example:\n\n    state : string\n        Init state\n\n    .. code-block:: bash\n\n        salt '*' system.init 3\n\n    .. note:\n\n        state 0\n            Stop the operating system.\n\n        state 1\n            State 1 is referred to as the administrative state. In\n            state 1 file systems required for multi-user operations\n            are mounted, and logins requiring access to multi-user\n            file systems can be used. When the system comes up from\n            firmware mode into state 1, only the console is active\n            and other multi-user (state 2) services are unavailable.\n            Note that not all user processes are stopped when\n            transitioning from multi-user state to state 1.\n\n        state s, S\n            State s (or S) is referred to as the single-user state.\n            All user processes are stopped on transitions to this\n            state. In the single-user state, file systems required\n            for multi-user logins are unmounted and the system can\n            only be accessed through the console. Logins requiring\n            access to multi-user file systems cannot be used.\n\n       state 5\n            Shut the machine down so that it is safe to remove the\n            power. Have the machine remove power, if possible. The\n            rc0 procedure is called to perform this task.\n\n       state 6\n             Stop the operating system and reboot to the state defined\n             by the initdefault entry in /etc/inittab. The rc6\n             procedure is called to perform this task.\n    "
    cmd = ['shutdown', '-i', state, '-g', '0', '-y']
    ret = __salt__['cmd.run'](cmd, python_shell=False)
    return ret

def poweroff():
    if False:
        return 10
    "\n    Poweroff a running system\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' system.poweroff\n    "
    return shutdown()

def reboot(delay=0, message=None):
    if False:
        i = 10
        return i + 15
    '\n    Reboot the system\n\n    delay : int\n        Optional wait time in seconds before the system will be rebooted.\n    message : string\n        Optional message to broadcast before rebooting.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' system.reboot\n        salt \'*\' system.reboot 60 "=== system upgraded ==="\n    '
    cmd = ['shutdown', '-i', '6', '-g', delay, '-y']
    if message:
        cmd.append(message)
    ret = __salt__['cmd.run'](cmd, python_shell=False)
    return ret

def shutdown(delay=0, message=None):
    if False:
        while True:
            i = 10
    '\n    Shutdown a running system\n\n    delay : int\n        Optional wait time in seconds before the system will be shutdown.\n    message : string\n        Optional message to broadcast before rebooting.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' system.shutdown\n        salt \'*\' system.shutdown 60 "=== disk replacement ==="\n    '
    cmd = ['shutdown', '-i', '5', '-g', delay, '-y']
    if message:
        cmd.append(message)
    ret = __salt__['cmd.run'](cmd, python_shell=False)
    return ret