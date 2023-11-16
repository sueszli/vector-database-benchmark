"""
macOS implementations of various commands in the "desktop" interface
"""
import salt.utils.platform
from salt.exceptions import CommandExecutionError
__virtualname__ = 'desktop'

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Only load on Mac systems\n    '
    if salt.utils.platform.is_darwin():
        return __virtualname__
    return (False, 'Cannot load macOS desktop module: This is not a macOS host.')

def get_output_volume():
    if False:
        return 10
    "\n    Get the output volume (range 0 to 100)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' desktop.get_output_volume\n    "
    cmd = 'osascript -e "get output volume of (get volume settings)"'
    call = __salt__['cmd.run_all'](cmd, output_loglevel='debug', python_shell=False)
    _check_cmd(call)
    return call.get('stdout')

def set_output_volume(volume):
    if False:
        return 10
    "\n    Set the volume of sound.\n\n    volume\n        The level of volume. Can range from 0 to 100.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' desktop.set_output_volume <volume>\n    "
    cmd = 'osascript -e "set volume output volume {}"'.format(volume)
    call = __salt__['cmd.run_all'](cmd, output_loglevel='debug', python_shell=False)
    _check_cmd(call)
    return get_output_volume()

def screensaver():
    if False:
        while True:
            i = 10
    "\n    Launch the screensaver.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' desktop.screensaver\n    "
    cmd = 'open /System/Library/Frameworks/ScreenSaver.framework/Versions/A/Resources/ScreenSaverEngine.app'
    call = __salt__['cmd.run_all'](cmd, output_loglevel='debug', python_shell=False)
    _check_cmd(call)
    return True

def lock():
    if False:
        return 10
    "\n    Lock the desktop session\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' desktop.lock\n    "
    cmd = '/System/Library/CoreServices/Menu\\ Extras/User.menu/Contents/Resources/CGSession -suspend'
    call = __salt__['cmd.run_all'](cmd, output_loglevel='debug', python_shell=False)
    _check_cmd(call)
    return True

def say(*words):
    if False:
        for i in range(10):
            print('nop')
    "\n    Say some words.\n\n    words\n        The words to execute the say command with.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' desktop.say <word0> <word1> ... <wordN>\n    "
    cmd = 'say {}'.format(' '.join(words))
    call = __salt__['cmd.run_all'](cmd, output_loglevel='debug', python_shell=False)
    _check_cmd(call)
    return True

def _check_cmd(call):
    if False:
        i = 10
        return i + 15
    '\n    Check the output of the cmd.run_all function call.\n    '
    if call['retcode'] != 0:
        comment = ''
        std_err = call.get('stderr')
        std_out = call.get('stdout')
        if std_err:
            comment += std_err
        if std_out:
            comment += std_out
        raise CommandExecutionError('Error running command: {}'.format(comment))
    return call