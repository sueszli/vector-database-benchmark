"""
Execution module for creating shortcuts on Windows. Handles file shortcuts
(`.lnk`) and url shortcuts (`.url`). Allows for the configuration of icons and
hot keys on file shortcuts. Changing the icon and hot keys are unsupported for
url shortcuts.

.. versionadded:: 3005
"""
import logging
import os
import time
import salt.utils.path
import salt.utils.platform
import salt.utils.winapi
from salt.exceptions import CommandExecutionError
HAS_WIN32 = False
if salt.utils.platform.is_windows():
    import win32com.client
    HAS_WIN32 = True
log = logging.getLogger(__name__)
__virtualname__ = 'shortcut'
WINDOW_STYLE = {1: 'Normal', 3: 'Maximized', 7: 'Minimized', 'Normal': 1, 'Maximized': 3, 'Minimized': 7}

def __virtual__():
    if False:
        return 10
    "\n    Make sure we're on Windows\n    "
    if not salt.utils.platform.is_windows():
        log.debug('Shortcut module only available on Windows systems')
        return (False, 'Shortcut module only available on Windows systems')
    if not HAS_WIN32:
        log.debug('Shortcut module requires pywin32')
        return (False, 'Shortcut module requires pywin32')
    return __virtualname__

def get(path):
    if False:
        return 10
    '\n    Gets the properties for a shortcut\n\n    Args:\n        path (str): The path to the shortcut. Must have a `.lnk` or `.url` file\n            extension.\n\n    Returns:\n        dict: A dictionary containing all available properties for the specified\n            shortcut\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt * shortcut.get path="C:\\path\\to\\shortcut.lnk"\n    '
    if not os.path.exists(path):
        raise CommandExecutionError('Shortcut not found: {}'.format(path))
    if not path.endswith(('.lnk', '.url')):
        (_, ext) = os.path.splitext(path)
        raise CommandExecutionError('Invalid file extension: {}'.format(ext))
    with salt.utils.winapi.Com():
        shell = win32com.client.Dispatch('WScript.Shell')
        shortcut = shell.CreateShortcut(path)
        arguments = ''
        description = ''
        hot_key = ''
        icon_location = ''
        icon_index = 0
        window_style = ''
        working_dir = ''
        path = salt.utils.path.expand(shortcut.FullName)
        if path.endswith('.lnk'):
            target = shortcut.TargetPath
            if target:
                target = salt.utils.path.expand(target)
            else:
                msg = 'Not a valid shortcut: {}'.format(path)
                log.debug(msg)
                raise CommandExecutionError(msg)
            if shortcut.Arguments:
                arguments = shortcut.Arguments
            if shortcut.Description:
                description = shortcut.Description
            if shortcut.Hotkey:
                hot_key = shortcut.Hotkey
            if shortcut.IconLocation:
                (icon_location, icon_index) = shortcut.IconLocation.split(',')
                if icon_location:
                    icon_location = salt.utils.path.expand(icon_location)
            if shortcut.WindowStyle:
                window_style = WINDOW_STYLE[shortcut.WindowStyle]
            if shortcut.WorkingDirectory:
                working_dir = salt.utils.path.expand(shortcut.WorkingDirectory)
        else:
            target = shortcut.TargetPath
        return {'arguments': arguments, 'description': description, 'hot_key': hot_key, 'icon_index': int(icon_index), 'icon_location': icon_location, 'path': path, 'target': target, 'window_style': window_style, 'working_dir': working_dir}

def _set_info(path, target='', arguments='', description='', hot_key='', icon_index=0, icon_location='', window_style='Normal', working_dir=''):
    if False:
        for i in range(10):
            print('nop')
    '\n    The main worker function for creating and modifying shortcuts. the `create`\n    and `modify` functions are wrappers around this function.\n\n    Args:\n\n        path (str): The full path to the shortcut\n\n        target (str): The full path to the target\n\n        arguments (str, optional): Any arguments to be passed to the target\n\n        description (str, optional): The description for the shortcut. This is\n            shown in the ``Comment`` field of the dialog box. Default is an\n            empty string\n\n        hot_key (str, optional): A combination of hot Keys to trigger this\n            shortcut. This is something like ``Ctrl+Alt+D``. This is shown in\n            the ``Shortcut key`` field in the dialog box. Default is an empty\n            string. Available options are:\n\n            - Ctrl\n            - Alt\n            - Shift\n            - Ext\n\n        icon_index (int, optional): The index for the icon to use in files that\n            contain multiple icons. Default is 0\n\n        icon_location (str, optional): The full path to a file containing icons.\n            This is shown in the ``Change Icon`` dialog box by clicking the\n            ``Change Icon`` button. If no file is specified and a binary is\n            passed as the target, Windows will attempt to get the icon from the\n            binary file. Default is an empty string\n\n        window_style (str, optional): The window style the program should start\n            in. This is shown in the ``Run`` field of the dialog box. Default is\n            ``Normal``. Valid options are:\n\n            - Normal\n            - Minimized\n            - Maximized\n\n        working_dir (str, optional): The full path to the working directory for\n            the program to run in. This is shown in the ``Start in`` field of\n            the dialog box.\n\n    Returns:\n        bool: True if successful\n    '
    path = salt.utils.path.expand(path)
    with salt.utils.winapi.Com():
        shell = win32com.client.Dispatch('WScript.Shell')
        shortcut = shell.CreateShortcut(path)
        if path.endswith('.lnk'):
            if target:
                target = salt.utils.path.expand(target)
            if arguments:
                shortcut.Arguments = arguments
            if description:
                shortcut.Description = description
            if hot_key:
                shortcut.Hotkey = hot_key
            if icon_location:
                shortcut.IconLocation = ','.join([icon_location, str(icon_index)])
            if window_style:
                shortcut.WindowStyle = WINDOW_STYLE[window_style]
            if working_dir:
                shortcut.WorkingDirectory = working_dir
        shortcut.TargetPath = target
        shortcut.Save()
    return True

def modify(path, target='', arguments='', description='', hot_key='', icon_index=0, icon_location='', window_style='Normal', working_dir=''):
    if False:
        print('Hello World!')
    '\n    Modify an existing shortcut. This can be a file shortcut (``.lnk``) or a\n    url shortcut (``.url``).\n\n    Args:\n\n        path (str): The full path to the shortcut. Must have a `.lnk` or `.url`\n            file extension.\n\n        target (str, optional): The full path to the target\n\n        arguments (str, optional): Any arguments to be passed to the target\n\n        description (str, optional): The description for the shortcut. This is\n            shown in the ``Comment`` field of the dialog box. Default is an\n            empty string\n\n        hot_key (str, optional): A combination of hot Keys to trigger this\n            shortcut. This is something like ``Ctrl+Alt+D``. This is shown in\n            the ``Shortcut key`` field in the dialog box. Default is an empty\n            string. Available options are:\n\n            - Ctrl\n            - Alt\n            - Shift\n            - Ext\n\n        icon_index (int, optional): The index for the icon to use in files that\n            contain multiple icons. Default is 0\n\n        icon_location (str, optional): The full path to a file containing icons.\n            This is shown in the ``Change Icon`` dialog box by clicking the\n            ``Change Icon`` button. If no file is specified and a binary is\n            passed as the target, Windows will attempt to get the icon from the\n            binary file. Default is an empty string\n\n        window_style (str, optional): The window style the program should start\n            in. This is shown in the ``Run`` field of the dialog box. Default is\n            ``Normal``. Valid options are:\n\n            - Normal\n            - Minimized\n            - Maximized\n\n        working_dir (str, optional): The full path to the working directory for\n            the program to run in. This is shown in the ``Start in`` field of\n            the dialog box.\n\n    Returns:\n        bool: True if successful\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Modify an existing shortcut. Set it to target notepad.exe\n        salt * shortcut.modify "C:\\path\\to\\shortcut.lnk" "C:\\Windows\\notepad.exe"\n    '
    if not os.path.exists(path):
        raise CommandExecutionError('Shortcut not found: {}'.format(path))
    if not path.endswith(('.lnk', '.url')):
        (_, ext) = os.path.splitext(path)
        raise CommandExecutionError('Invalid file extension: {}'.format(ext))
    return _set_info(path=path, arguments=arguments, description=description, hot_key=hot_key, icon_index=icon_index, icon_location=icon_location, target=target, window_style=window_style, working_dir=working_dir)

def create(path, target, arguments='', description='', hot_key='', icon_index=0, icon_location='', window_style='Normal', working_dir='', backup=False, force=False, make_dirs=False, user=None):
    if False:
        while True:
            i = 10
    '\n    Create a new shortcut. This can be a file shortcut (``.lnk``) or a url\n    shortcut (``.url``).\n\n    Args:\n\n        path (str): The full path to the shortcut. Must have a `.lnk` or `.url`\n            file extension.\n\n        target (str): The full path to the target\n\n        arguments (str, optional): Any arguments to be passed to the target\n\n        description (str, optional): The description for the shortcut. This is\n            shown in the ``Comment`` field of the dialog box. Default is an\n            empty string\n\n        hot_key (str, optional): A combination of hot Keys to trigger this\n            shortcut. This is something like ``Ctrl+Alt+D``. This is shown in\n            the ``Shortcut key`` field in the dialog box. Default is an empty\n            string. Available options are:\n\n            - Ctrl\n            - Alt\n            - Shift\n            - Ext\n\n        icon_index (int, optional): The index for the icon to use in files that\n            contain multiple icons. Default is 0\n\n        icon_location (str, optional): The full path to a file containing icons.\n            This is shown in the ``Change Icon`` dialog box by clicking the\n            ``Change Icon`` button. If no file is specified and a binary is\n            passed as the target, Windows will attempt to get the icon from the\n            binary file. Default is an empty string\n\n        window_style (str, optional): The window style the program should start\n            in. This is shown in the ``Run`` field of the dialog box. Default is\n            ``Normal``. Valid options are:\n\n            - Normal\n            - Minimized\n            - Maximized\n\n        working_dir (str, optional): The full path to the working directory for\n            the program to run in. This is shown in the ``Start in`` field of\n            the dialog box.\n\n        backup (bool, optional): If there is already a shortcut with the same\n            name, set this value to ``True`` to backup the existing shortcut and\n            continue creating the new shortcut. Default is ``False``\n\n        force (bool, optional): If there is already a shortcut with the same\n            name and you aren\'t backing up the shortcut, set this value to\n            ``True`` to remove the existing shortcut and create a new with these\n            settings. Default is ``False``\n\n        make_dirs (bool, optional): If the parent directory structure does not\n            exist for the new shortcut, create it. Default is ``False``\n\n        user (str, optional): The user to be the owner of any directories\n            created by setting ``make_dirs`` to ``True``. If no value is passed\n            Salt will use the user account that it is running under. Default is\n            an empty string.\n\n    Returns:\n        bool: True if successful\n\n    Raises:\n        CommandExecutionError: If the path is not a ``.lnk`` or ``.url`` file\n            extension.\n        CommandExecutionError: If there is an existing shortcut with the same\n            name and ``backup`` and ``force`` are both ``False``\n        CommandExecutionError: If the parent directory is not created and\n            ``make_dirs`` is ``False``\n        CommandExecutionError: If there was an error creating the parent\n            directories\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        # Create a shortcut and set the ``Shortcut key`` (``hot_key``)\n        salt * shortcut.create "C:\\path\\to\\shortcut.lnk" "C:\\Windows\\notepad.exe" hot_key="Ctrl+Alt+N"\n\n        # Create a shortcut and change the icon to the 3rd one in the icon file\n        salt * shortcut.create "C:\\path\\to\\shortcut.lnk" "C:\\Windows\\notepad.exe" icon_location="C:\\path\\to\\icon.ico" icon_index=2\n\n        # Create a shortcut and change the startup mode to full screen\n        salt * shortcut.create "C:\\path\\to\\shortcut.lnk" "C:\\Windows\\notepad.exe" window_style="Maximized"\n\n        # Create a shortcut and change the icon\n        salt * shortcut.create "C:\\path\\to\\shortcut.lnk" "C:\\Windows\\notepad.exe" icon_location="C:\\path\\to\\icon.ico"\n\n        # Create a shortcut and force it to overwrite an existing shortcut\n        salt * shortcut.create "C:\\path\\to\\shortcut.lnk" "C:\\Windows\\notepad.exe" force=True\n\n        # Create a shortcut and create any parent directories if they are missing\n        salt * shortcut.create "C:\\path\\to\\shortcut.lnk" "C:\\Windows\\notepad.exe" make_dirs=True\n    '
    if not path.endswith(('.lnk', '.url')):
        (_, ext) = os.path.splitext(path)
        raise CommandExecutionError('Invalid file extension: {}'.format(ext))
    if os.path.exists(path):
        if backup:
            log.debug('Backing up: %s', path)
            (file, ext) = os.path.splitext(path)
            ext = ext.strip('.')
            backup_path = '{}-{}.{}'.format(file, time.time_ns(), ext)
            os.rename(path, backup_path)
        elif force:
            log.debug('Removing: %s', path)
            os.remove(path)
        else:
            log.debug('Shortcut exists: %s', path)
            raise CommandExecutionError('Found existing shortcut')
    if not os.path.isdir(os.path.dirname(path)):
        if make_dirs:
            if not user:
                user = __opts__['user']
            if not __salt__['user.info'](user):
                user = __salt__['user.current']()
                if not user:
                    user = 'SYSTEM'
            try:
                __salt__['file.makedirs'](path=path, owner=user)
            except CommandExecutionError as exc:
                raise CommandExecutionError('Error creating parent directory: {}'.format(exc.message))
        else:
            raise CommandExecutionError('Parent directory not present: {}'.format(os.path.dirname(path)))
    return _set_info(path=path, arguments=arguments, description=description, hot_key=hot_key, icon_index=icon_index, icon_location=icon_location, target=target, window_style=window_style, working_dir=working_dir)